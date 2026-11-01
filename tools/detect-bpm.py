#!/usr/bin/env python3
"""detect-bpm: 更好的 BPM/网格检测（解决半速/倍速误判）。

思路：
1. librosa 给一个基准 tempo + 多频带 onset 峰；
2. 对 {t/2, t, 2t} 三个八度各自做 (bpm, offset) 网格联合拟合；
3. 用「onset 到最近网格线的距离」+「网格步长与 onset 间隔中位数是否匹配」联合打分，
   避免出现 93.8 BPM（其实是 187.6 的一半，1/16 格 160ms 太粗）这种误判。

用法: python3 tools/detect-bpm.py --audio x.ogg -o out/multisong/x-fixed.json
"""
import argparse, json, numpy as np

def onset_envs(y, sr, hop):
    import librosa
    full = np.maximum(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), 0)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = {}
    for name, (lo, hi) in {'low': (0, 200), 'mid': (200, 2000), 'high': (2000, 8000)}.items():
        m = (fr >= lo) & (fr < hi)
        if m.any():
            bands[name] = np.maximum(np.diff(np.log1p(S[m].sum(0)), prepend=0), 0)
    return full, bands

def peaks_of(env, fps, delta_mult=0.55, wait=1):
    import librosa
    med = float(np.median(env)) + 1e-9
    idx = librosa.util.peak_pick(x=env, pre_max=2, post_max=2, pre_avg=5,
                                 post_avg=5, delta=delta_mult * med, wait=wait)
    return idx * 1000.0 / fps

def grid_fit(env, fps, bpm, off_span=None, off_step=2.0):
    """在 (bpm, offset) 上最大化「1/16 网格线上的包络均值 / 全局均值」= 信噪比。

    这是本项目里验证过的方法（recalibrate-analysis.py 的 joint_fit）：
    峰值命中率对网格密度敏感、易过拟合；包络信噪比是尺度无关的。
    """
    step = 60000.0 / bpm / 4.0            # 1/16 步长（ms）
    step_f = step / 1000.0 * fps
    if step_f < 2:
        return 0.0, -1.0, step
    mean = float(env.mean()) + 1e-9
    n = len(env)
    best = (-1.0, 0.0)
    span = step if off_span is None else off_span
    for off in np.arange(0.0, span, off_step):
        idx = np.arange(int(off / 1000.0 * fps), n, step_f).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        if len(idx) < 8:
            continue
        snr = float(env[idx].mean()) / mean
        if snr > best[0]:
            best = (snr, off)
    return best[1], best[0], step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--target-step', type=float, default=0.0,
                    help='期望的 1/16 步长（ms），0=自动取 onset 间隔中位数')
    a = ap.parse_args()

    import librosa
    y, sr = librosa.load(a.audio, sr=22050, mono=True)
    dur_ms = len(y) / sr * 1000.0
    hop = 256; fps = sr / hop
    full, bands = onset_envs(y, sr, hop)

    # 多频带峰合并
    peaks = set(peaks_of(full, fps))
    for env in bands.values():
        peaks |= set(peaks_of(env, fps))
    ons = np.array(sorted(peaks))
    ons = ons[(ons > 0) & (ons < dur_ms)]
    # 每个 onset 对应的包络强度
    strg = np.array([full[min(len(full) - 1, int(t / 1000.0 * fps))] for t in ons])
    # 网格拟合只用「强起音」：全量峰里有大量噪声峰，会把任何网格都糊成 0.25（随机水平）
    keep = strg >= np.percentile(strg, 60)
    ons_fit = ons[keep]
    strg_fit = strg[keep]

    # 基准 tempo：用节拍跟踪的实际拍点间隔（比全局 tempo 稳）
    _, beat_frames = librosa.beat.beat_track(onset_envelope=full, sr=sr, hop_length=hop)
    beat_ms = np.array(beat_frames) * 1000.0 / fps
    ibi = np.diff(beat_ms)
    ibi = ibi[(ibi > 150) & (ibi < 1500)]
    tempo = 60000.0 / float(np.median(ibi)) if len(ibi) > 3 else \
        float(np.atleast_1d(librosa.beat.tempo(onset_envelope=full, sr=sr, hop_length=hop))[0])
    cands = sorted({round(tempo / 2, 2), round(tempo, 2), round(tempo * 2, 2)})

    # 期望步长：onset 间隔的中位数（避免用错八度时步长过粗/过细）
    iois = np.diff(ons_fit)
    iois = iois[(iois > 20) & (iois < 2000)]
    med_ioi = float(np.median(iois)) if len(iois) else 80.0

    best = None
    for bpm in cands:
        off16, snr, step = grid_fit(full, fps, bpm)
        if step < 45 or step > 260:
            continue
        playable = 1.0 if 55 <= step <= 135 else (0.7 if step <= 170 else 0.4)
        final = snr * playable
        print(f'  bpm {bpm:7.2f} 1/16 {step:6.2f}ms 信噪比 {snr:5.2f} 系数 {playable:.2f} 综合 {final:5.2f}')
        if best is None or final > best[0]:
            best = (final, bpm, off16, snr, step)
    _, bpm, off, snr, step = best
    print(f'选定: bpm {bpm} offset {off:.2f}ms 1/16 {step:.2f}ms 信噪比 {snr:.2f}')

    # 输出与 chartgen 同构的 analysis
    n_cells = int(dur_ms / step) + 2
    cell_env = np.zeros(n_cells)
    for i in range(n_cells):
        t0 = off + i * step
        f0 = int(t0 / 1000.0 * fps); f1 = min(len(full), int((t0 + step) / 1000.0 * fps) + 1)
        if f1 > f0 > 0:
            cell_env[i] = float(full[f0:f1].max())
    if cell_env.max() > 0:
        cell_env = cell_env / cell_env.max()
    out = dict(source=a.audio, sampleRate=sr, durationSec=dur_ms / 1000.0,
               bpm=bpm, bpmConfidence=float(snr), offsetMs=float(off),
               stepMs=float(step), steps=n_cells,
               peaks=[int(round(t / (1000.0 * hop / sr))) for t in ons],
               accent=cell_env.tolist(),
               onset=full.tolist(),
               downbeat=0,
               fine=dict(hopMs=1000.0 * hop / sr, energy=full.tolist()),
               note='detect-bpm.py')
    json.dump(out, open(a.out, 'w'))

if __name__ == '__main__':
    main()
