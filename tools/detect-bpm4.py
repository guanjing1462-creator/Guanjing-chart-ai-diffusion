#!/usr/bin/env python3
"""BPM 检测 v4：相位直方图（circular）扫描，全向量化。

原理（最贴合「写谱」需求）：
  一首歌的正确 BPM，会让「所有起音时间对拍周期取模」得到的相位高度聚集。
  所以对每个 BPM 候选，计算相位分布集中度（circular concentration），
  取集中度最高者。这直接优化「音符能不能落在格子上」，比自相关更贴近用途。

流程：
  1. 多频带包络 → 起音峰（带强度）。
  2. 60~220 BPM 以 0.1 步长扫：对每个 bpm 用 rfft 求相位集中度（向量化，毫秒级）。
  3. 取前 5 名，各自在 ±0.5 BPM 内以 0.01 精修。
  4. 八度裁决：1/2、2× 也参与评分；接近时优先 [95,200]。
"""
import argparse
import json
import numpy as np


def onsets_of(audio, sr=11025, hop=128):
    import librosa
    y, _ = librosa.load(audio, sr=sr, mono=True)
    n_fft = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    env = np.zeros(S.shape[1])
    for lo, hi in ((0, 200), (200, 2000), (2000, 5000)):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        band = S[m].sum(axis=0)
        env += np.maximum(np.diff(np.log1p(band), prepend=0.0), 0.0)
    env = np.maximum(env, 0)
    fps = sr / hop
    dur_ms = len(y) / sr * 1000.0
    # 拾峰
    pk = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]))[0] + 1
    pk = pk.astype(np.int64)
    if len(pk) == 0:
        return np.array([]), env, fps, dur_ms
    times = pk * 1000.0 / fps
    strength = env[pk]
    thr = np.percentile(strength, 55)
    keep = strength >= thr
    return times[keep], env, fps, dur_ms


def phase_concentration(times_ms, bpm, bins=48):
    """相位集中度：把起音时间对拍周期取模，做相位直方图，取归一化后的峰值。"""
    period = 60000.0 / bpm
    ph = (times_ms % period) / period
    h, _ = np.histogram(ph, bins=bins, range=(0, 1))
    h = h.astype(np.float64)
    if h.sum() <= 0:
        return 0.0, 0.0
    h /= h.sum()
    # 用圆均值长度作为集中度（对相位平移不敏感）
    ang = 2 * np.pi * (np.arange(bins) + 0.5) / bins
    r = np.hypot((h * np.cos(ang)).sum(), (h * np.sin(ang)).sum())
    # 叠加峰占比（真正落在同一窄相位桶里的比例）
    peak = h.max() * bins / 4.0
    return float(r), float(peak)


def scan(times_ms, lo=60.0, hi=220.0, step=0.1, bins=48):
    """向量化相位扫描：把时间对每个周期取模，一次性算圆均值与峰占比。"""
    bpms = np.arange(lo, hi + 1e-9, step)
    periods = 60000.0 / bpms
    out = np.empty(len(bpms))
    chunk = 256
    for c0 in range(0, len(bpms), chunk):
        pp = periods[c0:c0 + chunk][:, None]              # (C,1)
        ph = (times_ms[None, :] % pp) / pp               # (C,N)
        idx = np.clip((ph * bins).astype(np.int32), 0, bins - 1)
        h = np.zeros((len(pp), bins))
        rows = np.repeat(np.arange(len(pp)), times_ms.size)
        flat = rows * bins + idx.ravel()
        h += np.bincount(flat, minlength=len(pp) * bins).reshape(len(pp), bins)
        tot = h.sum(axis=1, keepdims=True) + 1e-12
        h = h / tot
        ang = 2 * np.pi * (np.arange(bins) + 0.5) / bins
        r = np.hypot((h * np.cos(ang)).sum(1), (h * np.sin(ang)).sum(1))
        peak = h.max(axis=1) * bins / 4.0
        out[c0:c0 + len(pp)] = r * 0.5 + peak * 0.5
    return bpms, out


def refine(times_ms, bpm0, span=0.5, step=0.01):
    """窄范围精修（同样走向量化扫描）。"""
    bpms, sc = scan(times_ms, lo=max(40.0, bpm0 - span), hi=bpm0 + span, step=step)
    k = int(np.argmax(sc))
    return float(sc[k]), float(bpms[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--bpm-only', action='store_true')
    ap.add_argument('--bpm', type=float, default=0.0,
                    help='强制指定 BPM（音频被变速/升调时用：真值 128 而音频是 136 就填 128）')
    a = ap.parse_args()

    import librosa
    times, env, fps, dur_ms = onsets_of(a.audio)
    if len(times) < 20:
        print('起音太少，无法检测', flush=True)
        return
    if a.bpm > 0:
        bpm = float(a.bpm)
        period = 60000.0 / bpm
        ph = (times % period) / period
        h, edges = np.histogram(ph, bins=48, range=(0, 1))
        k = int(np.argmax(h))
        off = float((edges[k] + edges[k + 1]) / 2 * period)
        step = period / 4.0
        print(f'强制 BPM {bpm:.2f}（offset {off:.2f}ms 1/16 {step:.2f}ms）', flush=True)
        if a.bpm_only:
            return
        n_cells = int(dur_ms / step) + 2
        cell_env = np.zeros(n_cells)
        for i in range(n_cells):
            t0 = off + i * step
            f0 = int(t0 / 1000.0 * fps)
            f1 = min(len(env), int((t0 + step) / 1000.0 * fps) + 1)
            if f1 > f0 > 0:
                cell_env[i] = float(env[f0:f1].max())
        if cell_env.max() > 0:
            cell_env = cell_env / cell_env.max()
        json.dump(dict(path=a.audio, bpm=round(bpm, 3), offsetMs=round(off, 2),
                       stepMs=round(step, 3), grid='1/16',
                       cellEnv=[round(float(x), 4) for x in cell_env]), open(a.out, 'w'))
        return
    bpms, scores = scan(times)
    order = np.argsort(-scores)
    picked = []
    for i in order:
        b = float(bpms[i])
        if all(abs(b - p[1]) > 1.5 for p in picked):
            picked.append((float(scores[i]), b))
        if len(picked) >= 5:
            break

    # 八度候选
    octs = []
    for s0, b0 in picked[:3]:
        for f in (0.5, 1.0, 2.0):
            b = b0 * f
            if 55.0 <= b <= 230.0:
                octs.append(b)
    octs = sorted(set(round(b, 2) for b in octs))
    finals = []
    for b in octs:
        s, bb = refine(times, b)
        finals.append((s, bb))
    finals.sort(key=lambda x: -x[0])

    best_s = finals[0][0]
    ok = [c for c in finals if c[0] >= 0.88 * best_s]
    in_range = [c for c in ok if 95.0 <= c[1] <= 200.0]
    pool = in_range if in_range else ok
    pool.sort(key=lambda x: -x[0])
    score, bpm = pool[0]

    # offset：把相位直方图峰的中心作为 offset
    period = 60000.0 / bpm
    ph = (times % period) / period
    h, edges = np.histogram(ph, bins=48, range=(0, 1))
    k = int(np.argmax(h))
    off = float((edges[k] + edges[k + 1]) / 2 * period)
    step = period / 4.0
    for c in pool[:3]:
        print(f'  候选 bpm {c[1]:8.2f} 集中度 {c[0]:.4f}', flush=True)
    print(f'选定: bpm {bpm:.2f} offset {off:.2f}ms 1/16 {step:.2f}ms', flush=True)
    if a.bpm_only:
        return

    n_cells = int(dur_ms / step) + 2
    cell_env = np.zeros(n_cells)
    for i in range(n_cells):
        t0 = off + i * step
        f0 = int(t0 / 1000.0 * fps)
        f1 = min(len(env), int((t0 + step) / 1000.0 * fps) + 1)
        if f1 > f0 > 0:
            cell_env[i] = float(env[f0:f1].max())
    if cell_env.max() > 0:
        cell_env = cell_env / cell_env.max()
    json.dump(dict(path=a.audio, bpm=round(bpm, 3), offsetMs=round(off, 2),
                   stepMs=round(step, 3), grid='1/16',
                   cellEnv=[round(float(x), 4) for x in cell_env]), open(a.out, 'w'))


if __name__ == '__main__':
    main()
