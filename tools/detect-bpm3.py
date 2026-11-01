#!/usr/bin/env python3
"""BPM 检测 v3：自相关定周期 + 梳状对齐精修 + 八度裁决（全向量化，秒级）。

为什么不用 v1 的「网格信噪比」：
  v1 在真实歌曲上会偏掉（Heart Fraction 真值 128，v1 给 136.3，+6.5%），
  因为它是「在 1/16 网格上取包络最大值再算 SNR」，对同步切分（syncopation）不敏感，
  而且八度靠「步长舒适度」这种启发式。

v3 的做法：
  1. 多频带 onset 包络（低/中/高三段相加）。
  2. 对包络做 FFT 自相关 → 在 60~220 BPM 对应的 lag 范围内找峰（这是节奏检测的标准做法，
     对「强弱拍交替」不敏感，只关心周期性）。
  3. 每个候选周期用**梳状滤波**对齐：在 0..period 内扫 offset，取拍点窗口内的最大包络均值，
     并惩罚空拍（预测有拍却没有起音）—— 空拍惩罚专门用来排除倍速。
  4. 八度裁决：半速/倍速评分接近时，优先 [95,200] 常规范围。
  5. 最后 ±0.6 BPM 内 0.01 步长精修。
"""
import argparse
import json
import numpy as np


def onset_envelope(y, sr, hop):
    """多频带 onset 包络：三个频带各自求 onset strength 再相加。"""
    import librosa
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    env = np.zeros(S.shape[1], dtype=np.float64)
    for lo, hi in ((0, 200), (200, 2000), (2000, 8000)):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        band = S[m].sum(axis=0)
        d = np.diff(np.log1p(band), prepend=0.0)
        env += np.maximum(d, 0.0)
    env = np.maximum(env, 0)
    if env.max() > 0:
        env = env / env.max()
    return env


def periodicity(env, fps, bpm_lo=55.0, bpm_hi=225.0, n_fft=None):
    """FFT 自相关，返回 (lags, ac)。lag 单位：帧。"""
    n = len(env)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(env, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:n]
    ac = ac / (ac[0] + 1e-12)
    lag_lo = int(fps * 60.0 / bpm_hi)
    lag_hi = int(fps * 60.0 / bpm_lo)
    lag_lo = max(2, lag_lo)
    lag_hi = min(len(ac) - 1, lag_hi)
    lags = np.arange(lag_lo, lag_hi + 1)
    return lags, ac[lags]


def comb_score(env_max, fps, bpm, win_ms=30.0):
    """梳状对齐评分：扫 offset，返回最佳 (score, offset_ms)。

    env_max 是「窗口内最大值」的预计算结果，所以这里每个候选只要一次索引。
    """
    period = 60000.0 / bpm
    dur_ms = len(env_max) / fps * 1000.0
    n_beat = int(dur_ms / period) - 1
    if n_beat < 8:
        return -1e9, 0.0
    best = (-1e9, 0.0)
    # offset 粗扫 12 个点即可（后面有精修）
    for k in range(12):
        off = period * k / 12.0
        idx = ((off + period * np.arange(n_beat)) / 1000.0 * fps).astype(np.int32)
        idx = idx[(idx >= 0) & (idx < len(env_max))]
        if len(idx) < 8:
            continue
        v = env_max[idx]
        hit = float(v.mean())
        empty = float((v < 0.25 * (v.max() + 1e-9)).mean())
        s = hit - 0.8 * empty
        if s > best[0]:
            best = (s, off)
    return best


def refine(env_max, fps, bpm0, span=0.8, step=0.01):
    best = (-1e9, bpm0, 0.0)
    b = max(40.0, bpm0 - span)
    while b <= bpm0 + span:
        s, off = comb_score(env_max, fps, b)
        if s > best[0]:
            best = (s, b, off)
        b += step
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--bpm-only', action='store_true')
    a = ap.parse_args()

    import librosa
    from scipy.ndimage import maximum_filter1d

    y, sr = librosa.load(a.audio, sr=11025, mono=True)
    dur_ms = len(y) / sr * 1000.0
    hop = 128
    fps = sr / hop
    env = onset_envelope(y, sr, hop)

    # 拍点窗口（±30ms）内的包络最大值，一次算好
    win = max(3, int(0.030 * fps))
    env_max = maximum_filter1d(env, size=win)

    # 1) 自相关找候选周期
    lags, ac = periodicity(env, fps)
    # 取局部峰
    cand_bpm = []
    for i in range(1, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.12:
            cand_bpm.append((ac[i], 60000.0 * fps / (60.0 * lags[i] * 1.0) if False else 60.0 * fps / lags[i]))
    cand_bpm.sort(key=lambda x: -x[0])
    # 取前 8 个不同 BPM（间隔 >2）
    picked = []
    for strength, b in cand_bpm:
        if all(abs(b - p) > 2.0 for _, p in picked):
            picked.append((strength, b))
        if len(picked) >= 8:
            break
    if not picked:
        picked = [(1.0, 120.0)]

    # 2) 每个候选（含 1/2、2×）做梳状评分
    scored = []
    seen = set()
    for _, b0 in picked:
        for f in (0.5, 1.0, 2.0):
            b = b0 * f
            if not (55.0 <= b <= 225.0):
                continue
            key = round(b)
            if key in seen:
                continue
            seen.add(key)
            s, off = comb_score(env_max, fps, b)
            scored.append((s, b, off))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        scored = [(0.0, 120.0, 0.0)]

    # 3) 八度裁决：评分接近时优先常规范围
    best_s = scored[0][0]
    ok = [c for c in scored if c[0] >= 0.85 * best_s]
    in_range = [c for c in ok if 95.0 <= c[1] <= 200.0]
    pool = in_range if in_range else (ok if ok else scored[:1])
    pool.sort(key=lambda x: -x[0])

    # 4) 精修
    finals = []
    for s, b, off in pool[:3]:
        rs, rb, ro = refine(env_max, fps, b)
        finals.append((rs, rb, ro))
    finals.sort(key=lambda x: -x[0])
    s, bpm, off = finals[0]

    step = 60000.0 / bpm / 4.0
    for c in finals[:3]:
        print(f'  候选 bpm {c[1]:8.2f} 评分 {c[0]:.3f}', flush=True)
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
