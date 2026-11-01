#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单首歌 → **32 带 mel onset 通量**条件（S1 采样器用），口径与
`tools/build-mel-cond.py` 逐字段一致。

链路（与 build-mel-cond.py 同一套代码，不是「差不多」）：
  load_mono（tools/build-audio-cond-fixed.py 的 22050 单声道解码）
  → melspectrogram(n_mels=32, n_fft=1024, hop_length=256, sr=22050,
                   fmin=0, fmax=8000, center=True)
  → log1p
  → 沿时间**正向差分** max(Δ,0)   ← onset 通量，不是能量
        （能量版逐谱相关只有 0.287，通量版 0.523；见 build-mel-cond 的说明）
  → 按 1/32 格（step = 60000/bpm/8 ms，格起点 = --offset，默认 0）取格内**均值**
  → 逐通道 P99 归一到 [0,1]（P99 取「有效格」= 格起点落在音频时长内的那些格）

mel 帧时刻：center=True → 第 j 帧中心 = j*hop 采样 = j*HOP/SR*1000 ms，
格 k 覆盖 [q+k*step, q+(k+1)*step)，取格内帧均值（4 带版是取 max，这里按任务书用均值）。

输出 npz 键
-----------
  M        [N,32] float16   条件（P99 归一后，0..1）
  start_ms [N]     float32  每格的绝对起点 ms（= offset + k*step），对齐核查用
  bpm / step_ms / offset_ms / duration_ms / n_cells / feature / agg
         标量（np.float64 / np.int64），采样器不依赖它们，只作审计信息
  valid_cells  有效格数（格起点在音频时长内）
  threshold_hint  首帧/尾帧中位？不写；保持最小

用法
----
  python3 tools/make-mel-cond.py \
      --audio "ref/multisong/Heart Fraction.ogg" --bpm 136.3 \
      --out out/diffusion/round154-sampler/Heart-Fraction-mel.npz

  # 若这张谱的第一条红线不在 0（offset ≠ 0），把谱面 offset 传进来：
  #   --offset 1234.5
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- mel 参数（与 build-mel-cond.py 完全一致）----
SR = 22050
MEL_N_FFT = 1024
MEL_HOP = 256
N_MELS = 32
N_BANDS = 32                      # S1 的 cond_channels


def load_module(rel, name):
    p = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BFX = load_module('tools/build-audio-cond-fixed.py', 'bfx')


def onset_flux(L):
    """log1p mel → 正向时间差分（onset 通量）。首帧补 0 保持帧对齐。"""
    return np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0.0).astype(np.float32)


def cell_mean(mel, tf, q_ms, step_ms, n, mode='mean'):
    """格内均值/最大：[n,32]，格 k = [q+k*step, q+(k+1)*step)。与 build-mel-cond 同代码。"""
    n = int(n)
    ch = mel.shape[0]
    out = np.zeros((n, ch), dtype=np.float32)
    F = mel.shape[1]
    if n <= 0 or F == 0:
        return out
    t0 = q_ms + np.arange(n, dtype=np.float64) * step_ms
    i0 = np.searchsorted(tf, t0, side='left').astype(np.int64)
    i1 = np.searchsorted(tf, t0 + step_ms, side='left').astype(np.int64)
    i0 = np.clip(i0, 0, F)
    i1 = np.clip(i1, 0, F)
    ok = i1 > i0
    if not ok.any():
        return out
    if mode == 'max':
        for k in np.flatnonzero(ok):
            out[k] = mel[:, i0[k]:i1[k]].max(axis=1)
        return out
    cs = np.concatenate([np.zeros((ch, 1), np.float64),
                         np.cumsum(mel.astype(np.float64), axis=1)], axis=1)
    seg = (cs[:, i1[ok]] - cs[:, i0[ok]]) / (i1[ok] - i0[ok])[None, :]
    out[ok] = seg.T.astype(np.float32)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='单首歌 → 32 带 mel onset 通量条件')
    ap.add_argument('--audio', required=True, help='音频文件（ogg/wav/mp3）')
    ap.add_argument('--bpm', type=float, required=True, help='BPM（1/32 格 = 60000/bpm/8 ms）')
    ap.add_argument('--out', required=True, help='输出 npz')
    ap.add_argument('--offset', type=float, default=0.0,
                    help='网格起点 ms（= 谱面第一条红线 offset；默认 0）')
    ap.add_argument('--feature', choices=['flux', 'energy'], default='flux',
                    help='flux=log1p mel 正向差分（默认，onset）；energy=log1p mel 本身（对照用）')
    ap.add_argument('--agg', choices=['mean', 'max'], default='mean', help='格内聚合（默认 mean）')
    ap.add_argument('--max-seconds', type=float, default=0.0, help='只取前 N 秒（调试用，0=全曲）')
    a = ap.parse_args(argv)

    if not (a.bpm > 0):
        ap.error('--bpm 必须 > 0')
    if not os.path.exists(a.audio):
        raise SystemExit(f'[错误] 音频不存在：{a.audio}')
    step_ms = 60000.0 / float(a.bpm) / 8.0
    t_all = time.time()

    import librosa
    y = BFX.load_mono(a.audio)
    if y is None:
        raise SystemExit(f'[错误] 解码失败：{a.audio}')
    if a.max_seconds > 0:
        y = y[:int(round(a.max_seconds * SR))]
    dur_ms = len(y) / SR * 1000.0

    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=MEL_N_FFT,
                                       hop_length=MEL_HOP, n_mels=N_MELS,
                                       fmin=0.0, fmax=8000.0, center=True)
    L = np.log1p(S).astype(np.float32)
    F = L.shape[1]
    tf = (np.arange(F, dtype=np.float64) * MEL_HOP) / SR * 1000.0
    del S
    mel = onset_flux(L) if a.feature == 'flux' else L

    q = float(a.offset)
    if q < 0 or q >= 60000.0 / float(a.bpm):
        raise SystemExit(f'[错误] --offset {q} 应在 [0, 一拍 {60000.0/a.bpm:.1f}) ms 内')
    lim = min(dur_ms, float(tf[-1]))
    n_cells = int(np.floor((lim - q) / step_ms)) + 1
    if n_cells <= 0:
        raise SystemExit(f'[错误] 音频太短：{dur_ms:.0f}ms < 1 格 {step_ms:.2f}ms')
    raw = cell_mean(mel, tf, q, step_ms, n_cells, a.agg)      # [n_cells,32]

    t0 = q + np.arange(n_cells, dtype=np.float64) * step_ms
    valid = (t0 >= 0.0) & (t0 < lim)
    p = np.percentile(raw[valid], 99, axis=0) if valid.any() else np.ones(N_MELS)
    scale = np.where(p > 1e-9, p, 1.0)
    M = np.clip(raw / scale[None, :], 0.0, 1.0).astype(np.float16)

    # ---- 自证（不用谱面也能查的几条）----
    Mf = M.astype(np.float32)
    tot = Mf.sum(axis=1)
    env = np.maximum(librosa.onset.onset_strength(y=y, sr=SR, hop_length=MEL_HOP), 0)
    # 条件总强度 vs 单路 onset 包络（同一 1/32 格）的逐格相关：口径一致性检查
    e_cell = cell_mean(env[None, :], tf, q, step_ms, n_cells, 'mean')[:, 0]
    r_env = BFX.pearson(tot[valid], e_cell[valid])
    # 相邻格自相关（通量条件的相位锐度直觉）：lag1
    r_lag1 = BFX.pearson(tot[valid][1:], tot[valid][:-1]) if valid.sum() > 8 else 0.0

    stats = dict(
        audio=a.audio, audio_seconds=round(dur_ms / 1000.0, 2), sr=SR,
        mel=dict(n_mels=N_MELS, n_fft=MEL_N_FFT, hop=MEL_HOP, sr=SR,
                 fmin=0.0, fmax=8000.0, center=True, feature=a.feature,
                 agg='cell-' + a.agg, norm='per-song-per-channel-P99',
                 clip=[0.0, 1.0]),
        bpm=float(a.bpm), step_ms=round(step_ms, 4), offset_ms=float(q),
        n_cells=int(n_cells), valid_cells=int(valid.sum()),
        duration_covered_ms=round(valid.sum() * step_ms, 1),
        M_shape=[int(n_cells), N_MELS], M_dtype='float16',
        M_stats=dict(mean=round(float(Mf.mean()), 5),
                     frac_gt0=round(float((Mf > 0).mean()), 4),
                     max=round(float(Mf.max()), 4),
                     p99_scale=[round(float(v), 6) for v in scale]),
        selfcheck=dict(r_total_vs_onset_strength=round(float(r_env), 4),
                       r_lag1=round(float(r_lag1), 4),
                       note='r_total_vs_onset_strength 只证明「格内聚合后仍与起音包络同向」，'
                            '不证明谱面对齐（谱面相关见 build-mel-cond.py 的逐谱自证）'),
        seconds=round(time.time() - t_all, 1))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    np.savez_compressed(a.out, M=M,
                        start_ms=(q + np.arange(n_cells) * step_ms).astype(np.float32),
                        bpm=np.float64(a.bpm), step_ms=np.float64(step_ms),
                        offset_ms=np.float64(q), duration_ms=np.float64(dur_ms),
                        n_cells=np.int64(n_cells), valid_cells=np.int64(valid.sum()),
                        feature=np.str_(a.feature), agg=np.str_(a.agg))
    json.dump(stats, open(os.path.splitext(a.out)[0] + '.json', 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'音频 {a.audio}：{dur_ms/1000:.1f}s，bpm {a.bpm} → 1/32 格 '
          f'{step_ms:.2f}ms，{n_cells} 格（有效 {valid.sum()}）')
    print(f'条件 {tuple(M.shape)} {M.dtype}：均值 {stats["M_stats"]["mean"]:.4f}，'
          f'>0 占 {stats["M_stats"]["frac_gt0"]*100:.1f}%，'
          f'通道 P99 中位 {np.median(scale):.4f}')
    print(f'自证：条件总强度 vs onset 包络 r={r_env:+.3f}（同格聚合后同向），'
          f'lag1 r={r_lag1:+.3f}')
    print(f'写出 {a.out}（{os.path.getsize(a.out)/1e6:.2f}MB），用时 '
          f'{stats["seconds"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
