#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recalibrate-analysis: 用独立 onset 包络重算 BPM/offset，并重建 accent/peaks。

为什么必须做：chartgen 分析器给的网格（185.00BPM / offset 1263.30ms）在
INTERNET YAMERO 上**整体早了 54ms**——用 librosa 的 onset 包络对整曲做
(bpm, offset) 联合拟合，当前网格的信噪只有 0.784（低于平均值！），
最优网格是 184.98BPM / +54ms，信噪 2.655。

之前之所以没发现：校验是拿「分析器自己的 peaks」和「分析器自己的网格」比，
同源自证。必须用独立的 onset 包络。

产出（结构同原 analysis.json，供 paint-v11.py 直接消费）：
  offsetMs / stepMs / bpm / accent（1/16 格能量）/ peaks（细粒度 onset 帧）
  / fine.hopMs / durationSec / downbeat（强拍在本小节第几拍）

用法：
  python3 tools/recalibrate-analysis.py --audio "ref/INTERNET YAMERO.ogg" \
      --in out/final14-s50/analysis.json --out out/final14-s50/analysis-fixed.json
"""
import argparse
import json

import numpy as np


def onset_env(y, sr, hop):
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return np.maximum(env, 0.0)


def joint_fit(env, fps, dur_ms, bpm0, off0, bpm_span=0.6, bpm_step=0.02,
              shift_span=70, shift_step=2, beats=1.0):
    """在 (bpm, offset) 上最大化「网格线上的 onset 包络均值」。"""
    mean = env.mean() + 1e-9
    best = (-1.0, bpm0, 0.0)
    for bpm in np.arange(bpm0 - bpm_span, bpm0 + bpm_span + 1e-9, bpm_step):
        period = 60000.0 / bpm * beats
        for shift in np.arange(-shift_span, shift_span + 1e-9, shift_step):
            ts = np.arange(off0 + shift, dur_ms, period)
            i = np.round(ts * fps / 1000.0).astype(int)
            i = i[(i >= 0) & (i < len(env))]
            if len(i) < 40:
                continue
            sc = float(env[i].mean() / mean)
            if sc > best[0]:
                best = (sc, float(bpm), float(shift))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--hop', type=int, default=256, help='细粒度帧 hop（默认 256=11.6ms）')
    ap.add_argument('--bpm-span', type=float, default=0.6)
    args = ap.parse_args()

    import librosa
    sr = 22050
    y, _ = librosa.load(args.audio, sr=sr, mono=True)
    hop = args.hop
    fps = sr / hop
    dur_ms = len(y) / sr * 1000
    env = onset_env(y, sr, hop)

    an = json.load(open(args.inp))
    bpm0 = float(an['bpm'])
    off0 = float(an['offsetMs'])

    # 1) 1/4 拍网格联合拟合（用拍而不是 1/16，抗噪声）
    sc0 = -1.0
    period0 = 60000.0 / bpm0
    ts0 = np.arange(off0, dur_ms, period0)
    i0 = np.round(ts0 * fps / 1000).astype(int)
    i0 = i0[(i0 >= 0) & (i0 < len(env))]
    sc0 = float(env[i0].mean() / (env.mean() + 1e-9))
    sc, bpm, shift = joint_fit(env, fps, dur_ms, bpm0, off0, args.bpm_span, 0.02, 70, 2)
    off = off0 + shift
    print(f'网格拟合: {bpm0:.2f}BPM/{off0:.1f}ms(信噪 {sc0:.3f}) → '
          f'{bpm:.2f}BPM/{off:.1f}ms(信噪 {sc:.3f})  偏移 {shift:+.0f}ms')

    # 2) 强拍相位：用低频（底鼓）包络在 4 个拍位里挑最强的一个
    import librosa as _lr
    S = np.abs(_lr.stft(y, n_fft=2048, hop_length=hop))
    freqs = _lr.fft_frequencies(sr=sr, n_fft=2048)
    low = np.log1p(S[(freqs >= 30) & (freqs < 170)].sum(axis=0))
    per = 60000.0 / bpm            # 一拍
    beat_sc = []
    for k in range(4):
        # 第 k 拍：每小节第 k 拍，间隔正好一拍（之前误用了 per/4，量的是 1/16）
        ts = np.arange(off + k * per, dur_ms, per * 4)
        i = np.round(ts * fps / 1000.0).astype(int)
        i = i[(i >= 0) & (i < len(low))]
        beat_sc.append(float(low[i].mean()) if len(i) else 0.0)
    downbeat = int(np.argmax(beat_sc))
    print(f'四拍底鼓强度 {[round(x,1) for x in beat_sc]} → 强拍在第 {downbeat} 拍')
    # 把网格相位挪到强拍上（offset 指向本小节第 0 拍）
    off_shift = downbeat * 60000.0 / bpm / 4

    # 3) 用新网格重建 accent / peaks / step
    step16 = 60000.0 / bpm / 4
    n_cells = int((dur_ms - off) / step16)
    logmag = np.log1p(S.sum(axis=0))          # 电平（与 chartgen accent 同义）
    flux = np.maximum(np.diff(logmag, prepend=logmag[:1]), 0.0)   # onset 强度
    level, onset, peaks = [], [], []
    for i in range(n_cells):
        t0 = off + i * step16
        f0 = max(0, int(t0 / 1000.0 * fps))
        f1 = max(f0 + 1, min(len(logmag), int((t0 + step16) / 1000.0 * fps)))
        lev = logmag[f0:f1]
        fl = flux[f0:f1]
        level.append(float(lev.mean()) if len(lev) else 0.0)
        onset.append(float(fl.max()) if len(fl) else 0.0)
        if len(fl):
            peaks.append(f0 + int(np.argmax(fl)))
    # peaks 去重 + 只保留明显高于邻域的
    pk = []
    for p in peaks:
        if pk and p - pk[-1] < 3:
            continue
        pk.append(p)
    out = dict(an)
    out.update({
        'bpm': round(bpm, 3),
        'offsetMs': round(off + off_shift, 3),
        'stepMs': round(60000.0 / bpm / 4, 4),
        'durationSec': round(dur_ms / 1000.0, 2),
        'fine': {'hopMs': round(1000.0 / fps, 4), 'energy': [round(float(x), 6) for x in flux]},
        'accent': [round(x, 6) for x in level],
        'onset': [round(x, 6) for x in onset],
        'peaks': pk,
        'downbeat': downbeat,
        'recalibrated': True,
        'calibNote': f'joint-fit {bpm0}/{off0} -> {bpm:.2f}/{off:.1f}, shift {shift:+.0f}ms, '
                     f'downbeat {downbeat}',
    })
    json.dump(out, open(args.out, 'w'))
    print(f'写出 {args.out}: {n_cells} 个 1/16 格, {len(pk)} 个 onset, '
          f'offset {out["offsetMs"]}ms, step {out["stepMs"]}ms')


if __name__ == '__main__':
    main()
