#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build-note-dataset: 从「人类 4K 谱 + 音频 mel」建「每格有没有音符/几个音符」的数据集。

思路：把 GJ 里手写的采音阈值换成从 36837 张人类谱学出来的模型。
每个 1/16 格（按该谱自己的 BPM）抽一组特征，标签是该格人类谱放了几个音符。

特征（每格）：
  - 8 个 mel 频带能量的 z 分数（按歌曲内归一）
  - 4 个频带的正向 flux 的 z 分数
  - 总能量 / 总 flux 的 z 分数
  - 本格在 16 分位置上的 one-hot（2 维 sin/cos + 4 维拍点强度）
  - 前后 8 格的能量均值 z（局部密度）
  - 距上一个 flux 峰的距离（格数，截断到 8）

用法：
  python3 tools/build-note-dataset.py --limit 1200 -o out/note-dataset.npz
"""
import argparse
import glob
import json
import os
import random

import numpy as np

FRAME_MS = 2048 / 22050 / 2 * 1000.0   # 46.44ms


def parse_osu_full(path):
    """返回 (offset_ms, beat_ms, notes[(t_ms, col, is_ln)])."""
    off = None
    beat = None
    notes = []
    sec = None
    for ln in open(path, encoding='utf-8', errors='ignore'):
        ln = ln.strip()
        if ln.startswith('['):
            sec = ln
            continue
        if sec == '[TimingPoints]' and ln and not ln.startswith('//'):
            p = ln.split(',')
            if len(p) >= 2:
                t = float(p[0])
                b = float(p[1])
                if b > 0 and beat is None:      # 第一条非继承点
                    off, beat = t, b
        elif sec == '[HitObjects]' and ln and not ln.startswith('//'):
            p = ln.split(',')
            if len(p) < 4:
                continue
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
            col = 0 if x < 128 else (1 if x < 256 else (2 if x < 384 else 3))
            notes.append((t, col, 1 if (typ & 128) else 0))
    return off, beat, notes


def chart_features(mel, offset, beat_ms, n_cells, band_edges):
    """mel [128,F] → 每格特征 [n_cells, K] 与标签占位。"""
    logm = np.log1p(np.maximum(mel, 0))
    # 频带能量与 flux
    band = np.stack([logm[a:b].mean(0) for a, b in band_edges])          # [B,F]
    flux = np.maximum(np.diff(band, axis=1, prepend=band[:, :1]), 0)     # [B,F]
    step_ms = beat_ms / 4.0
    F = mel.shape[1]
    cell_of_frame = ((np.arange(F) * FRAME_MS - offset) / step_ms)
    # 每格的帧范围（粗分箱，允许空）
    idx = np.floor(cell_of_frame).astype(int)
    ok = (idx >= 0) & (idx < n_cells)
    # 8 频带能量 + 8 频带 flux + 总能量 + 总flux + 局部能量 + 6 位置特征 + 距峰距离
    feats = np.zeros((n_cells, band.shape[0] * 2 + 3 + 6 + 1), dtype=np.float16)
    cnt = np.zeros(n_cells, dtype=np.float32)

    def agg(arr):
        out = np.zeros((n_cells, arr.shape[0]), dtype=np.float32)
        for b in range(arr.shape[0]):
            np.add.at(out[:, b], idx[ok], arr[b][ok])
        return out

    e = agg(band)
    f = agg(flux)
    np.add.at(cnt, idx[ok], 1)
    cnt = np.maximum(cnt, 1)
    e /= cnt[:, None]
    f /= cnt[:, None]

    def z(x):
        m, s = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
        return (x - m) / s

    ez, fz = z(e), z(f)
    tot_e = z(e.mean(1, keepdims=True))
    tot_f = z(f.mean(1, keepdims=True))
    # 局部能量（±8 格）
    k = np.ones(17) / 17
    loc = np.convolve(tot_e[:, 0], k, mode='same')
    loc = z(loc[:, None])
    # 位置特征
    pos = np.arange(n_cells) % 16
    strong = np.array([3.0, 1.0, 2.6, 1.0, 2.2, 1.0, 2.4, 1.0,
                       3.0, 1.6, 2.6, 1.0, 1.8, 1.0, 1.8, 1.0], dtype=np.float32)
    pfeat = np.stack([np.sin(2 * np.pi * pos / 16), np.cos(2 * np.pi * pos / 16),
                      np.sin(2 * np.pi * pos / 4), np.cos(2 * np.pi * pos / 4),
                      strong[pos] / 3.0, (pos % 4 == 0).astype(np.float32)], 1)
    # 距上一个 flux 峰
    fp = np.zeros(n_cells, dtype=np.float32)
    acc = 8.0
    for c in range(n_cells):
        acc = 0.0 if tot_f[c, 0] > 0.8 else min(8.0, acc + 1.0)
        fp[c] = acc / 8.0
    feats[:, :ez.shape[1]] = ez.astype(np.float16)
    feats[:, ez.shape[1]:ez.shape[1] + fz.shape[1]] = fz.astype(np.float16)
    base = ez.shape[1] + fz.shape[1]
    feats[:, base] = tot_e[:, 0]
    feats[:, base + 1] = tot_f[:, 0]
    feats[:, base + 2] = loc[:, 0]
    feats[:, base + 3:base + 9] = pfeat
    feats[:, base + 9] = fp
    # 上下文：前后 1/2 格的总能量与 flux（节奏型是局部的，单格太吵）
    te = tot_e[:, 0]
    tf = tot_f[:, 0]
    ctx = np.zeros((n_cells, 4), dtype=np.float32)
    ctx[1:, 0] = te[:-1]
    ctx[:-1, 1] = te[1:]
    ctx[2:, 2] = te[:-2]
    ctx[:-2, 3] = te[2:]
    ctf = np.zeros((n_cells, 4), dtype=np.float32)
    ctf[1:, 0] = tf[:-1]
    ctf[:-1, 1] = tf[1:]
    ctf[2:, 2] = tf[:-2]
    ctf[:-2, 3] = tf[2:]
    feats = np.concatenate([feats, ctx.astype(np.float16), ctf.astype(np.float16)], axis=1)
    return feats, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('-o', '--out', default='out/note-dataset.npz')
    args = ap.parse_args()

    feats_db = {}
    for ln in open('out/feature-db.jsonl', encoding='utf-8'):
        d = json.loads(ln)
        feats_db[d['id']] = d

    mel_files = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob('out/audio_features/*.npz')}
    charts = {os.path.splitext(os.path.basename(p))[0]: p
              for p in glob.glob('out/charts/*.osu')}
    ids = [i for i in mel_files if i in charts]
    random.Random(args.seed).shuffle(ids)
    print(f'可用配对 {len(ids)} 首，取 {args.limit}')

    band_edges = [(0, 16), (16, 32), (32, 48), (48, 64),
                  (64, 80), (80, 96), (96, 112), (112, 128)]
    X, y_size, y_meta, y_chart = [], [], [], []
    used = 0
    for cid in ids[:args.limit]:
        try:
            off, beat, notes = parse_osu_full(charts[cid])
            if off is None or beat is None or beat < 60 or beat > 2000 or not notes:
                continue
            mel = np.load(mel_files[cid])['mel']
            dur_ms = mel.shape[1] * FRAME_MS
            end = min(max(t for t, c, l in notes), dur_ms - 200)
            n_cells = int((end - off) / (beat / 4)) - 1
            if n_cells < 200 or n_cells > 20000:
                continue
            fx, cnt = chart_features(mel, off, beat, n_cells,
                                     [(a, b) for a, b in band_edges])
            sz = np.zeros(n_cells, dtype=np.int8)
            step = beat / 4
            for t, col, is_ln in notes:
                ci = int((t - off) / step)
                if 0 <= ci < n_cells:
                    sz[ci] = min(4, sz[ci] + 1)
            X.append(fx)
            y_size.append(sz)
            y_meta.append(np.full(n_cells, float(feats_db.get(cid, {}).get('overall_ett', 0)), dtype=np.float32))
            y_chart.append(np.full(n_cells, used, dtype=np.int16))
            used += 1
        except Exception as e:
            continue
    X = np.concatenate(X)
    y = np.concatenate(y_size)
    m = np.concatenate(y_meta)
    ch = np.concatenate(y_chart)
    np.savez_compressed(args.out, X=X, y=y, ett=m, chart=ch)
    print(f'写了 {args.out}: {X.shape[0]} 格 × {X.shape[1]} 特征，{used} 首谱')
    nz = (y > 0).mean()
    print(f'  正样本比例 {nz:.3f}  音符数分布 {np.bincount(y, minlength=5).tolist()}')


if __name__ == '__main__':
    main()
