#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 cond-windows.npz 的 199 个窗口补上 style/diff 标签，并预切好音频条件。

为什么需要这一步：
  `out/diff-data/cond-windows.npz` 只有钢琴卷窗口 X、song、start，
  没有 diff（难度档）/style（风格）——而训练脚本要这两列。
  这 6 首歌不在 `out/packs/{tech,jack,speed,stamina}/` 里（不在全量语料的风格目录下），
  所以没有风格标签：统一记 style=0，**在报告里写清楚这是已知局限**。

产物 out/diff-data/audio-cond-labels.npz：
    X     [199,4,128] uint8   直接从 cond-windows.npz 搬过来（不改）
    style [199] uint8         6 首歌都没有风格标签 → 全 0
    diff  [199] uint8         按逐曲 nps 分档（<8→0, <12→1, <16→2, else 3，与 diff-data.py 一致）
    song  [199] int16         与 X 行对齐（来自 cond-windows.npz）
    start [199] int32         同上
    cond  [199,128,4] float32 预切好的音频条件（cond = feat[song][start:start+128]）
    nps   [6] float32         逐曲 nps（只作记录）

用法：
    python3 tools/build-audio-cond-labels.py
"""
import argparse
import json
import os

import numpy as np

WIN = 128
COND_ROOT = 'out/multisong'
CHART_DIR = 'out/multisong/charts'


def parse_roll(path):
    """与 tools/diff-data.py 的 parse_roll 一致：返回 [(time_ms, [cols])]。"""
    xs = {64: 0, 192: 1, 320: 2, 448: 3}
    by_t = {}
    for ln in open(path, encoding='utf-8', errors='ignore'):
        p = ln.split(',')
        if len(p) < 5:
            continue
        try:
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
        except ValueError:
            continue
        if x not in xs:
            continue
        if typ & 1 or typ & 128:
            by_t.setdefault(t, set()).add(xs[x])
    return sorted((t, sorted(c)) for t, c in by_t.items())


def diff_bucket(nps):
    return 0 if nps < 8 else 1 if nps < 12 else 2 if nps < 16 else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', default='out/diff-data/cond-windows.npz')
    ap.add_argument('--cond-root', default=COND_ROOT)
    ap.add_argument('--charts', default=CHART_DIR)
    ap.add_argument('--out', default='out/diff-data/audio-cond-labels.npz')
    a = ap.parse_args()

    w = np.load(a.windows, allow_pickle=True)
    X, song, start = w['X'], w['song'].astype(np.int16), w['start'].astype(np.int32)
    songs = [str(s) for s in w['songs']]
    N = len(X)

    # ---- 逐曲 nps → diff 档 ----
    nps, diff_of = [], []
    for name in songs:
        path = os.path.join(a.charts, f'{name}.osu')
        rows = parse_roll(path)
        span = (rows[-1][0] - rows[0][0]) / 1000.0
        v = len(rows) / max(1e-6, span)
        nps.append(v)
        diff_of.append(diff_bucket(v))
        print(f'  {name:42s} 行 {len(rows):5d}  时长 {span:6.1f}s  nps {v:5.2f} '
              f'→ diff {diff_of[-1]}')
    nps = np.array(nps, dtype=np.float32)
    diff_of = np.array(diff_of, dtype=np.uint8)

    D = diff_of[song]
    S = np.zeros(N, dtype=np.uint8)          # 6 首歌没有风格标签

    # ---- 预切条件 ----
    feats = {}
    for name in songs:
        feats[name] = np.load(os.path.join(a.cond_root, f'{name}-cond.npz'))['feat'].astype(np.float32)
    C = np.zeros((N, WIN, 4), dtype=np.float32)
    for i in range(N):
        f = feats[songs[int(song[i])]]
        st = int(start[i])
        seg = f[st:st + WIN]
        if len(seg) < WIN:
            raise SystemExit(f'[错误] 窗口 {i}（{songs[int(song[i])]} @{st}）越界')
        C[i] = seg

    # ---- 对齐自检：条件总 onset vs 窗口行占用率 ----
    row = (X.sum(axis=1) > 0).astype(np.float32)          # [N,128] 该格有音符
    onset = C[:, :, 3]
    cors = []
    for i in range(N):
        if row[i].std() > 1e-6 and onset[i].std() > 1e-6:
            cors.append(float(np.corrcoef(row[i], onset[i])[0, 1]))
    cors = np.array(cors)
    print(f'\n对齐自检（corr(总 onset, 该格有音符)）：均值 {cors.mean():.3f} '
          f'中位 {np.median(cors):.3f} 正相关窗口占 {(cors > 0).mean() * 100:.1f}%')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez_compressed(a.out, X=X, style=S, diff=D, song=song, start=start,
                        cond=C, nps=nps, songs=np.array(songs))
    print(f'\n窗口 {N}，diff 分布 {np.bincount(D, minlength=4).tolist()}'
          f'（按 nps 分档，bincount 顺序 0:<8 1:8~12 2:12~16 3:>16）')
    print(f'条件形状 {C.shape}，通道均值 {[round(float(v), 4) for v in C.mean(axis=(0, 1))]}')
    print(f'写出 → {a.out}')


if __name__ == '__main__':
    main()
