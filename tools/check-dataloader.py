#!/usr/bin/env python3
"""最小 dataloader 验收：确认特征能组成训练 batch。

    python3 tools/check-dataloader.py out/features --batch 8

不是训练脚本，只是证明「数据能被正确读进来、堆成 batch、形状对得上」。
训练前先跑这个，比训到第 100 步发现维度错了省事得多。
"""
import argparse
import json
import os

import numpy as np


def load_one(root, row, max_frames=8192):
    with np.load(os.path.join(root, row['audio'])) as z:
        mel = z['mel'].astype(np.float32)[:, :max_frames]      # [128, T]
        onset = z['onset'].astype(np.float32)[:max_frames]
    with np.load(os.path.join(root, row['chart'])) as z:
        chart = z['tensor'].astype(np.float32)                  # [4K, 8192]
        valid = z['valid'].astype(np.float32)
        keys = int(z['keys'])

    mel_p = np.zeros((128, max_frames), dtype=np.float32)
    mel_p[:, :mel.shape[1]] = mel
    onset_p = np.zeros((max_frames,), dtype=np.float32)
    onset_p[:onset.shape[0]] = onset
    return mel_p, onset_p, chart, valid, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='out/features')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--keys', type=int, default=4, help='只取这个键数（batch 要求同 K）')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.root, 'index.jsonl'),
                                        encoding='utf-8')]
    rows = [r for r in rows if r['keys'] == args.keys]
    print(f'可用样本 {len(rows)}（{args.keys}K）')
    if len(rows) < args.batch:
        rows = rows or []

    sample = rows[:args.batch]
    mels, onsets, charts, valids = [], [], [], []
    for r in sample:
        m, o, c, v, k = load_one(args.root, r)
        mels.append(m)
        onsets.append(o)
        charts.append(c)
        valids.append(v)

    mel_b = np.stack(mels)
    onset_b = np.stack(onsets)
    chart_b = np.stack(charts)
    valid_b = np.stack(valids)

    print(f'\nbatch 形状：')
    print(f'  mel    {mel_b.shape}  {mel_b.dtype}')
    print(f'  onset  {onset_b.shape}  {onset_b.dtype}')
    print(f'  chart  {chart_b.shape}  {chart_b.dtype}')
    print(f'  valid  {valid_b.shape}  {valid_b.dtype}')

    print(f'\n统计：')
    print(f'  mel     均值 {mel_b.mean():.2f}  标准差 {mel_b.std():.2f}  '
          f'范围 [{mel_b.min():.2f}, {mel_b.max():.2f}]')
    print(f'  chart   非零率 {(chart_b > 0).mean()*100:.2f}%  '
          f'（start 通道稀疏度）')
    print(f'  valid   平均有效帧 {valid_b.mean()*8192:.0f} / 8192 '
          f'({valid_b.mean()*100:.0f}%)')

    # start 通道只应该有 0/1
    k = args.keys
    starts = chart_b[:, :k]
    uniq = np.unique(starts)
    print(f'  start 通道唯一值 {uniq[:6]}{" ..." if len(uniq) > 6 else ""}')

    # offset 通道应该在 [0,1]
    offs = chart_b[:, k:2 * k]
    nz = offs[offs > 0]
    print(f'  offset 通道 范围 [{nz.min():.3f}, {nz.max():.3f}]  '
          f'均值 {nz.mean():.3f}  （应在 0~1）')

    # LN 通道
    ln = chart_b[:, 2 * k:3 * k]
    print(f'  LN 体通道 非零率 {(ln > 0).mean()*100:.2f}%')

    # 简单的内存估算
    per_sample = mel_b[0].nbytes + chart_b[0].nbytes
    print(f'\n单样本内存 {per_sample/1024:.0f} KB，'
          f'batch={args.batch} 约 {per_sample*args.batch/1024/1024:.1f} MB')

    print('\n✅ 可以组 batch，形状和值域都正常')


if __name__ == '__main__':
    main()
