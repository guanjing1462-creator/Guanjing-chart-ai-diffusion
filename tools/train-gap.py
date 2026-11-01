#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train-gap: 「站在一个音符上，下一个音符隔几格」——用人类谱学节奏间隔。

这是要替掉 GJ 里手写的「onset 锚点 + 补音倍数」那套：
出谱时从一个音符出发，让模型给出下一个音符的间隔，走完全曲。

输入特征（每格）：build-note-dataset.py 产出的 34 维音频特征
  + 当前距上一个音符的格数（截断 8）
  + 本格在 16 分内的位置
标签：到下一个音符的格数（1..16，>16 截断到 16），只在有音符的格子上训练。

用法：
  python3 tools/train-gap.py --data out/note-dataset6k.npz --epochs 10 -o out/gap.pt
"""
import argparse

import numpy as np
import torch
import torch.nn as nn


class GapNet(nn.Module):
    def __init__(self, d_in, hidden=128, n_cls=17):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        )
        self.out = nn.Linear(hidden, n_cls)

    def forward(self, x):
        return self.out(self.body(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/note-dataset6k.npz')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('-o', '--out', default='out/gap.pt')
    args = ap.parse_args()

    d = np.load(args.data)
    X0 = d['X']                 # float16 [N,34]
    y = d['y']
    ch = d['chart']

    # 按谱算「到下一个音符的格数」
    gap = np.zeros(len(y), dtype=np.int64)
    order = np.argsort(ch, kind='stable')
    ys, cs = y[order], ch[order]
    starts = np.searchsorted(cs, np.unique(cs))
    starts = list(starts) + [len(cs)]
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        seg = ys[a:b]
        nz = np.nonzero(seg)[0]
        if len(nz) == 0:
            continue
        for j in range(len(nz) - 1):
            g = nz[j + 1] - nz[j]
            gap[order[a + nz[j]]] = min(16, g)
        gap[order[a + nz[-1]]] = 16
    mask = y > 0
    print(f'有音符的格子 {int(mask.sum())} / {len(y)}；间隔分布 '
          f'{np.bincount(gap[mask], minlength=17)[1:].tolist()}')

    # 额外特征：位置（16 分内）已含在 X 里；再加「距上一个音符」
    prev_gap = np.zeros(len(y), dtype=np.float32)
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        seg = ys[a:b]
        nz = np.nonzero(seg)[0]
        prev = 8
        for j in range(b - a):
            if j in set(nz.tolist()):
                prev = 0
            else:
                prev = min(8, prev + 1)
            prev_gap[a + j] = prev / 8.0
    # 难度也作为条件：高难谱的间隔分布和中低难完全不同
    ett = d['ett'].astype(np.float32)
    e_norm = np.clip(ett, 0, 40) / 40.0
    X = np.concatenate([X0, prev_gap[:, None].astype(np.float16),
                        e_norm[:, None].astype(np.float16),
                        (e_norm ** 2)[:, None].astype(np.float16)], axis=1)

    idx = np.nonzero(mask)[0]
    rng = np.random.RandomState(5)
    rng.shuffle(idx)
    n_val = int(len(idx) * 0.12)
    va, tr = idx[:n_val], idx[n_val:]
    print(f'训练 {len(tr)} 验证 {len(va)}')

    net = GapNet(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best = 1e9
    for ep in range(args.epochs):
        net.train()
        perm = np.random.RandomState(ep).permutation(len(tr))
        tot = n = 0
        for i in range(0, len(perm), args.batch):
            b = tr[perm[i:i + args.batch]]
            xb = torch.from_numpy(X[b].astype(np.float32))
            lb = torch.from_numpy(gap[b])
            loss = ce(net(xb), lb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
        net.eval()
        with torch.no_grad():
            acc = 0
            top3 = 0
            tot_n = 0
            for i in range(0, len(va), 16384):
                b = va[i:i + 16384]
                lo = net(torch.from_numpy(X[b].astype(np.float32)))
                pred = lo.argmax(1).numpy()
                lab = gap[b]
                acc += int((pred == lab).sum())
                top3 += int((lo.topk(3, dim=1).indices.numpy() == lab[:, None]).any(1).sum())
                tot_n += len(b)
        print(f'epoch {ep}: loss {tot/n:.4f}  acc {acc/tot_n:.3f}  top3 {top3/tot_n:.3f}')
        if tot / n < best:
            best = tot / n
            torch.save({'state': net.state_dict(), 'd_in': X.shape[1],
                        'meta': {'acc': acc / tot_n, 'top3': top3 / tot_n}}, args.out)
    print('saved →', args.out)


if __name__ == '__main__':
    main()
