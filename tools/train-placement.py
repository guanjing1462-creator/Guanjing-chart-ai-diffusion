#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train-placement: 用人类谱数据集训练「这一格放不放音符 / 放几个」。

两个头：
  note 头：BCE，P(这一格有音符)
  size 头：5 类 softmax（0/1/2/3/4 个），用来决定单音还是和弦

按谱切分训练/验证（不是按格子切），避免同一首歌泄漏。

用法：
  python3 tools/train-placement.py --data out/note-dataset.npz --epochs 12 -o out/placement.pt
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, d_in, hidden=96):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.10),
        )
        self.note = nn.Linear(hidden, 1)
        self.size = nn.Linear(hidden, 5)

    def forward(self, x):
        h = self.body(x)
        return self.note(h).squeeze(-1), self.size(h)


def auc(y, p):
    o = np.argsort(p)
    r = np.empty_like(o, dtype=np.float64)
    r[o] = np.arange(1, len(p) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/note-dataset.npz')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('-o', '--out', default='out/placement.pt')
    args = ap.parse_args()

    d = np.load(args.data)
    X = d['X']            # 保持 float16，按 batch 转 float32（4.6M×34 的 float32 会吃 626MB）
    y = d['y'].astype(np.int64)
    ch = d['chart']
    ett = d['ett'].astype(np.float32)
    # 把难度拼进特征：同一首歌高难谱和低难谱的落点完全不同，
    # 不把难度给模型，标签本身就是噪声（首训 AUC 只有 0.69 就是这个原因）
    e_norm = np.clip(ett, 0, 40) / 40.0
    X = np.concatenate([X, e_norm[:, None], (e_norm ** 2)[:, None],
                        np.zeros((len(X), 0), dtype=np.float32)], axis=1)
    # 每格是否放音符 + 放几个
    yn = (y > 0).astype(np.float32)

    charts = np.unique(ch)
    rng = np.random.RandomState(3)
    rng.shuffle(charts)
    n_val = max(1, int(len(charts) * args.val_frac))
    val_ids = set(charts[:n_val].tolist())
    m_val = np.array([c in val_ids for c in ch])
    print(f'谱 {len(charts)}（验证 {n_val}） 格 {len(y)}  训练 {int((~m_val).sum())} 验证 {int(m_val.sum())}')

    Xt = X
    yt = torch.from_numpy(yn)
    st = torch.from_numpy(y)
    mt = torch.from_numpy(m_val)

    net = Net(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = int((~mt).sum()) // args.batch + 1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    idx_tr = torch.nonzero(~mt).squeeze(1)
    best = -1
    for ep in range(args.epochs):
        net.train()
        perm = idx_tr[torch.randperm(len(idx_tr))]
        tot = n = 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb = torch.from_numpy(Xt[b].astype(np.float32))
            lo, ls = net(xb)
            loss = bce(lo, yt[b]) + 0.5 * ce(ls, st[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            n += 1
        # 验证
        net.eval()
        with torch.no_grad():
            idx_va = torch.nonzero(mt).squeeze(1)
            pv = []
            for i in range(0, len(idx_va), 16384):
                b = idx_va[i:i + 16384]
                lo, _ = net(torch.from_numpy(Xt[b].astype(np.float32)))
                pv.append(torch.sigmoid(lo))
            p = torch.cat(pv).numpy()
        yv = yn[idx_va.numpy()]
        a = auc(yv, p)
        # 以「与人类相同正例率」为阈值时的精确率/召回
        thr = float(np.quantile(p, 1 - yv.mean()))
        pred = p >= thr
        prec = float((pred & (yv > 0)).sum() / max(1, pred.sum()))
        rec = float((pred & (yv > 0)).sum() / max(1, (yv > 0).sum()))
        print(f'epoch {ep}: loss {tot/n:.4f}  AUC {a:.4f}  @同正例率 P {prec:.3f} R {rec:.3f}')
        score = a + prec + rec
        if score > best:
            best = score
            torch.save({'state': net.state_dict(), 'd_in': X.shape[1],
                        'meta': {'auc': a, 'prec': prec, 'rec': rec, 'epoch': ep}},
                       args.out)
    print('best saved →', args.out)

    # 报告：按 ett 分桶的 AUC（高难谱是否更难学）
    net.load_state_dict(torch.load(args.out)['state'])
    net.eval()
    ett = d['ett']
    idx_va = np.nonzero(m_val)[0]
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(Xt[idx_va].astype(np.float32)))[0]).numpy()
    for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 25), (25, 99)]:
        m = (ett[idx_va] >= lo) & (ett[idx_va] < hi)
        if m.sum() > 500:
            print(f'  ett {lo}-{hi}: n={int(m.sum()):7d} AUC {auc(yn[idx_va][m], p[m]):.4f}')


if __name__ == '__main__':
    main()
