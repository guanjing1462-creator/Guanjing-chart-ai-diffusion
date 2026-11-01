#!/usr/bin/env python3
"""训练谱面形状自回归模型（小型 Transformer，CPU 可跑）。

对比基线：二阶马尔可夫转移表（留出损失 tech 2.19 / jack 2.89 / speed 1.97 / stamina 2.25 bits）。

输入特征（都是类别）：风格、节奏档、小节位置、上一行形状、上上行形状
输出：下一行形状（4K 共 15 种可能）
"""
import argparse, glob, json, math, os, random, sys
import numpy as np
import torch
import torch.nn as nn

SHAPES = []
for k in range(1, 5):
    for c in __import__('itertools').combinations(range(4), k):
        SHAPES.append(c)
SHAPE_ID = {s: i for i, s in enumerate(SHAPES)}
NSHAPE = len(SHAPES)          # 15
GAPS = ['32', '16', '8', '4', '2', '1', 'slow']
GAP_ID = {g: i for i, g in enumerate(GAPS)}
NGAP = len(GAPS)              # 7
STYLES = ['tech', 'jack', 'speed', 'stamina']
STYLE_ID = {s: i for i, s in enumerate(STYLES)}
NPOS = 16


def parse_rows(path):
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
    return [(t, tuple(sorted(c))) for t, c in sorted(by_t.items())]


def gap_bucket(dt):
    if dt <= 60: return '32'
    if dt <= 110: return '16'
    if dt <= 190: return '8'
    if dt <= 330: return '4'
    if dt <= 650: return '2'
    if dt <= 1200: return '1'
    return 'slow'


def chart_samples(path, style):
    rows = parse_rows(path)
    if len(rows) < 60:
        return []
    out = []
    for i in range(2, len(rows)):
        t0, s0 = rows[i - 1]
        t1, s1 = rows[i]
        _, s2 = rows[i - 2]
        if s1 not in SHAPE_ID:
            continue
        g = gap_bucket(t1 - t0)
        pos = int((t1 % 2000) // 125)
        out.append((STYLE_ID[style], GAP_ID[g], pos, SHAPE_ID[s0], SHAPE_ID[s2], SHAPE_ID[s1]))
    return out


class ChartTransformer(nn.Module):
    def __init__(self, d=96, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(STYLE_ID.__len__() + 1, d)
        self.gap = nn.Embedding(NGAP, d)
        self.pos = nn.Embedding(NPOS, d)
        self.p1 = nn.Embedding(NSHAPE, d)
        self.p2 = nn.Embedding(NSHAPE, d)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 2, dropout=dropout,
                                       batch_first=True, norm_first=True),
            num_layers=layers)
        self.head = nn.Linear(d, NSHAPE)

    def forward(self, st, gp, ps, p1, p2):
        x = (self.emb(st) + self.gap(gp) + self.pos(ps) + self.p1(p1) + self.p2(p2))
        x = x.unsqueeze(1)
        x = self.blocks(x).squeeze(1)
        return self.head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--limit', type=int, default=0, help='每包最多用多少张（0=全部）')
    ap.add_argument('--out', default='out/models/neural-shape.pt')
    a = ap.parse_args()
    random.seed(3231)

    data = []
    for style in STYLES:
        files = sorted(glob.glob(f'out/packs/{style}/*.osu'))
        if a.limit:
            files = files[:a.limit]
        for f in files:
            data += chart_samples(f, style)
    print(f'样本总数 {len(data)}（来自 {sum(1 for s in STYLES for _ in sorted(glob.glob(f"out/packs/{s}/*.osu"))[:(a.limit or 10**9)])} 张谱）')
    random.shuffle(data)
    k = int(len(data) * 0.9)
    tr, te = data[:k], data[k:]
    print(f'训练 {len(tr)} / 测试 {len(te)}')

    def tens(split):
        arr = np.array(split, dtype=np.int64)
        return [torch.from_numpy(arr[:, i]) for i in range(arr.shape[1])]

    tr_t = tens(tr); te_t = tens(te)
    model = ChartTransformer()
    print('参数量 %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    n = len(tr)
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            st, gp, ps, p1, p2, y = [t[idx] for t in tr_t]
            opt.zero_grad()
            logits = model(st, gp, ps, p1, p2)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            st, gp, ps, p1, p2, y = te_t
            te_loss = 0.0
            for i in range(0, len(y), 4096):
                lg = model(st[i:i+4096], gp[i:i+4096], ps[i:i+4096], p1[i:i+4096], p2[i:i+4096])
                te_loss += lossf(lg, y[i:i+4096]).item() * min(4096, len(y) - i)
        te_loss /= len(y)
        model.train()
        print(f'epoch {ep}: 训练 {tot/n:.4f} nats / 留出 {te_loss:.4f} nats '
              f'（{te_loss/math.log(2):.3f} bits，基线 2.19–2.89）', flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save({'state': model.state_dict(), 'shapes': SHAPES, 'gaps': GAPS}, a.out)
    print('已保存', a.out)


if __name__ == '__main__':
    main()
