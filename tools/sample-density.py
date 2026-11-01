#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""密度校准版 DDPM 采样器（`tools/sample-diffusion.py` 的补丁版，与其输出格式兼容）。

为什么要另开一个
----------------
官方采样器从 `x_T ~ N(0,1)`（均值 0）起步，但**训练时** x_T 的分布是
    q(x_T) = N(√ᾱ_T·x_0, 1−ᾱ_T)，而 x_0 的均值是数据占位率（≈0.41，按条件是 0.28~0.54），
所以 √ᾱ_T·x_0 的均值是 0.15 左右。用均值 0 起步等于把轨迹的"负载"从 0.15 起，
200 步里靠模型一点点往上拉，结果系统性偏低：实测（32 窗口、style1/diff2）
th=0.5 占位率只有 0.239，模型 x0 预测均值 0.26，而该条件真实值是 0.50。

本脚本做三件事（都能单独关掉）：
1. `--init-mean`：x_T = √ᾱ_T·μ + √(1−ᾱ_T)·ε，μ 取该条件的训练集占位率（'auto'）
   或给定常数。实测把 th=0.5 的占位率从 0.273 拉到 0.415（目标 0.407），
   而且**行分布误差同时从 0.77 降到 0.33**（不只是阈值平移，结构也变好了）。
2. `--x0-clamp 0,1`：x0 是概率，clamp 到 [0,1]（官方是 [-1,2]，负值会把轨迹往下拽）。
3. `--threshold`：最后二值化阈值。

用法
----
  python3 tools/sample-density.py --ckpt out/models/diffusion4k-x0full.pt \\
      --style 1 --diff 2 --num 8 --threshold 0.5 --outdir out/diffusion/density/calib
输出与官方采样器一致：roll-*.npy（[4,128] uint8）、rolls-*.npz（X/P）、sample-meta-*.json。
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


td = _load('train_density', 'train-density.py')


def cond_mean(data_path, style, diff):
    """该条件的训练集占位率（'auto' 初始化用）。只读数据集，不改它。"""
    d = np.load(data_path)
    m = (d['style'] == style) & (d['diff'] == diff)
    if m.sum() < 20:
        return float(d['X'].mean()), int(m.sum())
    return float(d['X'][m].mean()), int(m.sum())


@torch.no_grad()
def sample(model, diff, n, style, diff_id, pred='x0', init_mean=0.0,
           x0_clamp=(0.0, 1.0), det_tail=0, verbose=True):
    T = diff.T
    aT = diff.abar[T - 1].sqrt().item()
    bT = (1 - diff.abar[T - 1]).sqrt().item()
    x = aT * init_mean + bT * torch.randn(n, 4, 128)
    sty = torch.full((n,), int(style), dtype=torch.long)
    dif = torch.full((n,), int(diff_id), dtype=torch.long)
    t0 = time.time()
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long)
        out = model(x, tb, sty, dif)
        a_t, ab = diff.alphas[t], diff.abar[t]
        if pred == 'x0':
            x0 = out.clamp(*x0_clamp)
        elif pred == 'v':
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * out
        else:
            x0 = (x - (1 - ab).sqrt() * out) / ab.sqrt()
        if t > 0:
            abp = diff.abar[t - 1]
            mean = (abp.sqrt() * diff.betas[t] * x0 + a_t.sqrt() * (1 - abp) * x) / (1 - ab)
            x = mean if t < det_tail else mean + (
                diff.betas[t] * (1 - abp) / (1 - ab)).sqrt() * torch.randn_like(x)
        else:
            x = x0
        if verbose and t % max(1, T // 5) == 0:
            print(f'  t={t:4d}/{T} {time.time() - t0:5.1f}s', flush=True)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out/models/diffusion4k-x0full.pt')
    ap.add_argument('--style', type=int, default=1)
    ap.add_argument('--diff', type=int, default=2)
    ap.add_argument('--num', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--init-mean', default='auto',
                    help="x_T 的均值：'auto'=该条件的训练集占位率；'0'=官方行为（N(0,1)）")
    ap.add_argument('--x0-clamp', default='0,1', help='x0 预测的 clamp 区间，官方是 -1,2')
    ap.add_argument('--det-tail', type=int, default=0)
    ap.add_argument('--data', default='out/diff-data/dataset.npz')
    ap.add_argument('--bpm', type=float, default=136.3)
    ap.add_argument('--outdir', default='out/diffusion/density/calib')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    state = ck['state'] if isinstance(ck, dict) and 'state' in ck else ck
    base = int(ck.get('base', 48)) if isinstance(ck, dict) else 48
    pred = ck.get('pred', 'eps') if isinstance(ck, dict) else 'eps'
    T = int(ck.get('timesteps', 200)) if isinstance(ck, dict) else 200
    model = td.UNet1D(base=base)
    model.load_state_dict(state, strict=True)
    model.eval()
    diff = td.Diffusion(T)

    if a.init_mean == 'auto':
        mu, nref = cond_mean(a.data, a.style, a.diff)
        note = f'auto（style{a.style}/diff{a.diff} 训练集占位率 {mu:.4f}，{nref} 窗口）'
    else:
        mu = float(a.init_mean)
        note = f'常数 {mu:.4f}' + ('（=官方 N(0,1) 行为）' if mu == 0 else '')
    clamp = tuple(float(v) for v in a.x0_clamp.split(','))

    print(f'▶ {a.ckpt} pred={pred} base={base} T={T}｜style {a.style} diff {a.diff} num {a.num}')
    print(f'  x_T 均值 {note}｜x0 clamp {clamp}｜阈值 {a.threshold}｜seed {a.seed}', flush=True)

    torch.manual_seed(a.seed)
    t0 = time.time()
    P = sample(model, diff, a.num, a.style, a.diff, pred=pred, init_mean=mu,
               x0_clamp=clamp, det_tail=a.det_tail, verbose=not a.quiet).numpy()
    cost = time.time() - t0
    Xb = (P > a.threshold).astype(np.uint8)

    rs = Xb.sum(1).astype(np.int64)     # uint8 求和得到 uint64，bincount 不认
    h = np.bincount(rs.ravel(), minlength=5) / rs.size
    step_ms = 60000.0 / a.bpm / 4
    st = {
        '窗口数': int(a.num), '占位率': round(float(Xb.mean()), 4),
        '每行平均音符数': round(float(rs.mean()), 3),
        '空行比例': round(float((rs == 0).mean()), 3),
        '行分布0-4': [round(float(v), 3) for v in h],
        '非空行平均和弦': round(float(rs[rs > 0].mean()), 3),
        '各列占位率': [round(float(v), 4) for v in Xb.mean(axis=(0, 2))],
        '每窗口音符数': [int(v) for v in Xb.sum(axis=(1, 2))],
        '每窗口占位率': [round(float(v), 4) for v in Xb.mean(axis=(1, 2))],
        '耗时_s': round(cost, 1), '阈值': a.threshold,
        'x_T均值': round(mu, 4), 'x0clamp': list(clamp),
        'x0预测': {'均值': round(float(P.mean()), 4), '标准差': round(float(P.std()), 4),
                   'q50': round(float(np.quantile(P, .5)), 4)},
    }
    print('\n=== 采样统计 ===')
    for k, v in st.items():
        print(f'  {k}: {v}')

    os.makedirs(a.outdir, exist_ok=True)
    stem = f's{a.style}d{a.diff}-mu{mu:.3f}-th{a.threshold}-seed{a.seed}'
    for i in range(a.num):
        np.save(os.path.join(a.outdir, f'roll-{stem}-{i:02d}.npy'), Xb[i])
    npz = os.path.join(a.outdir, f'rolls-{stem}.npz')
    np.savez_compressed(npz, X=Xb, P=P, style=np.int64(a.style), diff=np.int64(a.diff),
                        seed=np.int64(a.seed), threshold=np.float64(a.threshold),
                        init_mean=np.float64(mu))
    with open(os.path.join(a.outdir, f'sample-meta-{stem}.json'), 'w', encoding='utf-8') as f:
        json.dump({'ckpt': a.ckpt, 'style': a.style, 'diff': a.diff, 'num': a.num,
                   'seed': a.seed, 'threshold': a.threshold, 'init_mean': mu,
                   'x0_clamp': list(clamp), 'bpm': a.bpm, 'timesteps': T,
                   'stats': st}, f, ensure_ascii=False, indent=1)
    print(f'\n钢琴卷 → {a.outdir}/roll-{stem}-*.npy\n概率卷 → {npz}')


if __name__ == '__main__':
    main()
