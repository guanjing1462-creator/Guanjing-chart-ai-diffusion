#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""结构诊断：条件「单格脉冲」的输出响应宽度（相位分辨率）。

不采样、不算指标，直接问：**把条件里某一格（1/32 格 ≈ 40ms）抬高，
模型输出在时间轴上响应多宽？**

  · S4（stride=4 单点注入）：条件在进入网络前被 4 格一采样，响应必然是
    4 格宽的方块（再被卷积抹平一点）→ 半宽 ≈ 4 格 ≈ 160ms；
  · S1（多尺度全分辨率）：最浅层 256 位置注入，响应应在 1 格处最尖。

做法：取真实 val 窗的条件，在随机格 j 上把该格 32 带条件 +0.5（其余不动），
比较模型输出 x0 的差；对 offset −8..+8 统计 |Δ| 的均值/中位，归一化到 offset 0。
x_t 用真实窗加固定噪声（t=100），同一窗同一噪声下比较，只差条件扰动。
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 256
OFFSETS = list(range(-8, 9))


def load_module(relpath, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_model(trainer, ckpt, threads):
    torch.set_num_threads(max(1, threads))
    ck = torch.load(os.path.join(ROOT, ckpt), map_location='cpu', weights_only=False)
    inject = str(ck.get('inject', 'mid4'))
    m = trainer.UNet1D(base=int(ck.get('base', 48)), cond_dim=int(ck.get('cond_dim', 0)),
                       cond_ch=int(ck.get('cond_channels', 0)),
                       cond_stride=int(ck.get('cond_stride', 4 if inject == 'mid4' else 1)),
                       inject=inject)
    m.load_state_dict(ck['state'], strict=True)
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data-mel/dataset.npz')
    ap.add_argument('--cond-key', default='M')
    ap.add_argument('--ckpt-s1', default='out/models/r153-S1-multi.pt')
    ap.add_argument('--ckpt-s4', default='out/models/r153-S4-mid4.pt')
    ap.add_argument('--windows', type=int, default=8, help='用几个 val 窗平均')
    ap.add_argument('--impulses', type=int, default=32, help='每窗打几个单格脉冲')
    ap.add_argument('--amp', type=float, default=0.5, help='脉冲幅度（加到该格 32 带上）')
    ap.add_argument('--t', type=int, default=100)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='out/diffusion/round153-align/impulse.json')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    trainer = load_module('tools/train-diffusion32-align.py', 'tr153')
    d = np.load(os.path.join(ROOT, a.data), allow_pickle=True)
    split = d['split'].astype(np.int64)
    M = np.asarray(d[a.cond_key], dtype=np.float32)
    X = np.asarray(d['X'], dtype=np.float32)
    occ = X.reshape(len(X), -1).mean(axis=1)
    va = np.where(split == 1)[0]
    # 按占位率分层取窗（确定性）
    order = va[np.argsort(occ[va], kind='stable')]
    pick = order[np.linspace(0, len(order) - 1, a.windows).astype(int)]

    g = torch.Generator().manual_seed(a.seed)
    vt = torch.randint(a.t, a.t + 1, (len(pick),), generator=g)
    noise = torch.randn(len(pick), 4, WIN, generator=g)

    models = {}
    ck_all = {}
    for tag, path in (('S1', a.ckpt_s1), ('S4', a.ckpt_s4)):
        m, ck = load_model(trainer, path, a.threads)
        models[tag] = m
        ck_all = ck_all or ck
        print(f'{tag}: {path}（inject={ck.get("inject")}, '
              f'cond_stride={ck.get("cond_stride")}）', flush=True)

    diff = trainer.Diffusion(200)
    cond0 = torch.from_numpy(M[pick])                       # [n,256,32]
    x0 = torch.from_numpy(X[pick])
    xt = diff.q(x0, vt, noise)
    sty = torch.zeros(len(pick), dtype=torch.long)
    dc = ck_all.get('difficulty_condition') or {}
    lo_, hi_ = float(dc.get('lo', 0.067)), float(dc.get('hi', 0.490))
    dv = torch.from_numpy(np.clip((occ[pick] - lo_) / (hi_ - lo_), 0, 1).astype(np.float32))

    # 脉冲位置：与窗边界留 16 格余量
    rng = np.random.default_rng(a.seed)
    pos = rng.integers(16, WIN - 16, size=a.impulses)

    res = {}
    for tag, m in models.items():
        prof = np.zeros((len(OFFSETS),), dtype=np.float64)
        cnt = 0
        with torch.no_grad():
            base = m(xt, vt, sty, dv, cond0).numpy()
            for j in pos:
                c = cond0.clone()
                c[:, int(j), :] += a.amp
                out = m(xt, vt, sty, dv, c).numpy()
                dd = np.abs(out - base).mean(axis=0).mean(axis=0)   # [256]
                # 每个窗中心化（去掉整体幅度漂移）
                dd = dd - dd.min()
                for i, off in enumerate(OFFSETS):
                    k = int(j) + off
                    if 0 <= k < WIN:
                        prof[i] += dd[k]
                cnt += 1
        prof /= max(1, cnt)
        peak = float(prof[OFFSETS.index(0)])
        norm = prof / max(1e-12, peak)
        half = [o for o, v in zip(OFFSETS, norm) if v >= 0.5]
        # 半高宽（连续格数）：从 0 向两侧扩展，直到 < 0.5
        lo = 0
        while lo > -8 and norm[OFFSETS.index(lo - 1)] >= 0.5:
            lo -= 1
        hi_ = 0
        while hi_ < 8 and norm[OFFSETS.index(hi_ + 1)] >= 0.5:
            hi_ += 1
        res[tag] = {
            '峰值|Δx0|': round(peak, 5),
            '归一化响应': {str(o): round(float(v), 4) for o, v in zip(OFFSETS, norm)},
            '半高宽_格': (hi_ - lo + 1),
            '半高宽范围': [lo, hi_],
            'offset±1_相对': round(float((norm[OFFSETS.index(1)] + norm[OFFSETS.index(-1)]) / 2), 4),
            'offset±2_相对': round(float((norm[OFFSETS.index(2)] + norm[OFFSETS.index(-2)]) / 2), 4),
            'offset±4_相对': round(float((norm[OFFSETS.index(4)] + norm[OFFSETS.index(-4)]) / 2), 4),
        }
        print(f'{tag}: 峰值 |Δx0| {peak:.5f}，半高宽 {res[tag]["半高宽_格"]} 格；'
              f'归一化响应 ' + ' '.join(f'{o:+d}:{v:.2f}' for o, v in zip(OFFSETS, norm)
                                        if o in (-4, -2, -1, 0, 1, 2, 4)), flush=True)

    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump({'config': {'windows': [int(i) for i in pick], 'impulses': a.impulses,
                              'amp': a.amp, 't': int(a.t), 'offsets': OFFSETS},
                   'result': res}, f, ensure_ascii=False, indent=1)
    print(f'写出 {a.out}', flush=True)


if __name__ == '__main__':
    main()
