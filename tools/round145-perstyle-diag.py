#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 145 轮诊断（只统计）：
  1. tech 模型在不同 --diff 下的原始占位率与密度对齐 p132 —— 看「成簇」是否被难度条件门控；
  2. 复采多个噪声种子 —— 量 5 窗 p132 的采样噪声（判断 Δ 是否超出噪声）；
  3. 同一批 tech val 窗上，分模型 vs 单模型 baseline 的 x0 拟合损失 ——
     判断 p132 缺口是不是「训练不足」造成的（分模型步数少 4×）。
"""
import argparse, importlib.util, json, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STYLES = ['tech', 'jack', 'speed', 'stamina']
OCC = [0.1924, 0.2259, 0.1939, 0.1780]
DS_P132 = [0.5103, 0.0608, 0.6139, 0.1160]


def lm(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='out/diffusion/round145-perstyle')
    ap.add_argument('--data', default='out/diff-data-grouped/dataset.npz')
    ap.add_argument('--baseline-ckpt', default='out/diffusion/round138-train/diffusion32-gj-grouped.pt')
    ap.add_argument('--num', type=int, default=5)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--out', default='')
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    ev = lm('ev', os.path.join(HERE, 'eval-perstyle.py'))
    stats = lm('r141', os.path.join(HERE, 'round141-burst-stats.py'))
    res = {}

    def sample_stats(model, diff, idx, occ, seed, n):
        raw = ev.ddpm_sample(model, diff, n, 0.5, 'x0', seed).numpy()
        thr = float(np.quantile(raw, 1.0 - occ))
        X = (raw > thr).astype(np.uint8)
        st = stats.pooled_stats(X, raw)
        return st

    # ---- 1) 噪声种子离散度 ----
    res['seeds'] = {}
    for i, s in enumerate(STYLES):
        fp = os.path.join(a.root, f'{s}.pt')
        if not os.path.exists(fp):
            continue
        model, diff, pred, ck, nparam = ev.load_model(fp, a.threads)
        vals, occs = [], []
        for seed in [int(x) for x in a.seeds.split(',')]:
            st = sample_stats(model, diff, i, OCC[i], seed, a.num)
            vals.append(st['p132'])
            occs.append(st['占位率'])
            print(f'  {s} seed={seed}: p132 {st["p132"]:.4f}  占位率 {st["占位率"]:.4f}', flush=True)
        res['seeds'][s] = {'p132': vals, 'occ': occs, 'seed_list': [int(x) for x in a.seeds.split(',')],
                           'mean': round(float(np.mean(vals)), 4), 'std': round(float(np.std(vals)), 4),
                           'dataset': DS_P132[i],
                           'delta_mean': round(float(np.mean(vals)) - DS_P132[i], 4)}
        print(f'{s}: p132 均值 {np.mean(vals):.4f} ± {np.std(vals):.4f}（数据集 {DS_P132[i]:.4f}）', flush=True)

    # ---- 2) tech 的 diff 扫描（看难度条件是否门控成簇）----
    res['tech_diff_sweep'] = {}
    fp = os.path.join(a.root, 'tech.pt')
    if os.path.exists(fp):
        model, diff, pred, ck, _ = ev.load_model(fp, a.threads)
        for dv in [0.2, 0.35, 0.5, 0.65, 0.8]:
            raw = ev.ddpm_sample(model, diff, a.num, dv, 'x0', 0).numpy()
            raw_occ = float((raw > 0.5).mean())          # 未做密度对齐时的原始密度
            thr = float(np.quantile(raw, 1.0 - OCC[0]))
            st = stats.pooled_stats((raw > thr).astype(np.uint8), raw)
            res['tech_diff_sweep'][str(dv)] = {'原始占位率@0.5': round(raw_occ, 4),
                                               'p132对齐后': st['p132'],
                                               '占位率对齐后': st['占位率'],
                                               '原始p132@0.5': st['原始p132@0.5']}
            print(f'  tech diff={dv}: 原始占位率@0.5 {raw_occ:.4f}  对齐后 p132 '
                  f'{st["p132"]:.4f}（占位率 {st["占位率"]:.4f}）', flush=True)

    # ---- 3) 拟合诊断：同一批 tech val 窗，分模型 vs 单模型 ----
    d = np.load(a.data)
    X = d['X']; sp = d['split']; sty = d['style']
    va = np.where((sp == 1) & (sty == 0))[0]
    Xn = torch.from_numpy(np.ascontiguousarray(X[va])).float()
    n = len(Xn)
    T = 200
    abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T), 0)

    def fit_mse(model, lo, hi, perstyle):
        occ = Xn.reshape(n, -1).mean(1).numpy()
        dv = torch.from_numpy(np.clip((occ - lo) / (hi - lo), 0, 1).astype(np.float32))
        g = torch.Generator().manual_seed(1234)
        vt = torch.randint(0, T, (n,), generator=g)
        vn = torch.randn(n, 4, 256, generator=g)
        vec = torch.zeros(n)
        with torch.no_grad():
            for i in range(0, n, 512):
                sl = slice(i, min(i + 512, n)); m = min(512, n - i)
                aa = abar[vt[sl]][:, None, None]
                xt = aa.sqrt() * Xn[sl] + (1 - aa).sqrt() * vn[sl]
                if perstyle:
                    pred = model(xt, vt[sl], dv[sl], None)          # 分模型：无 style 张量
                else:
                    z = torch.zeros(m, dtype=torch.long)            # 单模型：style=tech
                    pred = model(xt, vt[sl], z, dv[sl], None)
                vec[sl] = ((pred - Xn[sl]) ** 2).mean(dim=(1, 2))
        return float(vec.mean())

    res['fit_tech_val'] = {}
    ck1 = torch.load(os.path.join(a.root, 'tech.pt'), map_location='cpu', weights_only=False)
    m1 = ev.load_module('ps', os.path.join(HERE, 'train-diffusion32-perstyle.py'))
    mm = m1.UNet1D(base=ck1['base'], temb_dim=128, cond_dim=0)
    mm.load_state_dict(ck1['state'], strict=True); mm.eval()
    l1, h1 = ck1['difficulty_condition']['lo'], ck1['difficulty_condition']['hi']
    res['fit_tech_val']['perstyle'] = round(fit_mse(mm, l1, h1, True), 5)
    ck2 = torch.load(a.baseline_ckpt, map_location='cpu', weights_only=False)
    film = lm('film', os.path.join(HERE, 'sample-diffusion32film.py'))
    mb = film.UNet1D(base=ck2['base'], n_style=4, cond_dim=0)
    mb.load_state_dict(ck2['state'], strict=True); mb.eval()
    l2, h2 = ck2['difficulty_condition']['lo'], ck2['difficulty_condition']['hi']
    res['fit_tech_val']['baseline'] = round(fit_mse(mb, l2, h2, False), 5)
    res['fit_tech_val']['predictor_constant'] = round(float(Xn.mean() * (1 - Xn.mean())), 5)
    res['fit_tech_val']['note'] = ('同一批 tech val 1062 窗、同一套固定 t/噪声；x0 未加权 MSE。'
                                  '分模型步数 567（batch64×9ep），单模型 baseline 2250（batch32×9ep）')
    print(f'  tech-val x0 MSE：分模型 {res["fit_tech_val"]["perstyle"]}  '
          f'baseline {res["fit_tech_val"]["baseline"]}  '
          f'常数预测 {res["fit_tech_val"]["predictor_constant"]}', flush=True)

    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'→ {a.out}')


if __name__ == '__main__':
    main()
