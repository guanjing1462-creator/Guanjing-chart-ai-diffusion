#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格分离度检验：同一份噪声 + 同一个难度条件，只改 style token，量四个风格到底有多不同。

做的事（只统计，不训练）：
  1. 对每个 style 调一次采样器（默认 tools/sample-diffusion32film.py），
     **固定 --sample-seed**（去噪前的噪声只由 seed/num/T 决定，与模型结构无关），
     `--target-occ` 用该风格数据集的占位率反解阈值（密度对齐，排除密度红利）；
  2. 用 round141 的统计口径（tools/round141-burst-stats.py 的 pooled_stats，直接 import）
     算每风格的占位率 / **p132**（相邻占用行间隔 == 1 的比例，跨窗口池化）；
  3. **两两风格之间的输出差异**：概率卷 P 的平均绝对差 MAE（同 seed → 逐窗对齐）、
     相对尺度（除以两侧 |P| 均值）、以及二值卷的翻转率（Hamming）；
  4. 参照量：同风格、另一份噪声（--sample-seed 1）的 MAE → style/noise 比值。
     这个比值越接近 0，说明「换风格」对输出的影响越小、风格条件越弱；
  5. 对照数据集：p132 的排序是否一致、tech/jack 的倍数是否接近数据集的 8.4 倍。

用法：
  python3 tools/film-style-separation.py --label film --sample-seed 0 \
      --ckpt out/diffusion/round144-film/diffusion32-film-last.pt \
      --outdir out/diffusion/round144-film/samples/film \
      --out out/diffusion/round144-film/style-separation-film.json
"""
import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

STYLES = ['tech', 'jack', 'speed', 'stamina']
# 数据集（train split==0）每风格占位率 —— 采样的密度对齐目标（round141/143 同一套数）
OCC = [0.1924, 0.2259, 0.1939, 0.1780]
# 数据集（train split==0）每风格 p132 —— 判定基准
DATASET_P132 = [0.5103, 0.0608, 0.6139, 0.1160]
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(sampler, ckpt, style, occ, seeds, num, diff, outdir, extra=None):
    """调一次采样器；返回 npz 路径。失败直接抛，不吞退出码。"""
    cmd = [sys.executable, sampler, '--ckpt', ckpt, '--style', str(style),
           '--diff', str(diff), '--num', str(num), '--seed', str(seeds['seed']),
           '--sample-seed', str(seeds['sample_seed']), '--threads', '1',
           '--target-occ', f'{occ:.4f}', '--quiet', '--outdir', outdir]
    if extra:
        cmd += extra
    print('  $ ' + ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + '\n' + r.stderr[-3000:] + '\n')
        raise SystemExit(f'[错误] 采样失败 style={style} 退出码 {r.returncode}')
    for line in r.stdout.splitlines():
        if '风格条件结构' in line or 'p132' in line or '占位率:' in line:
            print('    ' + line.strip())
    files = sorted(glob.glob(os.path.join(outdir, f'rolls-s{style}d*.npz')))
    # 同一 outdir 里换 sample-seed 时文件名相同 → 后写覆盖前写；所以每个 seed 单独目录
    if not files:
        raise SystemExit(f'[错误] {outdir} 里没有 roll npz')
    return files


def load_rolls(dirpath, style):
    files = sorted(glob.glob(os.path.join(dirpath, f'rolls-s{style}d*.npz')))
    if not files:
        raise SystemExit(f'[错误] {dirpath} 里没有 style {style} 的 rolls-*.npz')
    xs, ps = [], []
    for f in files:
        z = np.load(f)
        xs.append(np.asarray(z['X']))
        if 'P' not in z.files:
            raise SystemExit(f'[错误] {f} 没有 P（概率卷），无法算 MAE')
        ps.append(np.asarray(z['P']))
    return np.concatenate(xs), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, help='配置名（baseline / film）')
    ap.add_argument('--ckpt', required=True, help='权重路径')
    ap.add_argument('--sampler', default=os.path.join(HERE, 'sample-diffusion32film.py'))
    ap.add_argument('--outdir', required=True, help='采样输出根目录（每 style 一个子目录）')
    ap.add_argument('--out', default='', help='结果 json')
    ap.add_argument('--num', type=int, default=5, help='每风格窗口数（任务书：5）')
    ap.add_argument('--diff', type=float, default=0.5, help='连续难度条件（四风格统一）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sample-seed', type=int, default=0,
                    help='去噪噪声种子（四风格、跨模型相同 → 逐窗可对齐）')
    ap.add_argument('--noise-ref', type=int, default=1,
                    help='1=额外用 --sample-seed+1 采一遍，做「同风格换噪声」的参照量')
    ap.add_argument('--data', default='out/diff-data-grouped/dataset.npz')
    ap.add_argument('--skip-run', type=int, default=0, help='1=复用已有 npz，不重新采样')
    a = ap.parse_args()

    stats = load_module('r141stats', os.path.join(HERE, 'round141-burst-stats.py'))
    res = {'label': a.label, 'ckpt': a.ckpt, 'num_per_style': a.num, 'diff': a.diff,
           'seed': a.seed, 'sample_seed': a.sample_seed,
           'target_occ': dict(zip(STYLES, OCC)),
           'outdir': a.outdir,
           'protocol': '同一 sample-seed（噪声逐元素相同）+ 同一 diff=0.5，只改 style；'
                       '按该风格数据集占位率反解阈值（密度对齐）；p132 = 相邻占用行间隔==1 '
                       '的比例，跨窗口池化（与 round141/143 报表同口径）'}

    # 1) 采样
    P, X = {}, {}
    for i, name in enumerate(STYLES):
        if a.skip_run:
            d0 = os.path.join(a.outdir, f'{name}-noise{a.sample_seed}')
            print(f'[{i+1}/4] {name}: 复用 {d0}', flush=True)
        else:
            d0 = os.path.join(a.outdir, f'{name}-noise{a.sample_seed}')
            os.makedirs(d0, exist_ok=True)
            print(f'[{i+1}/4] {name}: 采样（target-occ {OCC[i]}）→ {d0}', flush=True)
            run_one(a.sampler, a.ckpt, i, OCC[i],
                    {'seed': a.seed, 'sample_seed': a.sample_seed}, a.num, a.diff, d0)
        X[name], P[name] = load_rolls(d0, i)

    # 2) 每风格统计（round141 口径）
    per = {}
    for name in STYLES:
        st = stats.pooled_stats(X[name], P[name])
        j = STYLES.index(name)
        st['数据集p132'] = DATASET_P132[j]
        st['数据集占位率'] = OCC[j]
        st['p132减数据集'] = round(st['p132'] - DATASET_P132[j], 4)
        per[name] = st
    res['per_style'] = per

    # 3) 两两差异
    pairs, ham = {}, {}
    for i in range(4):
        for j in range(i + 1, 4):
            ni, nj = STYLES[i], STYLES[j]
            pi, pj = P[ni], P[nj]
            mae = float(np.abs(pi - pj).mean())
            scale = 0.5 * (float(np.abs(pi).mean()) + float(np.abs(pj).mean()))
            pairs[f'{ni}-{nj}'] = {'mae': round(mae, 6),
                                   'mae_rel': round(mae / max(1e-9, scale), 4),
                                   'scale_mean_abs_P': round(scale, 4)}
            ham[f'{ni}-{nj}'] = round(float((X[ni] != X[nj]).mean()), 4)
    res['pairwise_mae'] = pairs
    res['pairwise_hamming'] = ham
    mae_mean = float(np.mean([v['mae'] for v in pairs.values()]))
    rel_mean = float(np.mean([v['mae_rel'] for v in pairs.values()]))
    res['cross_style_mae_mean'] = round(mae_mean, 6)
    res['cross_style_mae_rel_mean'] = round(rel_mean, 4)
    res['cross_style_hamming_mean'] = round(float(np.mean(list(ham.values()))), 4)

    # 4) 参照：同风格、换噪声
    if a.noise_ref:
        noisemae = {}
        for i, name in enumerate(STYLES):
            d1 = os.path.join(a.outdir, f'{name}-noise{a.sample_seed + 1}')
            os.makedirs(d1, exist_ok=True)
            print(f'[参照] {name}: 另一份噪声（sample-seed {a.sample_seed + 1}）→ {d1}', flush=True)
            run_one(a.sampler, a.ckpt, i, OCC[i],
                    {'seed': a.seed, 'sample_seed': a.sample_seed + 1}, a.num, a.diff, d1)
            _, P2 = load_rolls(d1, i)
            noisemae[name] = round(float(np.abs(P[name] - P2).mean()), 6)
        res['same_style_other_noise_mae'] = noisemae
        nm = float(np.mean(list(noisemae.values())))
        res['cross_noise_mae_mean'] = round(nm, 6)
        res['style_over_noise'] = round(mae_mean / max(1e-9, nm), 4)

    # 5) 排序与倍数（主指标）
    p132 = [per[n]['p132'] for n in STYLES]                  # tech jack speed stamina
    order = [STYLES[k] for k in np.argsort(p132)[::-1]]
    ds_order = [STYLES[k] for k in np.argsort(DATASET_P132)[::-1]]
    res['p132'] = dict(zip(STYLES, p132))
    res['p132_desc'] = order
    res['dataset_p132_desc'] = ds_order
    res['order_matches_dataset'] = order == ds_order
    res['ratio_tech_over_jack'] = round(per['tech']['p132'] / max(1e-9, per['jack']['p132']), 2)
    res['dataset_ratio_tech_over_jack'] = round(
        DATASET_P132[0] / max(1e-9, DATASET_P132[1]), 2)
    # 「成簇组 vs 非成簇组」的平均 p132 之外的另一个判定量：speed+tech 与 jack+stamina 的均值差
    hi = 0.5 * (per['tech']['p132'] + per['speed']['p132'])
    lo = 0.5 * (per['jack']['p132'] + per['stamina']['p132'])
    res['cluster_gap_mean'] = round(hi - lo, 4)
    res['cluster_gap_dataset'] = round(0.5 * (DATASET_P132[0] + DATASET_P132[2])
                                       - 0.5 * (DATASET_P132[1] + DATASET_P132[3]), 4)
    res['max_abs_delta_vs_dataset'] = round(
        max(abs(per[n]['p132'] - DATASET_P132[STYLES.index(n)]) for n in STYLES), 4)

    # ---- 打印 ----
    print(f'\n=== {a.label}：四风格（{a.num} 窗/风格，密度对齐，同噪声）===')
    print(f'{"style":>8} {"占位率":>7} {"p132":>7} {"数据集":>7} {"Δ":>8}')
    for n in STYLES:
        r = per[n]
        print(f'{n:>8} {r["占位率"]:>7.4f} {r["p132"]:>7.4f} {r["数据集p132"]:>7.4f} '
              f'{r["p132减数据集"]:>+8.4f}')
    print(f'  p132 降序: {order}  （数据集 {ds_order}）  一致={order == ds_order}')
    print(f'  tech/jack 倍数: {res["ratio_tech_over_jack"]:.2f}'
          f'（数据集 {res["dataset_ratio_tech_over_jack"]:.2f}）')
    print(f'  成簇组均值 - 非成簇组均值: {res["cluster_gap_mean"]:.4f}'
          f'（数据集 {res["cluster_gap_dataset"]:.4f}）')
    print('\n  两两概率卷差异（MAE / 相对 / 二值翻转率）:')
    for k, v in pairs.items():
        print(f'    {k:>16}  {v["mae"]:.4f}  {v["mae_rel"]:.3f}  {ham[k]:.4f}')
    print(f'  换风格 MAE 均值 {mae_mean:.4f}（相对 {rel_mean:.3f}）；'
          f'换风格二值翻转率均值 {res["cross_style_hamming_mean"]:.4f}')
    if a.noise_ref:
        print(f'  换噪声 MAE 均值 {res["cross_noise_mae_mean"]:.4f} → '
              f'style/noise = {res["style_over_noise"]:.4f}')
    res['_dataset_p132_ref'] = dict(zip(STYLES, DATASET_P132))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
