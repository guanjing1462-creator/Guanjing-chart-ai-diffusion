#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 collect-*-probs.npz 复算「细尺度（1 格 ≈ 40ms）对齐」，并做逐窗配对统计。

为什么需要这个
--------------
主协议的 r250 / r1000 先把音符与条件按 250ms / 1000ms 分箱：
190BPM 下 1 格 ≈ 39.5ms，**250ms ≈ 6.3 格**，所以 r250 对「1 格级相位」不敏感
——它量的是包络跟随。要回答「相位分辨率是 160ms 还是 40ms」，
必须直接看 **1 格分箱**（以及 2/4/8… 格的阶梯）上的对齐强度：
  · 对给定C：模型是否忠实跟随它拿到的那个条件（条件保真度）；
  · 对真C  ：条件平移后是否与真谱失配（相位敏感性 = 细相位是否被用上）。
每窗按格分箱求和后再算 Pearson，最后取逐窗中位 + 逐窗配对检验。
本脚本只用已采样好的概率卷，不重新采样。

用法
----
  python3 tools/diag-align-cell.py --probs out/diffusion/round153-align/collect-val-probs.npz
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 256
BINS = (1, 2, 4, 8, 16, 32, 64)
SHIFTS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
REPORT_SHIFTS = (-4, -1, 1, 4)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def bin_sum(v, k):
    nb = WIN // k
    return v[:nb * k].reshape(nb, k).sum(axis=1)


def wilcoxon_p(x, y):
    try:
        from scipy.stats import wilcoxon
        d = np.asarray(x) - np.asarray(y)
        if len(d) >= 6 and np.any(d != 0):
            return float(f'{wilcoxon(x, y).pvalue:.4g}')
    except Exception:
        pass
    return None


def boot_ci(x, y, n_boot=2000, seed=99):
    g = np.random.default_rng(seed)
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    diffs = np.array([d[g.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {'mean': round(float(d.mean()), 4),
            'ci95': [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            'p_one_sided': round(float((diffs <= 0).mean()), 4)}


def paired(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return None
    d = x - y
    out = {'n': int(len(x)), 'diff_mean': round(float(d.mean()), 4),
           'wins': int((d > 0).sum()), 'losses': int((d < 0).sum())}
    p = wilcoxon_p(x, y)
    if p is not None:
        out['wilcoxon_p'] = p
    out['bootstrap'] = boot_ci(x, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', default='out/diffusion/round153-align/collect-val-probs.npz')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--calib', action='store_true',
                    help='用占用校准阈值（把占用拉到人类水平）而不是 0.5')
    ap.add_argument('--out', default='out/diffusion/round153-align/cell-ladder.json')
    a = ap.parse_args()

    z = np.load(os.path.join(ROOT, a.probs), allow_pickle=True)
    X = z['X'].astype(np.uint8)
    C = z['C'].astype(np.float32)
    ctot = C.sum(axis=2)                                  # [n,256] 真条件总强度
    perm = z['perm']
    n = len(X)
    occ_target = float(X.reshape(n, -1).mean())
    groups = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith('P_')}
    print(f'{a.probs}: {n} 窗，{len(groups)} 组概率卷；人类占用 {occ_target:.3f}；'
          f'阈值 {"占用校准" if a.calib else a.threshold}', flush=True)

    def given_cond(g):
        if g.endswith('错配'):
            return ctot[perm]
        if '条件+' in g or '条件-' in g:
            k = int(g.split('条件')[1].split('格')[0])
            return np.roll(ctot, k, axis=1)
        return ctot

    def roll_of(g):
        p = groups[g]
        if a.calib:
            return (p > np.quantile(p, 1.0 - occ_target)).astype(np.uint8)
        return (p > a.threshold).astype(np.uint8)

    rows = {'人类（同 split 真谱）': (X.astype(np.uint8), ctot)}
    for g in groups:
        rows[g] = (roll_of(g), given_cond(g))
    # 每项都是 ([n,4,256] 0/1 卷, [n,256] 所对条件总强度)

    # 逐窗、逐分箱宽度的 r（对给定C / 对真C）
    rid = {tag: {b: {'given': [], 'true': []} for b in BINS} for tag in rows}
    scan = {tag: {'given': {k: [] for k in SHIFTS}, 'true': {k: [] for k in SHIFTS}}
            for tag in rows}
    for tag, (rollall, cgiven) in rows.items():
        assert rollall.shape[0] == n and cgiven.shape == (n, WIN), (tag, rollall.shape)
        for j in range(n):
            dens_full = rollall[j].astype(np.float64).sum(axis=0)      # [256]
            for b in BINS:
                d = bin_sum(dens_full, b)
                rid[tag][b]['given'].append(pearson(d, bin_sum(cgiven[j], b)))
                rid[tag][b]['true'].append(pearson(d, bin_sum(ctot[j], b)))
            for k in SHIFTS:
                scan[tag]['given'][k].append(pearson(dens_full, np.roll(cgiven[j], k)))
                scan[tag]['true'][k].append(pearson(dens_full, np.roll(ctot[j], k)))

    # ---- 控制台：阶梯表 ----
    order = ['人类（同 split 真谱）']
    for base in ('S1', 'S4'):
        order += [f'{base}-配对', f'{base}-错配', f'{base}-零条件']
        order += [f'{base}-条件{k:+d}格' for k in SHIFTS if k != 0]

    def med(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return round(float(np.median(v)), 4) if len(v) else None

    print('\n=== 细尺度阶梯：r（逐窗中位）——「对给定C」= 跟随它拿到的条件 ===')
    print(f'{"组":30s} ' + ' '.join(f'{b:>7d}' for b in BINS) + '   （单位：格，1 格≈40ms）')
    for tag in order:
        if tag not in rows:
            continue
        print(f'{tag:30s} ' + ' '.join(f'{med(rid[tag][b]["given"]):>7.3f}' for b in BINS))

    print('\n=== 细尺度阶梯：r（逐窗中位）——「对真C」= 与本窗真谱的对齐 ===')
    print(f'{"组":30s} ' + ' '.join(f'{b:>7d}' for b in BINS))
    for tag in order:
        if tag not in rows:
            continue
        print(f'{tag:30s} ' + ' '.join(f'{med(rid[tag][b]["true"]):>7.3f}' for b in BINS))

    print('\n=== 1 格分箱（40ms）条件平移扫描：对给定C / 对真C ===')
    print(f'{"组":30s} ' + ' '.join(f'{k:>13d}' for k in SHIFTS))
    for tag in order:
        if tag not in rows:
            continue
        print(f'{tag:30s} ' + ' '.join(
            f'{med(scan[tag]["given"][k]):>6.3f}/{med(scan[tag]["true"][k]):>6.3f}'
            for k in SHIFTS))

    # ---- 关键配对统计 ----
    stats = {}
    stats['1格对齐 S1-配对 − S4-配对'] = paired(rid['S1-配对'][1]['given'],
                                              rid['S4-配对'][1]['given'])
    stats['1格对齐 S1-配对 − 人类'] = paired(rid['S1-配对'][1]['given'],
                                          rid['人类（同 split 真谱）'][1]['given'])
    stats['1格对齐 S4-配对 − 人类'] = paired(rid['S4-配对'][1]['given'],
                                          rid['人类（同 split 真谱）'][1]['given'])
    stats['2格对齐 S1-配对 − S4-配对'] = paired(rid['S1-配对'][2]['given'],
                                              rid['S4-配对'][2]['given'])
    for base in ('S1', 'S4'):
        for k in REPORT_SHIFTS:
            g = f'{base}-条件{k:+d}格'
            stats[f'{base} {k:+d}格·对真C − 配对(对真C) @1格'] = paired(
                rid[g][1]['true'], rid[f'{base}-配对'][1]['true'])
            stats[f'{base} {k:+d}格·对给定C − 配对(对给定C) @1格'] = paired(
                rid[g][1]['given'], rid[f'{base}-配对'][1]['given'])
    stats['S1-配对 − S1-错配（对真C）@1格'] = paired(rid['S1-配对'][1]['true'],
                                                 rid['S1-错配'][1]['true'])
    stats['S4-配对 − S4-错配（对真C）@1格'] = paired(rid['S4-配对'][1]['true'],
                                                 rid['S4-错配'][1]['true'])

    print('\n=== 关键配对统计（1 格分箱，逐窗）===')
    for k, v in stats.items():
        if not v:
            continue
        print(f'  {k}: {v["diff_mean"]:+.4f}（n={v["n"]}，{v["wins"]}胜{v["losses"]}负'
              + (f'，p={v["wilcoxon_p"]:.3g}' if 'wilcoxon_p' in v else '')
              + f'；bootstrap 95% CI [{v["bootstrap"]["ci95"][0]:+.4f}, '
                f'{v["bootstrap"]["ci95"][1]:+.4f}]，单侧 p={v["bootstrap"]["p_one_sided"]:.4f}）')

    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump({'config': {'probs': a.probs, 'threshold': a.threshold,
                              'calib': bool(a.calib), 'occ_target': round(occ_target, 4),
                              'bins_cells': list(BINS)},
                   'ladder_given': {t: {str(b): med(rid[t][b]['given']) for b in BINS}
                                    for t in rows},
                   'ladder_true': {t: {str(b): med(rid[t][b]['true']) for b in BINS}
                                   for t in rows},
                   'scan_given': {t: {str(k): med(scan[t]['given'][k]) for k in SHIFTS}
                                  for t in rows},
                   'scan_true': {t: {str(k): med(scan[t]['true'][k]) for k in SHIFTS}
                                 for t in rows},
                   'stats': stats}, f, ensure_ascii=False, indent=1)
    print(f'\n写出 {a.out}')


if __name__ == '__main__':
    main()
