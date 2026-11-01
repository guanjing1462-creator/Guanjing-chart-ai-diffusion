#!/usr/bin/env python3
"""键型分化判别测试。

思路：
  1. 用人类包统计每档的特征向量（npr、1/32、jack、手位移、同手率、行大小分布）。
  2. 对生成器产出的每张谱算同样特征。
  3. 用「最近质心」判别：这张谱最像哪一档？
     —— 若绝大多数谱都被判到自己那一档，说明键型分化是成功的。

用法：python3 tools/test-tier-differentiation.py [--charts 目录]
"""
import argparse
import collections
import glob
import importlib.util
import json
import os

import numpy as np

spec = importlib.util.spec_from_file_location('ac', 'tools/audit-chart.py')
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

KEYS = ['npr', 'g32', 'jack', 'hand', 'same_hand', 'p1', 'p2', 'p3', 'p4']


def features(path, step_ms):
    n, _ = ac.parse_osu(path)
    r = ac.rows_of(n)
    if len(r) < 50:
        return None
    vals = []
    g32 = 0
    ss = sj = 0
    c2 = collections.Counter()
    sizes = collections.Counter()
    for t, cs in r:
        sizes[len(cs)] += 1
        if len(cs) == 2:
            c2[tuple(sorted(cs))] += 1
    for (t0, a), (t1, b) in zip(r, r[1:]):
        for c in b:
            vals.append(min(abs(c - q) for q in a))
        dt = t1 - t0
        g32 += dt / step_ms <= 0.6
        if len(a) == 1 and len(b) == 1 and dt >= 81:
            ss += 1
            sj += a[0] == b[0]
    t2 = sum(c2.values())
    nsz = sum(sizes.values())
    return dict(
        npr=len(n) / len(r),
        g32=g32 / max(1, len(r) - 1),
        jack=sj / max(1, ss),
        hand=sum(vals) / len(vals),
        same_hand=(c2[(0, 1)] + c2[(2, 3)]) / max(1, t2),
        p1=sizes[1] / nsz, p2=sizes[2] / nsz, p3=sizes[3] / nsz, p4=sizes[4] / nsz,
    )


def pack_centroids(pack_dir, limit=120):
    rows = {k: [] for k in KEYS}
    files = sorted(glob.glob(os.path.join(pack_dir, '*.osu')))[:limit]
    for f in files:
        # 人类谱没有分析文件，用 onset 间隔中位数估 step
        n, _ = ac.parse_osu(f)
        r = ac.rows_of(n)
        if len(r) < 100:
            continue
        gaps = [r[i + 1][0] - r[i][0] for i in range(min(len(r) - 1, 400))]
        step = float(np.median(gaps)) if gaps else 100.0
        fv = features(f, max(40.0, step))
        if fv:
            for k in KEYS:
                rows[k].append(fv[k])
    return {k: (float(np.mean(v)), float(np.std(v) + 1e-6)) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charts', default='out/eval')
    ap.add_argument('--song', default='Heart_Fraction')
    a = ap.parse_args()

    packs = {'Tech': 'out/packs/tech', 'Jack': 'out/packs/jack',
             'Speed': 'out/packs/speed', 'Stamina': 'out/packs/stamina'}
    print('统计人类包特征（各 120 张）...')
    refs = {}
    for tier, d in packs.items():
        if not os.path.isdir(d):
            continue
        refs[tier] = pack_centroids(d)
        print('  %-8s npr %.2f  1/32 %.2f  jack %.3f  hand %.2f' % (
            tier, refs[tier]['npr'][0], refs[tier]['g32'][0],
            refs[tier]['jack'][0], refs[tier]['hand'][0]))

    an = json.load(open(f'out/multisong/{a.song}-fx.json'))
    step = an['stepMs']
    charts = sorted(glob.glob(os.path.join(a.charts, '*.osu')))
    if not charts:
        print(f'没有找到待评估谱面：{a.charts}')
        return
    print('\n判别结果：')
    ok = 0
    for f in charts:
        fv = features(f, step)
        if not fv:
            continue
        dist = {}
        for tier, ref in refs.items():
            d = 0.0
            for k in KEYS:
                mu, sd = ref[k]
                d += ((fv[k] - mu) / sd) ** 2
            dist[tier] = d ** 0.5
        nearest = min(dist, key=dist.get)
        name = os.path.basename(f).replace('.osu', '')
        truth = name.split('--')[1] if '--' in name else '?'
        hit = (nearest == truth)
        ok += hit
        print('  %-34s 真档 %-8s 判为 %-8s %s  (距离 %s)' % (
            name[:34], truth, nearest, '✅' if hit else '❌',
            ' '.join(f'{k}:{v:.1f}' for k, v in sorted(dist.items()))))
    print(f'\n判别正确率 {ok}/{len(charts)} = {ok / max(1, len(charts)) * 100:.0f}%')


if __name__ == '__main__':
    main()
