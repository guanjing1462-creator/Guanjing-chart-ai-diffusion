#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus-baseline: 用与 audit-chart 完全相同的口径统计人类谱库，
得到「人类是什么样」的分布，作为生成谱的靶子。

与指纹库的区别：指纹库是生成参数，本脚本是**验收口径**（judged the same
way as our generated charts）。所有指标定义与 audit-chart.py 一致。

用法：
  python3 tools/corpus-baseline.py --charts out/charts --features out/feature-db.jsonl \
      --per-bucket 40 -o out/human-baseline.json
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import statistics as st
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('audit_chart', os.path.join(HERE, 'audit-chart.py'))
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

GAP_EDGES = [(0, 30), (30, 60), (60, 100), (100, 140), (140, 200), (200, 290), (290, 410), (410, 10**9)]
GAP_LABELS = ['<30', '30-60', '60-100', '100-140', '140-200', '200-290', '290-410', '>410']


def metrics(path):
    notes, meta = ac.parse_osu(path)
    if len(notes) < 200:
        return None
    rows = ac.rows_of(notes)
    dur = (notes[-1][0] - notes[0][0]) / 1000.0
    if dur < 20:
        return None
    out = {'id': os.path.splitext(os.path.basename(path))[0],
           'notes': len(notes), 'rows': len(rows), 'durSec': round(dur, 1),
           'nps': len(notes) / dur, 'rps': len(rows) / dur,
           'notesPerRow': len(notes) / len(rows),
           'chordRowRate': sum(1 for t, cs in rows if len(cs) > 1) / len(rows)}
    # gap 分布（ms）
    gaps = [rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1)]
    hist = Counter()
    for g in gaps:
        for lab, (lo, hi) in zip(GAP_LABELS, GAP_EDGES):
            if lo <= g < hi:
                hist[lab] += 1
                break
    tot = sum(hist.values()) or 1
    out['gapHist'] = {k: round(hist[k] / tot, 4) for k in GAP_LABELS}
    # jack：同列相邻音符 <120ms
    bycol = defaultdict(list)
    for t, c, e in notes:
        bycol[c].append(t)
    jacks = pairs = 0
    for c, ts in bycol.items():
        for a, b in zip(ts, ts[1:]):
            pairs += 1
            if b - a < 120:
                jacks += 1
    out['jackRate'] = jacks / max(1, pairs)
    # 真 jack：相邻行（<=120ms）共用列的比例（站点 D 的主要驱动项）
    rj = elig = 0
    for i in range(len(rows) - 1):
        if rows[i + 1][0] - rows[i][0] > 120:
            continue
        elig += 1
        if set(rows[i][1]) & set(rows[i + 1][1]):
            rj += 1
    out['rowJack'] = rj / max(1, elig)
    # 列分布
    colc = Counter(c for t, c, e in notes)
    n = sum(colc.values())
    out['colShare'] = [colc[c] / n for c in range(4)]
    import math
    out['colEntropy'] = -sum(p * math.log(p + 1e-12) for p in out['colShare']) / math.log(4)
    out['handBalance'] = (colc[0] + colc[1]) / n
    # 4-gram 丰富度
    sig = [tuple(cs) for t, cs in rows]
    n4 = [tuple(sig[i:i + 4]) for i in range(len(sig) - 3)]
    c4 = Counter(n4)
    out['ngram4Distinct'] = len(c4) / max(1, len(n4))
    out['ngram4TopShare'] = (c4.most_common(1)[0][1] / max(1, len(n4))) if c4 else 0
    # 连续同列
    runs, cur = [], 1
    for i in range(len(rows) - 1):
        same = rows[i][1][-1] == rows[i + 1][1][0] and rows[i + 1][0] - rows[i][0] < 200
        cur = cur + 1 if same else 1
        runs.append(cur)
    out['maxConsec'] = max(runs) if runs else 0
    # LN
    lns = [(t, c, e) for t, c, e in notes if e]
    out['lnRatio'] = len(lns) / len(notes)
    if lns:
        L = sorted((e - t) for t, c, e in lns)
        out['lnLenP50'] = L[len(L) // 2]
        out['lnLenP90'] = L[int(len(L) * 0.9)]
        ts = [t for t, c, e in lns]
        best = j = 0
        for i in range(len(ts)):
            while ts[i] - ts[j] > 4000:
                j += 1
            best = max(best, i - j + 1)
        out['lnWall4s'] = best
    else:
        out['lnLenP50'] = out['lnLenP90'] = 0
        out['lnWall4s'] = 0
    # 密度起伏（4s 窗 rps 的变异系数）
    seg = defaultdict(int)
    for t, cs in rows:
        seg[t // 4000] += 1
    vals = [v for k, v in sorted(seg.items()) if k > 0]
    if len(vals) > 5:
        m = sum(vals) / len(vals)
        out['rpsCV'] = (st.pstdev(vals) / m) if m else 0
    else:
        out['rpsCV'] = 0
    return out


def agg(ms, keys):
    res = {}
    for k in keys:
        v = sorted(x[k] for x in ms)
        if not v:
            continue
        res[k] = {'p10': round(v[int(len(v) * 0.1)], 3),
                  'p50': round(v[len(v) // 2], 3),
                  'p90': round(v[int(len(v) * 0.9)], 3),
                  'mean': round(sum(v) / len(v), 3)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charts', default='out/charts')
    ap.add_argument('--features', default='out/feature-db.jsonl')
    ap.add_argument('--per-bucket', type=int, default=40)
    ap.add_argument('-o', default='out/human-baseline.json')
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()

    feats = {}
    for l in open(args.features):
        d = json.loads(l)
        feats[d['id']] = d

    buckets = defaultdict(list)
    for path in glob.glob(os.path.join(args.charts, '*.osu')):
        cid = os.path.splitext(os.path.basename(path))[0]
        d = feats.get(cid)
        if not d or not d.get('ett'):
            continue
        ett = d['ett']
        if 22 <= ett < 26:
            b = 'D13-14'
        elif 26 <= ett < 29:
            b = 'D14-15'
        elif 29 <= ett < 31.5:
            b = 'D15-15.7'
        elif 31.5 <= ett < 34:
            b = 'D15.7-16.8'
        else:
            continue
        if d.get('ln_ratio', 0) > 0.15:
            b = b + '/LN'
        buckets[b].append(path)

    rng = random.Random(args.seed)
    out = {'perBucket': {}, 'charts': {}}
    for b, lst in sorted(buckets.items()):
        rng.shuffle(lst)
        ms = []
        for p in lst:
            if len(ms) >= args.per_bucket:
                break
            m = metrics(p)
            if m and 0.5 < m['nps'] < 40:
                ms.append(m)
                out['charts'][m['id']] = dict(m, bucket=b)
        if ms:
            keys = ['nps', 'rps', 'notesPerRow', 'chordRowRate', 'jackRate', 'rowJack', 'colEntropy',
                    'handBalance', 'ngram4Distinct', 'ngram4TopShare', 'maxConsec',
                    'lnRatio', 'lnLenP50', 'lnLenP90', 'lnWall4s', 'rpsCV']
            g = {k: round(sum(x['gapHist'][k] for x in ms) / len(ms), 4) for k in GAP_LABELS}
            out['perBucket'][b] = {'n': len(ms), 'stats': agg(ms, keys), 'gapHist': g}
            print(f"{b:16s} n={len(ms):3d}  nps {out['perBucket'][b]['stats']['nps']['p50']:5.1f}  "
                  f"rps {out['perBucket'][b]['stats']['rps']['p50']:5.1f}  "
                  f"jack {out['perBucket'][b]['stats']['jackRate']['p50']:.3f}  "
                  f"gap<100 {g['<30']+g['30-60']+g['60-100']:.3f}")
    json.dump(out, open(args.o, 'w'), ensure_ascii=False)
    print('写出', args.o)


if __name__ == '__main__':
    main()
