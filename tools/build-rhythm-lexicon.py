#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build-rhythm-lexicon: 从人类谱库学「节奏词汇表」。

现状问题（v10 诊断）：painter 只按「段落能量配额 + 1/32 半格补偿」撒行，
节奏层完全没有人类结构——出来的谱是流水账：40ms 半格占两成、行间隔分布
与人类不符、和弦不落在重音上。

本工具学的东西（每键型桶）：
  1. **小节节奏模板**：16 个 1/16 格里哪几格有行（occupancy mask）+ 哪几格是
     和弦（chord mask）。人类谱的小节节奏是可枚举的词汇表，不是任意撒点。
  2. **每桶的行/小节分布**：密度控制（配合能量曲线做段内起伏）。
  3. **模板词表熵**：模板集中度（太低=机械循环，太高=没有动机）。
  4. **相位对齐**：每张谱先搜索 0..15 的小节相位，让强拍格有最多行，
     避免把模板学歪。

输出 out/rhythm-lexicon.json：
  {
    "buckets": {"rice/technical/high": {
        "templates": [{"occ": [[pos,size],...], "w": 0.012}, ...],
        "rowsPerBar": {"p10":..,"p50":..,"p90":..},
        "emptyBarRate": 0.05, ...
    }},
    "global": {...}
  }
用法：
  python3 tools/build-rhythm-lexicon.py --charts out/charts --features out/feature-db.jsonl \
      --per-bucket 60 -o out/rhythm-lexicon.json
"""
import argparse
import glob
import json
import os
import random
import statistics as st
from collections import Counter, defaultdict

CELL_PER_BAR = 16          # 1/16 基准网格（1/32 只留给音频里真实的连续击打）
# 重音权重：拍点 > 8 分 > 16 分 > 32 分（32 格/小节）
STRONG = [3.0, 0, 0, 0, 2.0, 0, 0, 0, 2.6, 0, 0, 0, 2.0, 0, 0, 0]


def parse_chart(path):
    """→ (notes[(t,col,end)], bpm, offset) ；只 4K、单 BPM。"""
    txt = open(path, encoding='utf-8', errors='ignore').read()
    if '[HitObjects]' not in txt:
        return None
    lines = txt.splitlines()
    mode = next((l for l in lines if l.startswith('Mode:')), 'Mode: 0')
    cs = next((l for l in lines if l.startswith('CircleSize:')), 'CircleSize: 4')
    try:
        if int(mode.split(':')[1].strip()) != 3:
            return None
        if round(float(cs.split(':')[1].strip())) != 4:
            return None
    except ValueError:
        return None
    if '[TimingPoints]' not in txt:
        return None
    tp = txt.split('[TimingPoints]')[1].split('[HitObjects]')[0]
    reds = []
    for l in tp.strip().splitlines():
        p = l.split(',')
        if len(p) < 7:
            continue
        try:
            if p[6].strip() == '1' and float(p[1]) > 0:
                reds.append((float(p[0]), float(p[1])))
        except ValueError:
            continue
    if not reds:
        return None
    beats = {round(b, 3) for _, b in reds}
    if len(beats) > 1:          # 变速谱：模板相位学不准，跳过（占比很低）
        return None
    offset, beat_ms = reds[0]
    if not (60 <= 60000 / beat_ms <= 300):
        return None
    objs = []
    body = txt.split('[HitObjects]')[1]
    for l in body.strip().splitlines():
        p = l.split(',')
        if len(p) < 5 or not p[2].lstrip('-').isdigit():
            continue
        col = int(int(p[0]) / 128)
        if col > 3:
            continue
        t = int(p[2])
        end = None
        if len(p) > 5:
            e = p[5].split(':')[0]
            if e.isdigit() and int(e) > t:
                end = int(e)
        objs.append((t, col, end))
    if len(objs) < 200:
        return None
    return objs, 60000.0 / beat_ms, offset


def bar_templates(objs, bpm, offset, max_bars=2000):
    """→ (templates Counter{ (occ tuple[16], chord tuple[16]) }, rows_per_bar list, n_empty)"""
    cell_ms = 60000.0 / bpm / (CELL_PER_BAR / 4.0)
    cells = defaultdict(list)
    for t, c, e in objs:
        cells[int(round((t - offset) / cell_ms))].append(c)
    if not cells:
        return None
    keys = sorted(cells)
    # 相位搜索：让强拍格行数最多
    best_phase, best_score = 0, -1
    for ph in range(CELL_PER_BAR):
        sc = sum(len(cells[k]) * STRONG[(k - ph) % CELL_PER_BAR] for k in keys)
        if sc > best_score:
            best_score, best_phase = sc, ph
    # 分小节
    bars = {}
    for k in keys:
        b = (k - best_phase) // CELL_PER_BAR
        pos = (k - best_phase) % CELL_PER_BAR
        bars.setdefault(b, {})[pos] = len(cells[k])
    if len(bars) > max_bars:
        return None
    cnt = Counter()
    rpb = []
    n_empty = 0
    bs = sorted(bars)
    for b in range(bs[0], bs[-1] + 1):
        occ = bars.get(b, {})
        mask = _bits(occ.keys())
        # 和弦位：该格 >=2 个音符
        ch = _bits([p for p, n in occ.items() if n >= 2])
        cnt[(mask, ch)] += 1
        rpb.append(len(occ))
        if not occ:
            n_empty += 1
    return cnt, rpb, n_empty, best_phase


def _bits(positions):
    v = 0
    for p in positions:
        v |= 1 << p
    return v


def bits_to_pos(v):
    return [i for i in range(CELL_PER_BAR) if v >> i & 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charts', default='out/charts')
    ap.add_argument('--features', default='out/feature-db.jsonl')
    ap.add_argument('--per-bucket', type=int, default=60)
    ap.add_argument('--top-templates', type=int, default=160)
    ap.add_argument('-o', default='out/rhythm-lexicon.json')
    ap.add_argument('--seed', type=int, default=5)
    args = ap.parse_args()

    feats = {}
    for l in open(args.features):
        d = json.loads(l)
        feats[d['id']] = d

    def diff_bucket(ett):
        return 'low' if ett < 20 else 'mid' if ett <= 27 else 'high'

    def style_of(d):
        if d.get('ln'):
            return 'ln'
        if d.get('rc'):
            return 'rice'
        return 'ln' if d.get('ln_ratio', 0) > 0.02 else 'rice'

    def pattern_of(d):
        for k in ['stream', 'jumpstream', 'handstream', 'chordjack', 'technical', 'stamina', 'jackspeed']:
            if d.get(k):
                return k
        return 'technical'

    buckets = defaultdict(list)
    for path in glob.glob(os.path.join(args.charts, '*.osu')):
        cid = os.path.splitext(os.path.basename(path))[0]
        d = feats.get(cid)
        if not d or not d.get('ett'):
            continue
        buckets[(style_of(d), pattern_of(d), diff_bucket(d['ett']))].append((path, d['ett']))

    rng = random.Random(args.seed)
    out = {'alpha': 0.7, 'buckets': {}, 'global': {}}
    all_cnt, all_rpb = Counter(), []
    for key, lst in sorted(buckets.items()):
        lst.sort(key=lambda x: abs(x[1] - {'low': 15, 'mid': 24, 'high': 31}[key[2]]))
        rng.shuffle(lst)
        cnt, rpb, n_empty, n_bars = Counter(), [], 0, 0
        used = 0
        for path, _ in lst[:args.per_bucket * 3]:
            if used >= args.per_bucket:
                break
            p = parse_chart(path)
            if not p:
                continue
            res = bar_templates(*p)
            if not res:
                continue
            c2, r2, e2, _ = res
            cnt.update(c2)
            rpb += r2
            n_empty += e2
            n_bars += len(r2)
            used += 1
        if used < 6 or not rpb:
            continue
        tot = sum(cnt.values())
        top = cnt.most_common(args.top_templates)
        wtot = sum(w for _, w in top) or 1
        templates = []
        for (mask, ch), w in top:
            templates.append({'occ': bits_to_pos(mask), 'chord': bits_to_pos(ch),
                              'w': round(w / wtot, 4),
                              'rows': len(bits_to_pos(mask)),
                              'chords': len(bits_to_pos(ch))})
        rpb_s = sorted(rpb)
        entry = {
            'n_charts': used, 'n_bars': n_bars,
            'emptyBarRate': round(n_empty / max(1, n_bars), 4),
            'rowsPerBar': {'p10': rpb_s[int(len(rpb_s) * .1)], 'p50': rpb_s[len(rpb_s) // 2],
                           'p90': rpb_s[int(len(rpb_s) * .9)], 'mean': round(sum(rpb) / len(rpb), 2)},
            'templateEntropy': round(_entropy(cnt), 3),
            'templates': templates,
        }
        out['buckets']['/'.join(key)] = entry
        all_cnt.update(cnt)
        all_rpb += rpb
        print(f"{'/'.join(key):28s} charts={used:3d} bars={n_bars:5d} rows/bar p50={entry['rowsPerBar']['p50']:2d} "
              f"空小节={entry['emptyBarRate']*100:4.1f}% 模板熵={entry['templateEntropy']:.2f} 模板数={len(templates)}")
    # 全局（兜底）
    tot = sum(all_cnt.values())
    top = all_cnt.most_common(300)
    wtot = sum(w for _, w in top) or 1
    rpb_s = sorted(all_rpb)
    out['global'] = {
        'n_bars': len(all_rpb),
        'emptyBarRate': round(sum(1 for x in all_rpb if x == 0) / max(1, len(all_rpb)), 4),
        'rowsPerBar': {'p10': rpb_s[int(len(rpb_s) * .1)], 'p50': rpb_s[len(rpb_s) // 2],
                       'p90': rpb_s[int(len(rpb_s) * .9)], 'mean': round(sum(all_rpb) / len(all_rpb), 2)},
        'templates': [{'occ': bits_to_pos(m), 'chord': bits_to_pos(c), 'w': round(w / wtot, 4),
                       'rows': len(bits_to_pos(m)), 'chords': len(bits_to_pos(c))} for (m, c), w in top],
    }
    json.dump(out, open(args.o, 'w'), ensure_ascii=False)
    print(f'节奏词汇表: {len(out["buckets"])} 桶 + 全局 → {args.o}')


def _entropy(counter):
    tot = sum(counter.values()) or 1
    import math
    return -sum((v / tot) * math.log(v / tot + 1e-12) for v in counter.values())


if __name__ == '__main__':
    main()
