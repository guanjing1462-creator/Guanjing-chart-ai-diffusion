#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""键型指纹库：从人类谱库提取每 (大类×子键型×难度档) 的写法统计。

这就是「培训」——36837 张人类谱浓缩成每键型的生成参数：
  - col_trans[4][4]   列转移矩阵（stream 是四列轮转、chordjack 是和弦连打、
                       tech 是跳跃+休止——人类手感的本质就是这个矩阵）
  - row_size_dist     行大小分布（单/双/三/四押占比）
  - row_gap_dist      行间隔分布（按 1/16 格数分桶——密度节奏感）
  - ln_ratio / ln_len_dist / ln_head_chord  LN 长度与位置习惯
  - hand_alt          相邻行跨手交替率
  - nps               密度

输出 out/style-fingerprints.json，pattern-painter.py 消费。
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

CELL = 81.081  # 1/16 @185bpm 之外按谱面 BPM 算


def parse_chart(path):
    """→ (objs[(t, col, end|None)], bpm)。只取 4K。"""
    txt = open(path, encoding='utf-8', errors='ignore').read()
    if '[HitObjects]' not in txt:
        return None, None
    mode = [l for l in txt.splitlines() if l.startswith('Mode:')]
    cs = [l for l in txt.splitlines() if l.startswith('CircleSize:')]
    if mode and mode[0].split(':')[1].strip() != '3':
        return None, None
    if cs and round(float(cs[0].split(':')[1].strip())) != 4:
        return None, None
    bpm = 120.0
    for l in txt.splitlines():
        if l.startswith('BPMDisplay:') or l.startswith('BPM:'):
            continue
    tp = txt.split('[TimingPoints]')[1].split('[HitObjects]')[0] if '[TimingPoints]' in txt else ''
    reds = [l.split(',') for l in tp.strip().splitlines() if len(l.split(',')) > 6 and l.split(',')[6].strip() == '1']
    if reds:
        beat_len = float(reds[0][1])
        if beat_len > 0:
            bpm = round(60000.0 / beat_len, 2)
    body = txt.split('[HitObjects]')[1]
    objs = []
    for l in body.strip().splitlines():
        p = l.split(',')
        if len(p) < 5 or not p[0].lstrip('-').isdigit():
            continue
        col = int(p[0]) // 128
        if col > 3:
            continue
        t = int(p[2])
        end = None
        if len(p) > 5:
            e = p[5].split(':')[0]
            if e not in ('0', ''):
                end = int(e)
        objs.append((t, col, end))
    return objs, bpm


def fingerprint(objs, bpm):
    beat = 60000.0 / (bpm or 120)
    cell = beat / 4
    rows = defaultdict(list)
    for t, c, e in objs:
        rows[t].append((c, e))
    rts = sorted(rows)

    # 列转移（相邻行、取行内最后一列 → 下一行第一列的转移；单音行才更能代表键型）
    trans = [[0] * 4 for _ in range(4)]
    prev_single = None
    for t in rts:
        cs = sorted(c for c, _ in rows[t])
        if len(cs) == 1:
            if prev_single is not None:
                trans[prev_single][cs[0]] += 1
            prev_single = cs[0]
        else:
            prev_single = None

    # 行大小
    rs = Counter(min(len(v), 4) for v in rows.values())
    tot_r = sum(rs.values())
    row_size = {str(k): rs[k] / tot_r for k in range(1, 5)}

    # 行间隔（格数桶：1,2,3,4-5,6-7,8+）
    gaps = Counter()
    for a, b in zip(rts, rts[1:]):
        g = max(1, round((b - a) / cell))
        gaps[min(g, 8)] += 1
    tot_g = sum(gaps.values())
    gap_keys = [1, 2, 3, 4, 6, 8]
    gap_bin = {1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 6, 7: 6, 8: 8}
    row_gap = {str(k): sum(v for gk, v in gaps.items() if gap_bin[gk] == k) / tot_g
               for k in gap_keys}

    # LN
    lns = [(t, c, e) for t, cs in rows.items() for c, e in cs if e]
    n_notes = len(objs)
    ln_ratio = len(lns) / max(1, n_notes)
    ln_len = Counter()
    for t, c, e in lns:
        ln_len[min(8, max(1, round((e - t) / cell)))] += 1
    tot_l = sum(ln_len.values()) or 1
    ln_len_dist = {str(k): ln_len[k] / tot_l for k in gap_keys}
    ln_head_chord = sum(1 for t, c, e in lns if len(rows[t]) > 1) / max(1, len(lns))

    # 手交替（相邻行主力列分属左右手）
    alt = same = 0
    for a, b in zip(rts, rts[1:]):
        ca = sorted(c for c, _ in rows[a])[0]
        cb = sorted(c for c, _ in rows[b])[0]
        if (ca < 2) != (cb < 2):
            alt += 1
        else:
            same += 1
    hand_alt = alt / max(1, alt + same)

    span_s = (rts[-1] - rts[0]) / 1000 if len(rts) > 1 else 1
    return {
        'col_trans': trans,
        'row_size': row_size,
        'row_gap': row_gap,
        'ln_ratio': round(ln_ratio, 4),
        'ln_len_dist': ln_len_dist,
        'ln_head_chord': round(ln_head_chord, 3),
        'hand_alt': round(hand_alt, 3),
        'nps': round(n_notes / max(1.0, span_s), 2),
        'chord_rate': round(n_notes / max(1, len(rts)), 3),
        'n_charts_sampled': 1,
    }


def merge(fps):
    """多张谱的指纹加权平均。"""
    n = len(fps)
    out = {'col_trans': [[0.0] * 4 for _ in range(4)]}
    keys_num = ['ln_ratio', 'ln_head_chord', 'hand_alt', 'nps', 'chord_rate']
    for k in keys_num:
        out[k] = sum(f[k] for f in fps) / n
    for key in ['row_size', 'row_gap', 'ln_len_dist']:
        out[key] = {kk: sum(f[key][kk] for f in fps) / n for kk in fps[0][key]}
    for i in range(4):
        for j in range(4):
            out['col_trans'][i][j] = sum(f['col_trans'][i][j] for f in fps) / n
    out['n_charts'] = n
    return out


def extract_patterns(objs, bpm, top_k=48, n_lens=(4, 6)):
    """图案短语库：连续单音行的列 n-gram（n=4/6）。人类手感的核心是
    图案块——梯子/轮转/折返，块越长越接近真实手部动作组。"""
    beat = 60000.0 / (bpm or 120)
    cell = beat / 4
    rows = defaultdict(list)
    for t, c, e in objs:
        rows[t].append(c)
    rts = sorted(rows)
    seq = []
    for t in rts:
        cs = rows[t]
        seq.append((t, cs[0]) if len(cs) == 1 else (t, None))
    grams = Counter()
    run = []
    prev_t = None
    for t, c in seq:
        if c is not None and (prev_t is None or t - prev_t <= cell * 3):
            run.append(c)
        else:
            run = [c] if c is not None else []
        for N in n_lens:
            if len(run) >= N:
                for i in range(len(run) - N + 1):
                    grams[(N, tuple(run[i:i + N]))] += 1
        prev_t = t
    out = []
    for N in n_lens:
        bucket = [(g, n) for (nn, g), n in grams.items() if nn == N]
        tot = sum(n for _, n in bucket) or 1
        out += [(list(g), n / tot, N) for g, n in bucket]
    out.sort(key=lambda x: -x[1])
    return [(g, w, N) for g, w, N in out[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charts', default='out/charts')
    ap.add_argument('--features', default='out/feature-db.jsonl')
    ap.add_argument('--per-bucket', type=int, default=40, help='每桶抽几张')
    ap.add_argument('-o', default='out/style-fingerprints.json')
    args = ap.parse_args()

    feats = {}
    for l in open(args.features):
        d = json.loads(l)
        feats[d['id']] = d

    # 桶：(大类, pattern, 难度档)。大类用 rc/hb/ln，难度档按 ett 三档。
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
        if not d:
            continue
        buckets[(style_of(d), pattern_of(d), diff_bucket(d['ett']))].append((path, d['ett']))

    rng = random.Random(7)
    out = {}
    for key, lst in sorted(buckets.items()):
        lst.sort(key=lambda x: abs(x[1] - {  # 每桶取最接近桶中位的 ett 的谱
            'low': 15, 'mid': 24, 'high': 31}[key[2]]))
        rng.shuffle(lst)
        fps = []
        for path, _ in lst[:args.per_bucket * 2]:
            objs, bpm = parse_chart(path)
            if not objs or len(objs) < 150 or bpm < 100 or bpm > 260:
                continue
            fp = fingerprint(objs, bpm)
            if fp['nps'] < 1 or fp['chord_rate'] > 2.6:
                continue
            fps.append(fp)
            if len(fps) >= args.per_bucket:
                break
        if len(fps) >= 6:
            out['/'.join(key)] = merge(fps)
            # 图案短语库：该桶样谱的 n-gram 合并（n=4/6，带长度标签）
            grams = Counter()
            for path, _ in lst[:args.per_bucket * 2]:
                objs, bpm = parse_chart(path)
                if not objs or bpm < 100 or bpm > 260:
                    continue
                for g, w, N in extract_patterns(objs, bpm):
                    grams[(N, tuple(g))] += w
            top = sorted(grams.items(), key=lambda x: -x[1])[:48]
            tot = sum(w for _, w in top) or 1
            out['/'.join(key)]['patterns'] = [
                {'g': list(g), 'n': N, 'w': round(w / tot, 4)} for (N, g), w in top]

    json.dump(out, open(args.o, 'w'), ensure_ascii=False, indent=1)
    print(f'指纹库: {len(out)} 个键型桶 → {args.o}')
    for k in sorted(out):
        f = out[k]
        print(f'  {k:28s} n={f["n_charts"]:3d}  nps {f["nps"]:5.1f}  和弦率 {f["chord_rate"]:.2f}  '
              f'LN {f["ln_ratio"]*100:4.1f}%  手交替 {f["hand_alt"]:.2f}')


if __name__ == '__main__':
    main()
