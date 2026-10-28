#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pattern-painter：节奏骨架 × 键型指纹 → 完整谱面。

两阶段架构的着色阶段——键型与难度分配不再是扩散模型的弱响应，
而是从人类谱库统计出来的显式生成参数：
  1. grid 吸附：骨架时间戳吸到 1/16 格（治「节奏糊」）
  2. 行构成：按指纹 row_size_dist 分配单/双/三押
  3. 列序列：按指纹 col_trans 马尔科夫采样（stream=轮转、jack=同列、
     tech=跳跃休止——人类手感的本质）
  4. LN：按 ln_ratio/ln_len_dist 插入（站点上限内），释放后留 1 格
  5. 手均衡：列采样带弱手偏置
"""
import argparse
import json
import os
import random
from collections import defaultdict

BPM = 185.0
BEAT = 60000.0 / BPM
CELL = BEAT / 4
OFFSET = 1263.28
LN_CAP_RATIO = 0.175     # 站点 Koishi 18% 硬上限内留余量
LN_MIN_CELL = 1          # 最短 1 格
LN_MAX_CELL = 4          # 最长 1 拍（人类 p90≈3.3 格）


def snap_cells(skeleton):
    """骨架时间戳 → 1/16 格序列。冲突顺延 ±2 格（自由时间轴的行
    会挤进同一格，直接去重会丢掉近一半的密度——v4 初版踩过）。"""
    occupied = {}
    dropped = 0
    for t in sorted(skeleton):
        base = round((t - OFFSET) / CELL)
        for d in (0, 1, -1, 2, -2, 3):
            c = base + d
            if c >= 0 and c not in occupied:
                occupied[c] = t
                break
        else:
            dropped += 1
    return sorted(occupied), dropped


def add_subcells(cells, fp, seg_sec=4.0, dps=None):
    """段落形状保持的 1/32 半格补偿：全曲等比放大到指纹 nps，
    但每段最多 +50%（v6 首轮全曲均摊，密集区行频爆炸 → D 超
    站点可测上限）。半格只插在行间隔 ≤2 格的位置（人类的 1/32
    是爆发段行为）。"""
    cellset = set(cells)
    seg_len = max(1, int(seg_sec * 1000 / CELL))
    # 分段（按格索引）
    segs = {}
    for c in cells:
        segs.setdefault(int(c // seg_len), []).append(c)
    dur_s = (cells[-1] - cells[0]) * CELL / 1000 if len(cells) > 1 else 1
    # 指纹 nps 是「音符/s」；补偿目标是行频 = nps ÷ 和弦率，
    # 音符数由行数×和弦率自然长出来（否则双重放大，难度旋钮失效）
    row_nps = fp['nps'] / max(0.5, fp['chord_rate'])
    if dps:
        row_nps = dps          # 显式行频目标（难度精确锚定用）
    scale = row_nps * dur_s / max(1, len(cells))
    added = []
    rnd = random.Random(31)
    for k in sorted(segs):
        seg = segs[k]
        target = min(len(seg) * scale, len(seg) * 2.2)
        need = int(target - len(seg))
        if need <= 0:
            continue
        sset = set(seg)
        dense = [c for i, c in enumerate(seg)
                 if (c + 1 in sset or c - 1 in sset or
                     (i + 1 < len(seg) and seg[i + 1] - c <= 2))]
        rnd.shuffle(dense)
        for c in dense:
            if need <= 0:
                break
            for half in (c + 0.5, c - 0.5):
                if half >= 0 and half not in added and round(half * 2) not in cellset:
                    added.append(half)
                    need -= 1
                    break
    return sorted(cells + [round(h * 2) / 2 for h in added])


def norm_row(dist):
    """{'1':p1,...} → [(k, cum_p)]"""
    items = sorted((int(k), v) for k, v in dist.items())
    out, cum = [], 0.0
    for k, v in items:
        cum += v
        out.append((k, cum))
    s = out[-1][1]
    return [(k, c / s) for k, c in out]


def sample_dist(rnd, dist):
    r = rnd.random()
    for k, cum in norm_row(dist):
        if r <= cum:
            return k
    return norm_row(dist)[-1][0]


def sample_trans(rnd, trans, prev, avoid=()):
    """按转移矩阵采样下一列；避开本行已用列。"""
    row = trans[prev]
    tot = sum(row)
    if tot < 4:   # 该列几乎没出现过的退化情况 → 均匀
        cands = [c for c in range(4) if c not in avoid]
        return rnd.choice(cands) if cands else prev
    r = rnd.random() * tot
    cum = 0.0
    for c, w in enumerate(row):
        cum += w
        if r <= cum and c not in avoid:
            return c
    cands = [c for c in range(4) if c not in avoid]
    return rnd.choice(cands) if cands else prev


CHORD_PAIRS_SAME = [(0, 1), (2, 3)]          # 同手双押
CHORD_PAIRS_ALT = [(0, 2), (1, 3), (0, 3), (1, 2)]
CHORD_TRI = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def pick_chord(rnd, size, hand_alt, used=()):
    """按手交替率加权选和弦组合。"""
    if size >= 4:
        return [0, 1, 2, 3]
    if size == 3:
        cands = [t for t in CHORD_TRI if not set(t) <= set(used)]
        return list(rnd.choice(cands)) if cands else [0, 1, 2]
    same = CHORD_PAIRS_SAME
    alt = [p for p in CHORD_PAIRS_ALT if not set(p) <= set(used)]
    cands = same + alt
    cands = [p for p in cands if not set(p) <= set(used)] or cands
    if not cands:
        return [0, 1]
    w_same = 1 - hand_alt
    w_alt = hand_alt / max(1, len(alt))
    weights = [w_same if p in same else w_alt for p in cands]
    return list(rnd.choices(cands, weights=weights)[0])


def paint(skeleton, fp, seed=7, target_ln=None, structure=None, dps=None):
    rnd = random.Random(seed)
    trans = fp['col_trans']
    row_size = fp['row_size']
    ln_ratio = fp['ln_ratio'] if target_ln is None else target_ln
    ln_len = fp['ln_len_dist']
    hand_alt = fp['hand_alt']
    ln_head_chord = fp.get('ln_head_chord', 0.5)

    # 1) grid 吸附（冲突顺延）+ 密集区 1/32 半格细分
    base_cells, dropped = snap_cells(skeleton)
    if dropped:
        print(f'  网格顺延后仍丢弃 {dropped} 行（网格容量饱和）')
    cells = add_subcells(base_cells, fp, dps=dps)
    n_sub = len(cells) - len(base_cells)
    if n_sub:
        print(f'  1/32 半格细分: +{n_sub} 行（补指纹 nps 缺口）')
    rows = []

    load = {0: 0, 1: 0, 2: 0, 3: 0}
    prev_col = rnd.randrange(4)
    ln_slots = []          # (cell, col, len_cells)
    prev_cell = None
    # ---- 两阶段填充：行计划 + 列填充。三套列来源按优先级：
    #      ①段落记忆（重复乐段第二遍呼应第一遍：60% 原样 + 40% 变奏）
    #      ②人类图案块（4/6-gram，tech=折返、stream=轮转；接龙约束——
    #        块首列优先等于上一行末列，消灭硬拼接）
    #      ③马尔科夫（兜底与和弦行） ----
    structure = structure or {}
    seg_cells = structure.get('seg_cells', 0)
    seg_ids = structure.get('seg_ids', [])

    plan = []
    for i, cell in enumerate(cells):
        want = sample_dist(rnd, row_size) if rnd.random() < 0.9 else 1
        seg_k = int(cell // seg_cells) if seg_cells else -1
        plan.append((cell, want, seg_k, None))
    # 段内行序号（对齐用）：等价段按格序号归一
    seg_pos = {}
    pos_counter = {}
    for i, (cell, want, seg_k, _) in enumerate(plan):
        key = seg_ids[seg_k] if 0 <= seg_k < len(seg_ids) else -1
        pos = pos_counter.get(key, 0)
        plan[i] = (cell, want, key, pos)
        pos_counter[key] = pos + 1
    seg_memory = {}   # (group, pos) -> 列

    patterns = fp.get('patterns') or []
    pat4 = [(p['g'], p['w']) for p in patterns if p['n'] == 4]
    pat6 = [(p['g'], p['w']) for p in patterns if p['n'] == 6]
    pat_cum, acc = [], 0.0
    for g, w in pat4 + pat6:
        acc += w
        pat_cum.append((g, acc))
    pat_tot = acc or 1

    def sample_pattern(head_col=None):
        """加权抽图案块；接龙：优先首列 == head_col 的块（2 倍权重）。"""
        best, bw = None, -1.0
        for g, w in pat4 + pat6:
            ww = w * (2.0 if head_col is not None and g[0] == head_col else 1.0)
            ww *= rnd.random()
            if ww > bw:
                best, bw = g, ww
        return best

    rows = []
    i = 0
    while i < len(plan):
        cell, want, gkey, pos = plan[i]
        # ①段落记忆：重复段且第一遍已写过同位置
        if gkey >= 0 and (gkey, pos) in seg_memory and rnd.random() < 0.6:
            cols = list(seg_memory[(gkey, pos)])
            if rnd.random() < 0.4:   # 变奏：单音行移位列
                if len(cols) == 1 and rnd.random() < 0.5:
                    cols = [(cols[0] + rnd.choice((1, 3))) % 4]
            prev_col = cols[-1]
            prev_cell = cell
            for c in cols:
                load[c] += 1
            rows.append((cell, cols))
            i += 1
            continue
        # ②图案块：连续单音 run ≥4 行
        if want == 1 and patterns:
            run, j = [], i
            while j < len(plan) and plan[j][1] == 1 and len(run) < 12 and \
                    (j == i or plan[j][0] - plan[j - 1][0] <= 2):
                run.append(plan[j][0])
                j += 1
            if len(run) >= 4:
                k2 = 0
                while k2 < len(run):
                    g = sample_pattern(head_col=prev_col)
                    take = min(len(g), len(run) - k2)
                    for m in range(take):
                        cols = [g[m]]
                        load[cols[0]] += 1
                        rows.append((run[k2 + m], cols))
                        # 记进段落记忆（重复段呼应）
                        if gkey >= 0:
                            seg_memory[(gkey, pos + k2 + m)] = cols
                        prev_col = cols[-1]
                    k2 += take
                prev_cell = run[-1]
                i = j
                continue
        # ③常规行：马尔科夫 + 和弦
        dense_gap = prev_cell is not None and (cell - prev_cell) <= 2
        cols, tries = [], 0
        while len(cols) < want and tries < 8:
            if not cols:
                c = sample_trans(rnd, trans, prev_col)
                if dense_gap and c == prev_col and rnd.random() < 0.7:
                    c = sample_trans(rnd, trans, prev_col, avoid=(prev_col,))
            else:
                c = sample_trans(rnd, trans, cols[-1], avoid=cols)
            if c not in cols:
                cols.append(c)
            tries += 1
        if not cols:
            continue
        # 弱手偏置：左手少 → 优先左手（简单版负载均衡）
        if rnd.random() < 0.25:
            weak = 0 if load[0] + load[1] < load[2] + load[3] else 2
            if weak not in cols and rnd.random() < 0.5:
                cols[-1] = weak
        for c in cols:
            load[c] += 1
        prev_col = cols[-1]
        prev_cell = cell
        rows.append((cell, cols))
        if gkey >= 0:
            seg_memory[(gkey, pos)] = cols
        # LN 头候选
        if ln_ratio > 0.02 and rnd.random() < ln_ratio:
            if len(cols) > 1 and rnd.random() > ln_head_chord:
                i += 1
                continue
            ln_slots.append((cell, rnd.choice(cols)))
        i += 1

    # 2) LN 长度 + 站点占比钳制 + 释放间隔
    max_ln = int(len(rows) * min(ln_ratio, LN_CAP_RATIO))
    rnd.shuffle(ln_slots)
    ln_slots = ln_slots[:max_ln]
    row_at = {c: cols for c, cols in rows}
    col_rows = defaultdict(list)
    for cell, cols in rows:
        for c in cols:
            col_rows[c].append(cell)
    ln_map = {}
    for cell, c in ln_slots:
        L = sample_dist(rnd, ln_len)
        L = max(LN_MIN_CELL, min(LN_MAX_CELL, L))
        nxt = [x for x in col_rows[c] if x > cell]
        if nxt:
            avail = nxt[0] - cell - 1          # 释放后留 1 格
            L = min(L, max(1, avail))
        if L < LN_MIN_CELL:
            continue
        ln_map[(cell, c)] = L

    # 3) 输出
    out = []
    for cell, cols in rows:
        t = int(OFFSET + cell * CELL)
        for c in cols:
            x = int((c + 0.5) * 128)
            L = ln_map.get((cell, c))
            if L:
                out.append(f'{x},192,{t},128,0,{int(t + L * CELL)}:0:0:0:0:')
            else:
                out.append(f'{x},192,{t},1,0,0:0:0:0:')
    out.sort(key=lambda l: (int(l.split(',')[2]), int(l.split(',')[0])))
    return out, len(rows), len(ln_map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skeleton', required=True, help='raw .osu（gen-ai-chart 的模型骨架）')
    ap.add_argument('--fingerprints', default='out/style-fingerprints.json')
    ap.add_argument('--style', required=True, help='桶名，如 rice/technical/mid')
    ap.add_argument('--diff', choices=['low', 'mid', 'high'], default=None,
                    help='覆盖难度档（默认取 style 里带的）')
    ap.add_argument('--ln-cap', type=float, default=None, help='覆盖 LN 率（站点合规用）')
    ap.add_argument('--structure', default=None,
                    help='skeleton-from-audio 输出的段落结构 JSON（重复段呼应）')
    ap.add_argument('--dps', type=float, default=None,
                    help='显式行频目标（行/s），覆盖难度桶推导——难度精确锚定')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--version', required=True)
    ap.add_argument('--title', default='INTERNET YAMERO')
    ap.add_argument('--artist', default='KOTOKO')
    ap.add_argument('--creator', default='mug-prototype')
    ap.add_argument('--audio-name', default='INTERNET YAMERO.ogg')
    ap.add_argument('-o', required=True)
    args = ap.parse_args()

    fps = json.load(open(args.fingerprints))
    key = args.style if args.style in fps else None
    if key is None:
        style, pattern = args.style.split('/', 1)
        diff = args.diff or 'mid'
        key = f'{style}/{pattern}/{diff}'
    assert key in fps, f'指纹库没有 {key}，现有: {sorted(fps)}'
    fp = fps[key]

    txt = open(args.skeleton, encoding='utf-8').read()
    body = txt.split('[HitObjects]')[1]
    skel = sorted({int(l.split(',')[2]) for l in body.strip().splitlines()
                   if l.strip() and l.split(',')[0].lstrip('-').isdigit()})
    print(f'骨架: {len(skel)} 行 → 指纹 {key}（nps {fp["nps"]} 和弦率 {fp["chord_rate"]} '
          f'手交替 {fp["hand_alt"]} LN {fp["ln_ratio"]*100:.0f}%）')

    out, n_rows, n_ln = paint(skel, fp, seed=args.seed, target_ln=args.ln_cap,
                             structure=(json.load(open(args.structure))
                                        if args.structure and os.path.exists(args.structure) else None),
                             dps=args.dps)
    head = f'''osu file format v14

[General]
AudioFilename: {args.audio_name}
Mode: 3

[Metadata]
Title: {args.title}
Artist: {args.artist}
Creator: {args.creator}
Version: {args.version}
Tags: ai mug-prototype fingerprint {key}

[Difficulty]
CircleSize: 4
OverallDifficulty: 8

[TimingPoints]
{OFFSET},{BEAT},4,2,1,20,1,0

[HitObjects]
'''
    open(args.o, 'w', encoding='utf-8').write(head + '\n'.join(out) + '\n')
    ln = sum(1 for l in out if ',128,' in l)
    print(f'写出: {args.o}  行 {n_rows}  音符 {len(out)}  LN {ln} ({ln/len(out)*100:.1f}%)')


if __name__ == '__main__':
    main()
