#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build-pattern-corpus: 从全部人类 4K 谱里学「键型」的统计模型。

按 Etterna 技能值把谱面分成 7 类（jack / chordjack / stream / jumpstream /
handstream / stamina / technical），然后对每一类统计：

  trans[type][prev_shape][gap_class] -> Counter(next_shape)
      shape = 该行按下的列（tuple），gap_class = 与上一行的时间间隔档位
  shape_freq[type][gap_class] -> Counter(shape)     无历史时的边际分布
  size_freq[type] -> Counter(row_size)
  hand_move[type] -> 平均手位移（列差绝对值）
  jack_rate[type] -> 单音同列连打比例

出谱时只要「按类型采样下一行的键型」，键型就天然是人类的、不会卡手
（人类谱里不会出现人类自己打不出来的形状）。

用法：
  python3 tools/build-pattern-corpus.py -o out/pattern-corpus.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location('ac', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit-chart.py'))
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

SKILLS = ['jackspeed_ett', 'chordjack_ett', 'stream_ett', 'jumpstream_ett',
          'handstream_ett', 'stamina_ett', 'technical_ett']
TYPE_NAME = {'jackspeed_ett': 'jack', 'chordjack_ett': 'chordjack',
             'stream_ett': 'stream', 'jumpstream_ett': 'jumpstream',
             'handstream_ett': 'handstream', 'stamina_ett': 'stamina',
             'technical_ett': 'technical'}


def gap_class(ms, step=None):
    """按**格**分档（gap/1/16 步长），而不是绝对毫秒。

    为什么必须这样：慢歌的 1/16 间隔是 125ms，绝对阈值会把它判成『8』档，
    而人谱在『8』档的转移暗示更大的跳跃 → 慢歌手位移系统性偏高 +0.15。
    """
    if step and step > 0:
        g = ms / step
        if g <= 0.62: return '32'
        if g <= 1.45: return '16'
        if g <= 2.45: return '8'
        if g <= 4.10: return '4'
        if g <= 8.10: return '2'
        if g <= 14.5: return '1'
        return 'slow'
    if ms <= 60:
        return '32'      # 1/32 级
    if ms <= 110:
        return '16'
    if ms <= 190:
        return '8'
    if ms <= 330:
        return '4'
    if ms <= 650:
        return '2'
    if ms <= 1200:
        return '1'
    return 'slow'


def type_weights(d, temp=1.6):
    """每首谱对 7 个类型的「像不像」权重（softmax(z/temp)）。
    这样 36598 首谱都参与每一类的统计，只是按专长加权——比硬分类用得多。"""
    vals = [(k, float(d[k])) for k in SKILLS if d.get(k) is not None]
    if len(vals) < 4:
        return {}
    arr = np.array([v for _, v in vals])
    z = (arr - arr.mean()) / (arr.std() + 1e-6)
    # 硬分类：只归到最强的那一类（软混合会把各类统计抹平，实测 jack 率都趋同到 0.03）
    i = int(np.argmax(z))
    if z[i] < 0.15:
        return {}
    return {TYPE_NAME[vals[i][0]]: 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='out/pattern-corpus.json')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dir', default=None,
                    help='只处理这个目录下的谱，并按目录名当类型（分类包独立训练用）')
    args = ap.parse_args()

    feats = {}
    if not args.dir:
        for ln in open('out/feature-db.jsonl', encoding='utf-8'):
            d = json.loads(ln)
            feats[d['id']] = d

    if args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, '*.osu')))
        forced_type = os.path.basename(os.path.normpath(args.dir))
    else:
        files = sorted(glob.glob('out/charts/*.osu'))
        forced_type = None
    if args.limit:
        files = files[:args.limit]
    print(f'谱面 {len(files)} 首')

    trans = defaultdict(Counter)          # (type, prev_shape, gap) -> Counter(next)
    marg = defaultdict(Counter)           # (type, gap) -> Counter(shape)
    sizes = defaultdict(Counter)          # type -> Counter(size)
    hand = defaultdict(list)              # type -> [|Δcol|]
    gaps = defaultdict(Counter)           # type -> Counter(gap bucket)
    jacks = defaultdict(lambda: [0, 0])   # type -> [jack, total]
    n_by_type = Counter()
    done = 0
    for p in files:
        cid = os.path.splitext(os.path.basename(p))[0]
        d = feats.get(cid) if not forced_type else {}
        if not forced_type and not d:
            continue
        tw = {forced_type: 1.0} if forced_type else type_weights(d)
        if not tw:
            continue
        try:
            notes, _ = ac.parse_osu(p)
        except Exception:
            continue
        if len(notes) < 100:
            continue
        rows = ac.rows_of(notes)
        if len(rows) < 50:
            continue
        # 这张谱自己的 1/16 步长（用行间隔的中位数估计，避免依赖 BPM 元数据）
        _g = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
        _g = sorted(x for x in _g if 20 < x < 400)
        _step = float(_g[len(_g) // 2]) if _g else None
        prev = None
        for i, (tm, cols) in enumerate(rows):
            sh = tuple(cols)
            g = gap_class(tm - prev[0]) if prev else 'start'
            psh = tuple(prev[1]) if prev else None
            mv = [min(abs(c - q) for q in psh) for c in sh] if psh else []
            isj = (len(sh) == 1 and psh is not None and len(psh) == 1)
            jv = 1 if (isj and sh[0] == psh[0]) else 0
            for t, w in tw.items():
                n_by_type[t] += w
                marg[(t, g)][sh] += w
                gaps[t][g] += w
                sizes[t][len(sh)] += w
                if psh is not None:
                    trans[(t, psh, g)][sh] += w
                    for x in mv:
                        hand[t].append((x, w))
                    # 只有 >=81ms（1/16 及以上）的同列连打才算「可弹的 jack」；
                    # 40ms 同列是单指 25 次/秒，物理上弹不了，计入会把指标带偏
                    if isj and g != '32':
                        jacks[t][1] += w
                        jacks[t][0] += w * jv
            prev = (tm, cols)
        done += 1
        if done % 5000 == 0:
            print(f'  已处理 {done}')

    out = {'types': {}, 'n': dict(n_by_type)}
    for t in n_by_type:
        dt = {
            'n': n_by_type[t],
            'sizes': {str(k): v for k, v in sizes[t].items()},
            'hand': (lambda a, w: {
                'mean': float((a * w).sum() / max(1e-9, w.sum())),
                'p50': float(np.percentile(np.repeat(a, np.maximum(1, (w * 20).astype(int))), 50)),
                'p90': float(np.percentile(np.repeat(a, np.maximum(1, (w * 20).astype(int))), 90)),
            } if len(a) else {'mean': 0.0, 'p50': 0.0, 'p90': 0.0})(
                np.array([x for x, _ in hand[t]]), np.array([w for _, w in hand[t]])),
            'jackRate': (jacks[t][0] / max(1, jacks[t][1])),
            'gapDist': {k: round(v / max(1, sum(gaps[t].values())), 5)
                        for k, v in gaps[t].items()},
            'marg': {}, 'trans': {},
        }
        for (tt, g), cnt in marg.items():
            if tt != t:
                continue
            tot = sum(cnt.values())
            dt['marg'][g] = {str(list(k)): round(v / tot, 5) for k, v in cnt.items()}
        for (tt, psh, g), cnt in trans.items():
            if tt != t:
                continue
            tot = sum(cnt.values())
            key = f'{list(psh)}|{g}'
            # 按行大小分组各保留一批：单音在上下文里是少数，直接 most_common(20)
            # 会把单音整个挤掉 → 采样器找不到候选 → 退回边际分布（不再看上一行）
            # 4 键一共只有 15 种形状，**全部保留**。
            # 之前每个上下文只留前 12 个 → 语料的列分布被截断带偏
            # （第 0 列 0.271 vs 实际谱面 0.242），生成出来的谱也跟着偏。
            dt['trans'][key] = {str(list(k)): round(v / tot, 5) for k, v in cnt.items()}
        out['types'][t] = dt
        print(f'  {t:11s} 加权谱数 {n_by_type[t]:8.0f}  手位移均值 {dt["hand"]["mean"]:.2f}  '
              f'jack {dt["jackRate"]:.3f}  转移上下文 {len(dt["trans"])}')

    json.dump(out, open(args.out, 'w'), ensure_ascii=False)
    print('写出', args.out, os.path.getsize(args.out) // 1024, 'KB')


if __name__ == '__main__':
    main()
