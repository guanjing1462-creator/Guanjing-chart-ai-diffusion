#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pattern-ablate: 在同一张谱上只改一种「难度结构」，看站点 D 怎么响应。

背景：v11 的谱在同样行频下比人类低 1.2-1.5 D。站点 D 到底奖励什么结构？
与其猜，不如在一张已知 D 的谱上做单变量替换：

  baseline  原样
  jack      把一部分单音行改成「同列连打」（上一行同列，1/16 间隔）
  anchor    连续和弦行共用一列（锚和弦）
  triple    2 押升 3 押（同时随机删一个单音保持音符数不变）
  trill     长流改成两列来回三连音
  handstream 连续和弦行换成跨手大跳（0/3 交替）

用法：
  python3 tools/pattern-ablate.py --osu X.osu --mode jack -o Y.osu --rate 0.5
"""
import argparse
import random


def parse(path):
    head, objs, sec = [], [], None
    for ln in open(path, encoding='utf-8', errors='ignore'):
        s = ln.rstrip('\n')
        if s.startswith('['):
            sec = s
            head.append(s)
            continue
        if sec == '[HitObjects]' and s and s[0].isdigit():
            objs.append(s.split(','))
        else:
            head.append(s)
    return head, objs


def group(objs):
    rows = {}
    for p in objs:
        rows.setdefault(int(p[2]), []).append(p)
    return [(t, rows[t]) for t in sorted(rows)]


def col_of(p):
    return int(p[0]) // 128


def set_col(p, c):
    p[0] = str(int((c + 0.5) * 128))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osu', required=True)
    ap.add_argument('--mode', required=True,
                    choices=['baseline', 'jack', 'anchor', 'triple', 'trill', 'handstream'])
    ap.add_argument('--rate', type=float, default=0.6, help='替换比例')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('-o', required=True)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    head, objs = parse(args.osu)
    rows = group(objs)
    removed = set()

    if args.mode == 'jack':
        prev_cols = None
        prev_t = None
        for t, ps in rows:
            if len(ps) == 1 and prev_cols and len(prev_cols) == 1 and prev_t is not None \
                    and t - prev_t <= 100 and rnd.random() < args.rate:
                set_col(ps[0], col_of(prev_cols[0]))
            prev_cols, prev_t = ps, t
    elif args.mode == 'anchor':
        anchor = None
        for t, ps in rows:
            if len(ps) >= 2:
                if anchor is not None and rnd.random() < args.rate:
                    # 把这一行里最靠后的那一列换成锚列
                    set_col(ps[-1], anchor)
                else:
                    anchor = col_of(ps[-1])
            else:
                anchor = None
    elif args.mode == 'triple':
        pool = [p for t, ps in rows if len(ps) == 1 for p in ps]
        rnd.shuffle(pool)
        k = 0
        for t, ps in rows:
            if len(ps) == 2 and rnd.random() < args.rate and k < len(pool):
                used = {col_of(p) for p in ps}
                free = [c for c in range(4) if c not in used]
                if not free:
                    continue
                add = rnd.choice(free)
                newp = list(ps[0])
                newp[0] = str(int((add + 0.5) * 128))
                newp[1] = '192'
                ps.append(newp)
                removed.add(id(pool[k]))
                k += 1
    elif args.mode == 'trill':
        run = []
        for t, ps in rows:
            if len(ps) == 1:
                run.append(ps[0])
            else:
                run = []
                continue
            if len(run) >= 3 and rnd.random() < args.rate:
                a = col_of(run[-3])
                b = 3 - a
                set_col(run[-2], b)
                set_col(run[-1], a)
    elif args.mode == 'handstream':
        for t, ps in rows:
            if len(ps) >= 2 and rnd.random() < args.rate:
                if rnd.random() < 0.5:
                    set_col(ps[0], 0)
                    set_col(ps[-1], 3)
                else:
                    set_col(ps[0], 3)
                    set_col(ps[-1], 0)

    # triple 模式：删掉等量的随机单音，保持音符数不变
    if removed:
        out = []
        for p in objs:
            if id(p) in removed:
                continue
            out.append(p)
        objs = out
    objs.sort(key=lambda p: (int(p[2]), int(p[0])))
    open(args.o, 'w', encoding='utf-8').write(
        '\n'.join(head + [','.join(p) for p in objs]) + '\n')
    print(f'{args.mode}: 写出 {args.o}  ({len(objs)} 音符)')


if __name__ == '__main__':
    main()
