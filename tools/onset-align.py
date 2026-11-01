#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""onset-align: 把任意谱（含扩散骨架）整行吸附到真实音频 onset 上。

扩散模型的时间分辨率受 VAE 潜空间限制（512 帧 = 371ms/帧），细到 46ms 的
落点本来就学不出来——它骨架对音差是**架构性的**，多训几个 epoch 解决不了。
工程上正确的分工是：扩散负责「结构 + 键型」，规则层负责「对音」。

本工具做后者，并按**行**（同刻多列 = 和弦）整体吸，不会把和弦拆散：
  1. 每行时间吸到 ±threshold 内最近的 onset（默认 60ms）
  2. 保序：吸完必须比上一行晚 >=24ms，否则顺延（不越过、不重叠）
  3. 同刻同列去重
  4. 打印吸附前后的对音统计

用法：
  python3 tools/onset-align.py --osu gen.osu --analysis out/final14-s50/analysis.json \
      --threshold 60 -o gen-aligned.osu
"""
import argparse
import bisect
import json
import statistics as st


def parse(path):
    head, objs, sec = [], [], None
    for ln in open(path, encoding='utf-8', errors='ignore'):
        s = ln.rstrip('\n')
        if s.startswith('['):
            sec = s
            head.append(s)
            continue
        if sec == '[HitObjects]' and s and s[0].isdigit():
            p = s.split(',')
            if len(p) >= 5:
                objs.append(p)
        else:
            head.append(s)
    return head, objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osu', required=True)
    ap.add_argument('--analysis', default='out/final14-s50/analysis.json')
    ap.add_argument('--threshold', type=float, default=60.0)
    ap.add_argument('-o', required=True)
    args = ap.parse_args()
    an = json.load(open(args.analysis))
    hop = an['fine']['hopMs']
    peaks = sorted(p * hop for p in an['peaks'])
    head, objs = parse(args.osu)

    groups = {}
    for p in objs:
        groups.setdefault(int(p[2]), []).append(p)

    new, errs0, errs1 = [], [], []
    moved = 0
    last_t = -1
    for t0 in sorted(groups):
        j = bisect.bisect_left(peaks, t0)
        cand = []
        if j < len(peaks):
            cand.append(peaks[j])
        if j > 0:
            cand.append(peaks[j - 1])
        pk = min(cand, key=lambda x: abs(x - t0)) if cand else t0
        dt = abs(pk - t0)
        errs0.append(dt)
        nt = t0
        if dt <= args.threshold and dt > 1 and pk > last_t + 24:
            nt = int(round(pk))
            moved += 1
        if nt <= last_t + 24:
            nt = last_t + 24
        last_t = nt
        errs1.append(abs(nt - pk))
        for p in groups[t0]:
            q = list(p)
            if q[3] == '128' and len(q) > 5:
                tail = q[5].split(':')
                dur = int(tail[0]) - t0 if tail[0].isdigit() else 0
                tail[0] = str(nt + max(80, dur))
                q[5] = ':'.join(tail)
            q[2] = str(nt)
            new.append(q)

    # 吸附后修 LN 覆盖：逐列维护「正在按住的 LN」，若后面的音符落进 LN 体内，
    # 就把 LN 尾收到音符前 30ms（至少 40ms 长）；收不动就丢掉那个音符。
    active = {}          # col -> (start, end, obj_index)
    by_t = {}
    for idx, qq in enumerate(new):
        c = int(qq[0])
        t = int(qq[2])
        end = None
        if qq[3] == '128' and len(qq) > 5:
            e = qq[5].split(':')[0]
            if e.isdigit():
                end = int(e)
        by_t.setdefault(t, []).append((idx, c, end))
    dropped = set()
    for t in sorted(by_t):
        for idx, c, end in by_t[t]:
            if c in active:
                a_start, a_end, a_idx = active[c]
                if a_start < t < a_end:
                    new_end = max(a_start + 40, t - 30)
                    if new_end <= t:
                        dropped.add(idx)
                        continue
                    tail = new[a_idx][5].split(':')
                    tail[0] = str(new_end)
                    new[a_idx][5] = ':'.join(tail)
                    if new_end <= t:
                        active.pop(c, None)
            if end is not None and end > t:
                active[c] = (t, end, idx)
    new = [q for i2, q in enumerate(new) if i2 not in dropped]

    seen, out = set(), []
    for q in new:
        key = (q[2], q[0])
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    out.sort(key=lambda q: (int(q[2]), int(q[0])))
    open(args.o, 'w', encoding='utf-8').write(
        '\n'.join(head + [','.join(q) for q in out]) + '\n')
    print(f'行 {len(groups)}  音符 {len(objs)} → {len(out)}  吸附 {moved} 行  '
          f'平均误差 {st.mean(errs0):.1f}ms → {st.mean(errs1):.1f}ms')
    print('写出', args.o)


if __name__ == '__main__':
    main()
