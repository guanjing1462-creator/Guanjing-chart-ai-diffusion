#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure-onset: 历史口径的「踩音率」量尺（与交接文档 70.4% 基线同口径）。

定义（HANDOFF §P0-B）：音符时间落在最近 onset ±41ms（= 1/16 格 @185BPM 的半格）
内的比例。同时给出别的阈值与真实误差分布，避免单一口径被「网格对齐」骗到。

用法：
  python3 tools/measure-onset.py --osu X.osu --analysis out/final14-s50/analysis.json
"""
import argparse
import bisect
import json
import statistics as st


def peaks_ms_of(an):
    hop = an['fine']['hopMs']
    return sorted(p * hop for p in an['peaks'])


def note_times(path):
    ts = []
    sec = None
    for ln in open(path, encoding='utf-8', errors='ignore'):
        s = ln.strip()
        if s.startswith('['):
            sec = s
            continue
        if sec == '[HitObjects]' and s and s[0].isdigit():
            p = s.split(',')
            if len(p) > 2:
                ts.append(int(p[2]))
    return sorted(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osu', required=True)
    ap.add_argument('--analysis', default='out/final14-s50/analysis.json')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    an = json.load(open(args.analysis))
    pm = peaks_ms_of(an)
    ts = note_times(args.osu)
    errs = []
    for t in ts:
        j = bisect.bisect_left(pm, t)
        c = []
        if j < len(pm):
            c.append(pm[j] - t)
        if j > 0:
            c.append(t - pm[j - 1])
        errs.append(min(c) if c else 999)
    res = {'file': args.osu, 'notes': len(ts), 'onsets': len(pm)}
    for th in (10, 20, 30, 41, 60):
        res[f'rate<{th}ms'] = round(sum(1 for e in errs if e <= th) / max(1, len(errs)), 4)
    res['meanMs'] = round(st.mean(errs), 2) if errs else None
    res['medianMs'] = round(st.median(errs), 2) if errs else None
    res['p90Ms'] = round(sorted(errs)[int(len(errs) * 0.9)], 1) if errs else None
    print(json.dumps(res, ensure_ascii=False))
    if args.json:
        json.dump(res, open(args.json, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
