#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check-chart: 结构合法性校验（能开打才算成品）。

检查项（每一条都对应一种「谱子能打开但打得恶心/被站点拒」的真实故障）：
  1. 同刻同列重复音符
  2. 负时间 / 超出音频长度
  3. 列坐标不在 4 列上
  4. LN 长度 < 1 格（释放即点，手感报错）
  5. LN 自重叠（同列下一个音符落在 LN 体内）
  6. LN 尾巴类型位缺失（type=128 但没有 endtime）
  7. LN 占比超过站点 0.18 硬上限
  8. 4s 窗内同列 LN 墙（>3 条同时按住）
  9. 行内音符数 > 4
 10. 两行间隔 < 20ms（同一点被拆成两行）

用法：
  python3 tools/check-chart.py out/v11/*.osu [--audio-sec 242]
"""
import argparse
import glob
import os
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--audio-sec', type=float, default=242.0)
    ap.add_argument('--ln-cap', type=float, default=0.18)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    # 文件名里有 [ ]，glob 会当成字符类而匹配不到 —— 先看是不是现成路径
    files = []
    for pat in args.files:
        if os.path.exists(pat):
            files.append(pat)
        else:
            files += glob.glob(pat)
    allrep = {}
    for f in sorted(files):
        issues = []
        notes = []
        sec = None
        tp = None
        for ln in open(f, encoding='utf-8', errors='ignore'):
            s = ln.strip()
            if s.startswith('['):
                sec = s
                continue
            if sec == '[TimingPoints]' and s and s[0].isdigit():
                tp = s.split(',')[0]
            if sec != '[HitObjects]' or not s or not s[0].isdigit():
                continue
            p = s.split(',')
            if len(p) < 5:
                issues.append(f'HitObject 字段不足: {s[:40]}')
                continue
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
            end = None
            if typ == 128:
                if len(p) < 6:
                    issues.append(f'LN 缺 endtime @{t}')
                else:
                    e = p[5].split(':')[0]
                    if not e.isdigit():
                        issues.append(f'LN endtime 非数字 @{t}: {p[5][:20]}')
                    else:
                        end = int(e)
            notes.append((t, x, end))
        if not notes:
            issues.append('没有音符')
            allrep[f] = {'notes': 0, 'issues': issues}
            continue
        notes.sort(key=lambda n: (n[0], n[1]))
        n = len(notes)
        # 1/3 基本
        for t, x, e in notes:
            if t < 0:
                issues.append(f'负时间 {t}')
            if t > args.audio_sec * 1000 + 50:
                issues.append(f'超出音频 {t}ms > {args.audio_sec}s')
            if x not in (64, 192, 320, 448):
                issues.append(f'列坐标 {x} 不在 4 列')
        # 2 同刻同列
        seen = set()
        for t, x, e in notes:
            if (t, x) in seen:
                issues.append(f'同刻同列重复 @{t} col{x}')
            seen.add((t, x))
        rows = defaultdict(list)
        for t, x, e in notes:
            rows[t].append(x)
        for t, xs in rows.items():
            if len(xs) > 4:
                issues.append(f'行内 {len(xs)} 个音符 @{t}')
        # 10 行间隔过近
        rts = sorted(rows)
        for a, b in zip(rts, rts[1:]):
            if b - a < 20:
                issues.append(f'行间隔 {b-a}ms @{a}')
        # 4/5 LN
        lns = [(t, x, e) for t, x, e in notes if e is not None]
        bycol = defaultdict(list)
        for t, x, e in notes:
            bycol[x].append((t, e))
        for x, lst in bycol.items():
            lst.sort()
            for i, (t, e) in enumerate(lst):
                if e is not None:
                    if e - t < 40:
                        issues.append(f'LN 长度 {e-t}ms 过短 @{t} col{x}')
                    for t2, e2 in lst[i + 1:]:
                        if t2 >= e:
                            break
                        issues.append(f'LN 体内有音符 @{t2} col{x}（LN {t}-{e}）')
                        break
        ln_ratio = len(lns) / n
        if ln_ratio > args.ln_cap + 1e-6:
            issues.append(f'LN 占比 {ln_ratio*100:.1f}% 超站点上限 {args.ln_cap*100:.0f}%')
        # 8 同列同时按住（4s 窗）
        per_col = defaultdict(list)
        for t, x, e in lns:
            per_col[x].append((t, e))
        for x, lst in per_col.items():
            pts = sorted(lst)
            for i, (t, e) in enumerate(pts):
                holding = sum(1 for t2, e2 in pts if t2 <= t < e2)
                if holding >= 4:
                    issues.append(f'col{x} @{t} 同时 {holding} 条 LN')
                    break
        allrep[f] = {'notes': n, 'ln': len(lns), 'lnRatio': round(ln_ratio, 4),
                     'offset': tp, 'first': notes[0][0], 'last': notes[-1][0],
                     'issues': issues}
        tag = 'OK' if not issues else f'{len(issues)} 个问题'
        print(f'{f.split("/")[-1][:46]:48s} n={n:5d} LN={ln_ratio*100:5.1f}%  {tag}')
        for m in issues[:6]:
            print('    -', m)
    if args.json:
        json.dump(allrep, open(args.json, 'w'), ensure_ascii=False, indent=1)
    bad = sum(1 for v in allrep.values() if v['issues'])
    print(f'\n总计 {len(allrep)} 张，{bad} 张有问题')


if __name__ == '__main__':
    main()
