#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cal-v11: v11 着色器的站点闭环锚定（目标 D → (dps, chord-boost)）。

站点 D 对 (行频, 和弦升音) 是二维响应，且非线性：只调 dps 会饱和在
~11.5 行/s。所以这里用**二维割线**：先在 dps 轴上收敛到「差一档」，
再用 chord-boost 补最后 1-1.5 D，全程写进 out/cal-v11.json 供复用。

用法：
  python3 tools/cal-v11.py --style rice/technical/high --target 15.6 --tier tech
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAINT = os.path.join(HERE, 'paint-v11.py')
SITE = os.path.join(ROOT, 'site-analysis')
CAL = os.path.join(ROOT, 'out/cal-v11.json')


def site_d(path):
    r = subprocess.run(['node', 'run-koishi.js', os.path.abspath(path)],
                       capture_output=True, text=True, cwd=SITE, timeout=180)
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
    try:
        d = json.loads(line)
    except Exception:
        return None, None, None
    return d.get('numericDifficulty'), d.get('pattern'), d.get('noteCount'), d.get('hint')


def gen(args, dps, boost, out):
    cmd = [sys.executable, PAINT, '--analysis', args.analysis, '--audio', args.audio,
           '--style', args.style, '--dps', f'{dps:.2f}', '--chord-boost', f'{boost:.2f}',
           '--lexicon', args.lexicon, '--version', f'cal {dps:.2f}/{boost:.2f}',
           '--seed', str(args.seed), '-o', out]
    if args.ln_cap is not None:
        cmd += ['--ln-cap', str(args.ln_cap)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(r.stdout[-800:], r.stderr[-800:])
        return None
    for l in r.stdout.strip().splitlines()[::-1]:
        if l.startswith('写出'):
            return l
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', default='out/final14-s50/analysis.json')
    ap.add_argument('--audio', default='ref/INTERNET YAMERO.ogg')
    ap.add_argument('--style', required=True)
    ap.add_argument('--lexicon', default='out/rhythm-lexicon32.json')
    ap.add_argument('--target', type=float, required=True)
    ap.add_argument('--tier', required=True)
    ap.add_argument('--seed', type=int, default=22)
    ap.add_argument('--ln-cap', type=float, default=None)
    ap.add_argument('-o', required=True)
    args = ap.parse_args()

    table = json.load(open(CAL)) if os.path.exists(CAL) else {}
    seed_row = table.get(args.style, {}).get('points', [])
    # 初值：从已测点里挑最接近目标的
    if seed_row:
        seed_row.sort(key=lambda p: abs(p['D'] - args.target))
        dps, boost = seed_row[0]['dps'], seed_row[0]['boost']
    else:
        dps, boost = args.target * 0.92, 0.0
    pt = os.path.join(ROOT, 'out/_cal_tmp.osu')
    log = []
    prev = None
    for it in range(9):
        info = gen(args, dps, boost, pt)
        if not info:
            print(f'it{it}: 生成失败（dps {dps:.2f} boost {boost:.2f}）')
            dps = max(6.0, dps - 0.4)
            continue
        D, pat, n, hint = site_d(pt)
        if D is None:
            if hint == 'BelowScope':
                # 低于量程 = 太简单 → 加密度，别降
                print(f'it{it}: dps {dps:.2f} boost {boost:.2f} → BelowScope（太简单），加密度',
                      flush=True)
                if dps < 13.0:
                    dps = min(13.0, dps + 0.7)
                else:
                    boost = min(1.0, boost + 0.12)
                continue
            print(f'it{it}: dps {dps:.2f} boost {boost:.2f} → 站点无法定级({hint})，降档')
            dps = max(6.0, dps - 0.6)
            continue
        err = args.target - D
        print(f'it{it}: dps {dps:.2f} boost {boost:.2f} → D {D:.2f} ({pat}, n={n}) 误差 {err:+.2f}',
              flush=True)
        log.append({'dps': round(dps, 2), 'boost': round(boost, 2), 'D': round(D, 2),
                    'pattern': pat, 'notes': n})
        if abs(err) <= 0.18:
            break
        # 一维割线：先动 dps（上限 13.2 后转 boost）
        if prev and abs(dps - prev[0]) > 1e-6:
            slope = (D - prev[1]) / (dps - prev[0])
        else:
            slope = 1.6
        prev = (dps, D)
        if dps < 12.8 and abs(slope) > 0.2:
            dps = max(6.0, min(13.2, dps + err / max(0.7, abs(slope))))
        else:
            boost = max(0.0, min(1.0, boost + err * 0.28))
    if log:
        best = min(log, key=lambda p: abs(p['D'] - args.target))
        # 最终产物用最优参数重生成
        gen(args, best['dps'], best['boost'], args.o)
        table.setdefault(args.style, {'points': []})['points'] += log
        table[args.style]['best'] = dict(best, tier=args.tier, target=args.target)
        json.dump(table, open(CAL, 'w'), ensure_ascii=False, indent=1)
        print(f'最优: dps {best["dps"]} boost {best["boost"]} → D {best["D"]} ({best["pattern"]})')
        print('写出', args.o)


if __name__ == '__main__':
    main()
