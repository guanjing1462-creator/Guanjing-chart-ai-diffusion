#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""难度精确锚定：目标 D 值 → 自动迭代行频 → 站点验证收敛。

用法：
  python3 tools/anchor-difficulty.py --skeleton sk.osu --structure st.json \
      --style rice/technical/mid --target 14.0 --tol 0.35 -o out.osu

算法：D 与行频（dps）单调，但非线性（站点判定器的响应曲线）。
先在预校准表上插值出初值，再用**割线法**迭代：
  dps_{k+1} = dps_k + (target - D_k) / slope_k
其中 slope 由最近两次实测估计（初值取全局斜率 1.075 D/0.1dps）。
每次迭代 = 生成 + 站点分析（~50s），一般 2-3 次收敛到 ±0.35。
"""
import argparse
import json
import subprocess
import sys
import os

PAINTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pattern-painter.py')
SITE = '/tmp/site-analysis'
# 预校准点（INTERNET YAMERO / rice-technical-mid / 185bpm；换歌需重扫）
CAL = [(7.0, None), (7.8, 13.56), (8.6, 14.37), (9.4, 15.28)]
GLOBAL_SLOPE = (15.28 - 13.56) / (9.4 - 7.8)   # ≈1.075 D per 0.1 dps


def run_koishi(path):
    r = subprocess.run(['node', 'run-koishi.js', path],
                       capture_output=True, text=True, cwd=SITE, timeout=90)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None, None
    return d.get('numericDifficulty'), d.get('pattern')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skeleton', required=True)
    ap.add_argument('--structure', default=None)
    ap.add_argument('--style', required=True)
    ap.add_argument('--target', type=float, required=True, help='目标 D 值')
    ap.add_argument('--tol', type=float, default=0.35)
    ap.add_argument('--max-iters', type=int, default=4)
    ap.add_argument('--version', required=True)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('-o', required=True)
    args = ap.parse_args()

    # 初值：预校准表插值
    known = [(dp, d) for dp, d in CAL if d]
    known.sort()
    dps = known[-1][1] and known[-1][0]
    for i in range(len(known) - 1):
        (d0, D0), (d1, D1) = known[i], known[i + 1]
        if D0 <= args.target <= D1:
            dps = d0 + (args.target - D0) / (D1 - D0) * (d1 - d0)
            break
    if dps is None:
        dps = args.target / GLOBAL_SLOPE * 0.86   # 粗估

    slope = GLOBAL_SLOPE * 10   # D per dps
    prev = None
    for it in range(args.max_iters):
        r = subprocess.run([sys.executable, PAINTER,
                            '--skeleton', args.skeleton, '--style', args.style,
                            '--structure', args.structure or '/nonexistent',
                            '--dps', f'{dps:.2f}', '--version', args.version,
                            '--seed', str(args.seed), '-o', args.o],
                           capture_output=True, text=True, timeout=90)
        D, pat = run_koishi(args.o)
        if D is None:
            print(f'iter {it}: 站点无法定级（dps {dps:.2f}），微调重试')
            dps += 0.15
            continue
        err = args.target - D
        print(f'iter {it}: dps {dps:.2f} → D {D:.2f} ({pat})  误差 {err:+.2f}')
        if abs(err) <= args.tol:
            print(f'收敛：{args.o}')
            return
        if prev:
            dd = dps - prev[0]
            if abs(dd) > 1e-6:
                s = (D - prev[1]) / dd
                if abs(s) > 0.1:
                    slope = s
        dps = max(6.0, min(11.0, dps + err / slope))
        prev = (dps, D)
    print(f'{args.max_iters} 次未收敛（最后 D {D:.2f}），产出保留在 {args.o}')


if __name__ == '__main__':
    main()
