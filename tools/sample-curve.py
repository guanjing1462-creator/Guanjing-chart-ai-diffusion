#!/usr/bin/env python3
"""曲线驱动采样：让扩散谱面有「难度起伏」（人类 4 秒分箱 NPS 的 CV≈0.30、峰均比≈1.40、峰位≈0.60）。

做法：把整首曲子切成 N 个窗口，每个窗口用**不同的连续难度条件**采样，
条件按一条形状曲线变化（默认：起手 0.75 → 峰值 1.0（在约 60% 处）→ 收尾 0.80），
再拼接成完整谱面。

用法：
  python3 tools/sample-curve.py --ckpt <权重> --style 3 --num 12 --base-cond 0.55 \
      --amp 0.35 --peak-pos 0.6 --bpm 136.3 --target-occ 0.32 -o out/curve.osu
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def shape_value(i, n, peak_pos, warm=0.75, tail=0.80):
    """起伏形状：起手 warm → 峰 1.0（在 peak_pos） → 收尾 tail。"""
    x = i / max(1, n - 1)
    if x <= peak_pos:
        t = x / max(1e-6, peak_pos)
        return warm + (1.0 - warm) * t
    t = (x - peak_pos) / max(1e-6, 1.0 - peak_pos)
    return 1.0 - (1.0 - tail) * t


def curve_value(i, n, base, amp, peak_pos, warm=0.75, tail=0.80):
    """第 i 个窗口的连续难度条件。"""
    shape = shape_value(i, n, peak_pos, warm, tail)
    return float(np.clip(base + amp * (shape - 0.5) * 2.0, 0.02, 0.98))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sampler', default='tools/sample-diffusion32occ.py')
    ap.add_argument('--converter', default='tools/roll-to-osu32.py')
    ap.add_argument('--style', type=int, default=3)
    ap.add_argument('--num', type=int, default=12, help='窗口数（每窗约 14s）')
    ap.add_argument('--base-cond', type=float, default=0.55)
    ap.add_argument('--amp', type=float, default=0.35, help='起伏幅度（条件域的峰谷差=2*amp）')
    ap.add_argument('--peak-pos', type=float, default=0.6)
    ap.add_argument('--warm', type=float, default=0.75, help='起手相对强度（1.0=与峰值同密度）')
    ap.add_argument('--tail', type=float, default=0.80, help='收尾相对强度')
    ap.add_argument('--target-occ', type=float, default=0.0, help='0=用固定阈值；否则按该风格密度对齐')
    ap.add_argument('--threshold', type=float, default=0.2)
    ap.add_argument('--bpm', type=float, default=136.3)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--report', default='', help='可选：把每个窗口的条件与统计写成 JSON')
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix='curve_')
    conds, rolls, occs = [], [], []
    for i in range(a.num):
        c = curve_value(i, a.num, a.base_cond, a.amp, a.peak_pos)
        conds.append(round(c, 3))
        occs.append(round(float(np.clip(a.target_occ * shape_value(i, a.num, a.peak_pos, a.warm, a.tail), 0.02, 0.85)), 4) if a.target_occ > 0 else None)
        d = os.path.join(tmp, f'w{i:02d}')
        cmd = [sys.executable, a.sampler, '--ckpt', a.ckpt, '--style', str(a.style),
               '--diff', f'{c:.3f}', '--num', '1', '--seed', str(a.seed + i),
               '--threads', str(a.threads), '--quiet', '--outdir', d]
        if a.target_occ > 0:
            # 关键：目标占位率也要跟着曲线缩放，否则窗口间密度被强制拉平，起伏消失
            sh = shape_value(i, a.num, a.peak_pos, a.warm, a.tail)
            occ_i = float(np.clip(a.target_occ * sh, 0.02, 0.85))
            cmd += ['--target-occ', f'{occ_i:.4f}']
        else:
            cmd += ['--threshold', str(a.threshold)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'窗口 {i} 采样失败：{(r.stderr or "").strip().splitlines()[-1:] }')
            return 1
        got = sorted(glob.glob(os.path.join(d, 'roll-*.npy')))
        if not got:
            print(f'窗口 {i} 没有产出')
            return 1
        rolls += got

    cmd = [sys.executable, a.converter] + rolls + ['--concat', '--bpm', str(a.bpm), '-o', a.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('转谱失败：', (r.stderr or '').strip()[-300:])
        return 1
    print(f'已生成 {a.out}')
    print(f'窗口条件序列：{conds}')
    if a.target_occ > 0:
        print(f'窗口目标占位率：{occs}')

    if a.report:
        json.dump({'conds': conds, 'occs': occs, 'num': a.num, 'base': a.base_cond, 'amp': a.amp,
                   'peak_pos': a.peak_pos, 'target_occ': a.target_occ,
                   'bpm': a.bpm, 'out': a.out}, open(a.report, 'w'), ensure_ascii=False, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
