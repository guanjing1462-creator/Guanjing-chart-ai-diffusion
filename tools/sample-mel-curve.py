#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分段难度起伏采样（round156 用）：mel 条件 + **逐窗 --diff / --target-occ** + 并行。

与 tools/sample-curve.py 的区别（不改那个文件，另建一份）：
  * 走新采样器 tools/sample-diffusion32alignocc.py 的 **mel 32 带条件**分支：
    每窗自带 --cond-mel + --start-cell（第 i 窗 = start_cell + i*256），所以起伏不会
    把「对音」丢掉，也不会因为整首条件被同一阈值切而拉平密度；
  * 每窗单独一次采样（num=1），于是 **--target-occ 在每窗内独立标定**，密度谷/峰真能分开；
  * 窗口采样用线程池并发（CPU 单线程 torch，多进程吃多核）。

plan.json 结构（数组）：
  [{
    "id": "delta",
    "mel": "out/diffusion/round156-osz/mel-heart.npz",
    "bpm": 136.3,
    "start_cell": 400,
    "style": 0,
    "seed": 7100,
    "windows": [{"diff": 0.44, "occ": 0.215}, ...],
    "outdir": "out/diffusion/round156-osz/delta",
    "osu": "out/diffusion/round156-osz/delta/out.osu",
    "title": "...", "artist": "...", "creator": "GJ", "version": "Delta",
    "audio": "heart.ogg", "ln": "none", "ln_min": 3, "od": 8.0
  }, ...]

用法：
  python3 tools/sample-mel-curve.py --plan out/.../plan.json --jobs 6
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

WIN = 256


def run_chart(ch, jobs, sampler, converter, root):
    os.makedirs(ch['outdir'], exist_ok=True)
    files, metas, errs = [], [], []

    def one(i):
        w = ch['windows'][i]
        d = os.path.join(ch['outdir'], f'w{i:02d}')
        cmd = [sys.executable, sampler,
               '--ckpt', ch['ckpt'],
               '--cond-mel', ch['mel'],
               '--start-cell', str(int(ch['start_cell']) + i * WIN),
               '--num', '1', '--style', str(ch.get('style', 0)),
               '--diff', f"{w['diff']:.3f}",
               '--target-occ', f"{w['occ']:.4f}",
               '--bpm', str(ch['bpm']),
               '--seed', str(int(ch['seed']) + i),
               '--threads', '1', '--quiet', '--outdir', d]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return i, None, (r.stderr or r.stdout).strip()[-400:]
        got = sorted(glob.glob(os.path.join(d, 'roll-*.npy')))
        if len(got) != 1:
            return i, None, f'窗口 {i} 产出 {len(got)} 个 roll'
        return i, got[0], None

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, p, e in ex.map(one, range(len(ch['windows']))):
            if e:
                errs.append(e)
            else:
                files.append((i, p))
    if errs:
        print(f"[{ch['id']}] 采样失败：{errs[0]}", file=sys.stderr)
        return None
    files.sort()
    k = ch['windows'][len(files) // 2]
    print(f"[{ch['id']}] {len(files)} 窗采样完成", flush=True)

    os.makedirs(os.path.dirname(ch['osu']) or '.', exist_ok=True)
    cmd = [sys.executable, converter] + [p for _, p in files] + [
        '--concat', '--bpm', str(ch['bpm']),
        '--title', ch['title'], '--artist', ch['artist'],
        '--creator', ch.get('creator', 'GJ'), '--version', ch['version'],
        '--audio', ch['audio'], '--od', str(ch.get('od', 8.0)),
        '--ln', ch.get('ln', 'none'), '--ln-min', str(ch.get('ln_min', 3)),
        '--json', '-o', ch['osu']]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{ch['id']}] 转谱失败：{(r.stderr or '').strip()[-400:]}", file=sys.stderr)
        return None
    meta = {'id': ch['id'], 'osu': ch['osu'], 'bpm': ch['bpm'], 'mel': ch['mel'],
            'start_cell': ch['start_cell'], 'style': ch.get('style'),
            'window_count': len(files),
            'windows': ch['windows'],
            'mean_diff': round(sum(w['diff'] for w in ch['windows']) / len(ch['windows']), 4),
            'mean_occ': round(sum(w['occ'] for w in ch['windows']) / len(ch['windows']), 4),
            'converter_tail': (r.stdout or '').strip().splitlines()[-6:]}
    json.dump(meta, open(os.path.join(ch['outdir'], 'chart-meta.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"[{ch['id']}] → {ch['osu']}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--jobs', type=int, default=6, help='整张 plan 的并发窗口数')
    ap.add_argument('--sampler', default='tools/sample-diffusion32alignocc.py')
    ap.add_argument('--converter', default='tools/roll-to-osu32.py')
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    done = []
    for ch in plan:
        # 每张谱内部也并发（jobs 就是全局并发上限，这里靠串行 chart + 并行窗口实现）
        m = run_chart(ch, a.jobs, a.sampler, a.converter, '')
        if m is None:
            return 1
        done.append(m)
    summary = os.path.join(os.path.dirname(a.plan), 'sample-summary.json')
    json.dump(done, open(summary, 'w'), ensure_ascii=False, indent=1)
    print(f'全部完成 → {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
