#!/usr/bin/env python3
"""连续难度条件的标定：扫一组条件值，实测 Koishi 难度，产出「条件 → 难度」曲线。

用法：
  python3 tools/calc-diff-curve.py --ckpt out/models/diffusion32-cont.pt \
      --conds 0.1,0.3,0.5,0.7,0.9 --num 4 --out out/diffusion/curve.json

每个条件：采样 num 个窗口 → 拼接转谱 → Koishi 测难度。
结果同时打印成表，并写入 JSON 供 tools/target-diffusion.py 使用。
"""
import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def sample_and_score(ckpt, style, cond, num, thr, bpm):
    tmp = tempfile.mkdtemp(prefix=f'cal_{cond}_')
    rc, _, err = run([sys.executable, 'tools/sample-diffusion32c.py', '--ckpt', ckpt,
                      '--style', str(style), '--diff', str(cond), '--num', str(num),
                      '--threads', '1', '--quiet', '--threshold', str(thr), '--outdir', tmp])
    if rc != 0:
        return None, f'sample 失败: {err.strip().splitlines()[-1] if err.strip() else rc}'
    rolls = sorted(glob.glob(os.path.join(tmp, 'roll-*.npy')))
    if not rolls:
        return None, '没有产出钢琴卷'
    osu = os.path.join(tmp, 'chart.osu')
    rc, _, err = run([sys.executable, 'tools/roll-to-osu32.py'] + rolls +
                     ['--concat', '--bpm', str(bpm), '-o', osu])
    if rc != 0:
        return None, f'转谱失败: {err.strip().splitlines()[-1] if err.strip() else rc}'
    rc, out, err = run(['node', 'tools/gj/batch-koishi.mjs', osu])
    if rc != 0:
        return None, f'Koishi 失败: {err.strip().splitlines()[-1] if err.strip() else rc}'
    try:
        d = json.loads(out)[0]
    except Exception:
        return None, 'Koishi 输出不是 JSON'
    keep = os.path.join('out/diffusion', f'cal-{cond}.osu')
    os.makedirs('out/diffusion', exist_ok=True)
    shutil.copy(osu, keep)
    shutil.rmtree(tmp, ignore_errors=True)
    return {'cond': cond, 'diff': d.get('diff'), 'tier': d.get('tier'),
            'pattern': d.get('pattern'), 'saved': keep}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out/models/diffusion32-cont.pt')
    ap.add_argument('--style', type=int, default=1)
    ap.add_argument('--conds', default='0.1,0.3,0.5,0.7,0.9')
    ap.add_argument('--num', type=int, default=4)
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--bpm', type=float, default=200.0)
    ap.add_argument('--repeat', type=int, default=1, help='每个条件重复几次取中位')
    ap.add_argument('--out', default='out/diffusion/curve.json')
    a = ap.parse_args()

    conds = [float(x) for x in a.conds.split(',') if x.strip()]
    rows = []
    for c in conds:
        ds = []
        for k in range(a.repeat):
            r, err = sample_and_score(a.ckpt, a.style, c, a.num, a.thr, a.bpm)
            if err:
                print(f'条件 {c}: {err}', flush=True)
                continue
            ds.append(r)
        if not ds:
            continue
        vals = [d['diff'] for d in ds if d.get('diff') is not None]
        rows.append({'cond': c,
                     'diff_median': round(statistics.median(vals), 2) if vals else None,
                     'diff_all': vals,
                     'below_scope': sum(1 for d in ds if d.get('diff') is None),
                     'pattern': ds[0].get('pattern'),
                     'tier': ds[0].get('tier'),
                     'saved': [d['saved'] for d in ds]})
        print(f'条件 {c:.2f} → 难度 {rows[-1]["diff_median"]} '
              f'（全部 {vals}，低于表 {rows[-1]["below_scope"]} 次，判定 {rows[-1]["pattern"]}）', flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'ckpt': a.ckpt, 'style': a.style, 'num': a.num, 'thr': a.thr,
               'bpm': a.bpm, 'rows': rows}, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'已写入 {a.out}')

    # 单调性检查
    seq = [(r['cond'], r['diff_median']) for r in rows if r['diff_median'] is not None]
    mono = all(b[1] >= a_[1] for a_, b in zip(seq, seq[1:]))
    print(f'共 {len(seq)} 个可用点，单调递增: {mono}')
    if len(seq) >= 2:
        print(f'覆盖难度 {seq[0][1]} – {seq[-1][1]}')


if __name__ == '__main__':
    sys.exit(main())
