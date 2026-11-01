#!/usr/bin/env python3
"""按目标难度出谱：给定 Koishi 目标难度，自动搜索 (ETT, npr-scale) 并验证。

做法：
  1. 用已测得的「ETT → 难度」曲线做二分，找到最接近目标的 ETT；
  2. 生成谱面，用观景台 Koishi（本地 Node worker）实测难度；
  3. 若偏差 >0.6，按实测斜率修正一次，再生成再测（最多 3 轮）；
  4. 输出最终谱面 + 实测难度 + 判定键型。

用法：
  python3 tools/target-difficulty.py --tier Tech --target 13 --song Heart_Fraction -o out/tgt/Tech-D13.osu
"""
import argparse
import json
import os
import subprocess
import sys

TIERS = {
    'Tech': dict(ptype='tech', corpus='tech', style='rice/technical/high', cls='tech',
                 tpct='20', cprob='0.55', fprob='0.74', sboost='1.25', djp='0',
                 cluster='0.70', npmode='quota', boost='1', swing='0', jack_mul=1.0),
    'Jack': dict(ptype='jack', corpus='jack', style='rice/technical/high', cls='tech',
                 tpct='35', cprob='0.30', fprob='0.30', sboost='1.45', djp='0',
                 cluster='0.70', npmode='prop', boost='1', swing='0.50', jack_mul=0.85),
    'Speed': dict(ptype='speed', corpus='speed', style='rice/stream/high', cls='stream',
                  tpct='18', cprob='0.35', fprob='0.92', sboost='1.15', djp='0',
                  cluster='0.70', npmode='quota', boost='0', swing='0', jack_mul=0.82),
    'Stamina': dict(ptype='stamina', corpus='stamina', style='rice/technical/high', cls='tech',
                    tpct='12', cprob='1.00', fprob='0.92', sboost='2.00', djp='0.30',
                    cluster='0', npmode='prop', boost='1', swing='0', jack_mul=1.0),
}
# 实测的默认曲线（ETT → Koishi 难度），Tech 档位标定，其余档先用同形曲线
ETT_CURVE = [(6, 10.5), (8, 11.7), (10, 13.6), (16, 14.8), (22, 14.7), (28, 14.9)]


def ett_for(target):
    """在标定曲线上反查 ETT（超出范围就取端点）。"""
    pts = sorted(ETT_CURVE)
    if target <= pts[0][1]:
        return pts[0][0]
    if target >= pts[-1][1]:
        return pts[-1][0]
    for (e0, d0), (e1, d1) in zip(pts, pts[1:]):
        if d0 <= target <= d1:
            if d1 == d0:
                return e1
            return round(e0 + (target - d0) * (e1 - e0) / (d1 - d0))
    return pts[-1][0]


def build(tier, song, ett, npr_scale, out):
    cfg = TIERS[tier]
    ck = json.load(open(f"out/packs/corpus-{cfg['corpus']}.json"))['types'][cfg['ptype']]
    jt = round(ck['jackRate'] * cfg['jack_mul'], 3)
    cmd = [sys.executable, 'tools/paint-gj.py',
           '--analysis', f'out/multisong/{song}-fx.json',
           '--audio', f'ref/multisong/{song.replace("_", " ")}.ogg'
           if os.path.exists(f'ref/multisong/{song.replace("_", " ")}.ogg')
           else f'ref/multisong/{song}.ogg',
           '--lexicon', 'out/rhythm-lexicon16.json', '--style', cfg['style'],
           '--style-class', cfg['cls'], '--ptype', cfg['ptype'],
           '--pattern-corpus', f"out/packs/corpus-{cfg['corpus']}.json",
           '--shape-model', f"out/models/shape-{cfg['corpus']}.json",
           '--fill-spacing', '0.35', '--fill-thr-pct', cfg['tpct'],
           '--fill-chord-p', cfg['cprob'], '--fill-prob', cfg['fprob'],
           '--size-boost', cfg['sboost'], '--jack-target', str(jt),
           '--disjoint-p', cfg['djp'], '--chord-cluster-p', cfg['cluster'],
           '--raw-table', '1', '--shape-prior', '1.0', '--same-hand-target', '0.25',
           '--quantize', '32', '--accent-cover', '1', '--fix-playable', '1',
           '--npr-mode', cfg['npmode'], '--strict', '--chord-boost', cfg['boost'],
           '--song-id', song, '--target-ett', str(ett), '--npr-scale', str(npr_scale),
           '--section-swing', cfg['swing'], '--section-motif', '0.7', '--motif-mirror', '0.1',
           '--creator', 'GJ', '--version', f'GJ {tier}', '--seed', '22', '-o', out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or '')


def measure(path):
    r = subprocess.run(['node', 'tools/gj/batch-koishi.mjs', path],
                       capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
        return data[0]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True, choices=list(TIERS))
    ap.add_argument('--target', type=float, required=True)
    ap.add_argument('--song', default='Heart_Fraction')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--max-rounds', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

    # 先按标定曲线估个起点，再在邻域里细扫；高目标再叠 npr 补偿
    e0 = ett_for(a.target)
    cands = [e0] + [e0 + d for d in (1, -1, 2, -2, 3, -3, 4, -4)]
    cands = [c for c in cands if 6 <= c <= 28]
    plan = [(c, 1.0) for c in cands]
    if a.target > 15.0:   # 高难度端密度饱和，靠和弦补（排在最前面优先试）
        plan = [(28, 1.6), (28, 1.3), (28, 1.9), (28, 1.0)] + plan
    plan = plan[:a.max_rounds]
    best = None
    for rnd, (ett, npr) in enumerate(plan, 1):
        ok, err = build(a.tier, a.song, ett, npr, a.out)
        if not ok:
            print(f'第 {rnd} 轮生成失败：{err.strip().splitlines()[-1] if err.strip() else ""}')
            return 1
        m = measure(a.out)
        if not m:
            print('Koishi 测不出')
            return 1
        diff = m['diff']
        print(f'第 {rnd} 轮：ETT {ett} npr ×{npr:.2f} → 难度 {diff} 判定 {m["pattern"]}')
        if diff is None:
            # 低于模型下限 = 比 Alpha Low（约 11）还简单，按 10.5 估算后继续搜索
            diff = 10.5
            print('  （低于模型下限，按 10.5 估算）')
        if best is None or abs(diff - a.target) < abs(best[0] - a.target):
            best = (diff, m['pattern'], ett, npr)
        if abs(diff - a.target) <= 0.6:
            print(f'  ✅ 达标（目标 {a.target}，实测 {diff}，判定 {m["pattern"]}）')
            return 0
    diff, pattern, ett, npr = best
    print(f'离目标最近的组合：ETT {ett} npr ×{npr:.2f} → {diff}（目标 {a.target}，差 {abs(diff - a.target):.2f}）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
