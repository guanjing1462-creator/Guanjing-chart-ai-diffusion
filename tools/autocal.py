#!/usr/bin/env python3
"""autocal: 自动把生成参数标定到「包的目标统计」，不再依赖手工调参。

用法: python3 tools/autocal.py --tier Tech --iters 4
思路: 生成 → 测 5 项指标 → 按误差调 2 个旋钮 → 重复，取误差最小的那轮。
这样换歌/换档时不需要人重调，通用性来自闭环而不是常数。
"""
import argparse, json, subprocess, sys, importlib.util, tempfile, os
import numpy as np

spec = importlib.util.spec_from_file_location('ac', 'tools/audit-chart.py')
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

TIERS = {
 'Rice':    dict(style='rice/technical/mid',  cls='rice', ptype='speed',   corpus='out/packs/corpus-speed.json',   boost=0.1, ln=False),
 'Stream':  dict(style='rice/stream/high',    cls='stream', ptype='speed', corpus='out/packs/corpus-speed.json',   boost=0,   ln=False),
 'Tech':    dict(style='rice/technical/high', cls='tech', ptype='tech',    corpus='out/packs/corpus-tech.json',    boost=1,   ln=False),
 'Jack':    dict(style='rice/technical/high', cls='tech', ptype='jack',    corpus='out/packs/corpus-jack.json',    boost=1,   ln=False),
 'LN':      dict(style='ln/technical/high',   cls='ln',   ptype='tech',    corpus='out/packs/corpus-tech.json',    boost=1.2, ln=True),
 'Stamina': dict(style='rice/technical/high', cls='tech', ptype='stamina', corpus='out/packs/corpus-stamina.json', boost=1,   ln=False),
}

def measure(path, ck):
    n, _ = ac.parse_osu(path)
    rows = ac.rows_of(n)
    mv, g32, ss, sj = [], 0, 0, 0
    for (t0, a), (t1, b) in zip(rows, rows[1:]):
        mv += [min(abs(c - q) for q in a) for c in b]
        d = t1 - t0
        g32 += d <= 60
        if len(a) == 1 and len(b) == 1 and d >= 81:
            ss += 1; sj += a[0] == b[0]
    t = max(1, len(rows) - 1)
    return dict(g32=g32 / t, hand=sum(mv) / len(mv),
                jack=sj / max(1, ss), npr=len(n) / len(rows))

def run(tier, fill_prob, size_boost, out, audio, analysis, song_id, dps):
    T = TIERS[tier]
    cmd = [sys.executable, 'tools/paint-gj.py',
           '--analysis', analysis,
           '--audio', audio,
           '--lexicon', 'out/rhythm-lexicon16.json',
           '--style', T['style'], '--style-class', T['cls'], '--ptype', T['ptype'],
           '--pattern-corpus', T['corpus'],
           '--fill-spacing', '0.5', '--fill-thr-pct', '20',
           '--fill-chord-p', '0.55', '--fill-prob', f'{fill_prob:.3f}',
           '--size-boost', f'{size_boost:.3f}',
           '--disjoint-p', '0', '--chord-cluster-p', '0.70',
           '--raw-table', '1', '--accent-cover', '1', '--fix-playable', '1',
           '--npr-mode', 'quota', '--strict', '--dps', str(dps),
           '--chord-boost', str(T['boost']), '--model', 'out/placement3k.pt',
           '--song-id', song_id, '--target-ett', '18',
           '--creator', 'GJ', '--version', f'GJ autocal {tier}', '--seed', '22',
           '-o', out]
    if T['ln']:
        cmd += ['--ln-cap', '0.18']
    subprocess.run(cmd, check=True, capture_output=True)
    return measure(out, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--audio', default='ref/INTERNET YAMERO.ogg')
    ap.add_argument('--analysis', default='out/final14-s50/analysis-fixed.json')
    ap.add_argument('--song-id', default='INTERNET_YAMERO')
    ap.add_argument('--dps', type=float, default=12.6)
    a = ap.parse_args()
    T = TIERS[a.tier]
    corpus = json.load(open(T['corpus']))['types'][T['ptype']]
    sz = corpus['sizes']
    tgt = dict(g32=corpus['gapDist'].get('32', 0), hand=corpus['hand']['mean'],
               jack=corpus['jackRate'],
               npr=sum(int(k) * v for k, v in sz.items()) / sum(sz.values()))
    print('目标:', {k: round(v, 3) for k, v in tgt.items()})

    fp, sb = 0.55, 1.20          # 故意从「没调过」的中间值起步
    best = None
    for it in range(a.iters):
        out = f'/tmp/autocal-{a.tier}-{it}.osu'
        m = run(a.tier, fp, sb, out, a.audio, a.analysis, a.song_id, a.dps)
        err = abs(m['g32'] - tgt['g32']) / max(0.05, tgt['g32']) + \
              abs(m['npr'] - tgt['npr']) / max(0.05, tgt['npr']) + \
              abs(m['hand'] - tgt['hand']) / max(0.05, tgt['hand'])
        print(f'  第{it+1}轮 fill={fp:.2f} boost={sb:.2f} → '
              f"1/32 {m['g32']:.2f}/{tgt['g32']:.2f} hand {m['hand']:.2f}/{tgt['hand']:.2f} "
              f"npr {m['npr']:.2f}/{tgt['npr']:.2f} err={err:.3f}")
        if best is None or err < best[0]:
            best = (err, it, fp, sb, m, out)
        # 闭环调整：1/32 不够就多拆，npr 高了就少和弦
        fp = float(np.clip(fp + 0.9 * (tgt['g32'] - m['g32']), 0.05, 1.0))
        sb = float(np.clip(sb * (tgt['npr'] / max(1e-6, m['npr'])) ** 1.5, 0.4, 3.0))
    e, it, fp, sb, m, out = best
    print(f'最优：第{it+1}轮 fill={fp:.2f} boost={sb:.2f} err={e:.3f}')
    json.dump(dict(tier=a.tier, song=a.song_id, fill_prob=fp, size_boost=sb, metrics=m, target=tgt,
                   out=out), open(f'out/autocal-{a.tier}.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
