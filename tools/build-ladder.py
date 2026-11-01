#!/usr/bin/env python3
"""同一首歌的难度阶梯：D14 / D18 / D22 × 六档。"""
import json, os, subprocess, sys
TIERS = [
 ('Jack',    'rice/technical/high', 'tech', 'jack',    1.0, 'corpus-jack',    35, 0.30, 0.30, 1.45, 0.0,  0.70, 'prop',  False, '0.50'),
 ('Tech',    'rice/technical/high', 'tech', 'tech',    1.0, 'corpus-tech',    20, 0.55, 0.74, 1.25, 0.0,  0.70, 'quota', False, '0'),
 ('Speed',   'rice/stream/high',    'stream','speed',  0.0, 'corpus-speed',   18, 0.35, 0.92, 1.15, 0.0,  0.70, 'quota', False, '0'),
 ('Stamina', 'rice/technical/high', 'tech', 'stamina', 1.0, 'corpus-stamina', 12, 1.00, 0.92, 2.00, 0.30, 0.00, 'prop',  False, '0'),
 ('LN',      'ln/technical/high',   'ln',   'tech',    1.2, 'corpus-tech',    25, 0.55, 0.82, 1.20, 0.0,  0.70, 'quota', True,  '0'),
 ('Hybrid',  'rice/technical/high', 'tech', 'hybrid',  1.0, 'corpus-hybrid',  20, 0.45, 0.85, 1.20, 0.0,  0.70, 'quota', False, '0'),
]
SONG = 'Heart_Fraction'
AUDIO = 'ref/multisong/Heart Fraction.ogg'
ETS = [10, 16, 22, 28]
OUT = 'out/ladder'
os.makedirs(OUT, exist_ok=True)
for ett in ETS:
    for (name, style, cls, ptype, boost, corpus, tpct, cprob, fprob, sboost,
         djp, cluster, npmode, ln, swing) in TIERS:
        ck = json.load(open(f'out/packs/{corpus}.json'))['types'][ptype]
        jt = round(ck['jackRate'] * (0.82 if name in ('LN', 'Speed') else (1.10 if name == 'Jack' else 1.0)), 3)
        dest = f'{OUT}/{SONG}--{name}--D{ett}.osu'
        cmd = [sys.executable, 'tools/paint-gj.py', '--analysis', f'out/multisong/{SONG}-fx.json',
               '--audio', AUDIO, '--lexicon', 'out/rhythm-lexicon16.json', '--style', style,
               '--style-class', cls, '--ptype', ptype, '--pattern-corpus', f'out/packs/{corpus}.json',
               '--fill-spacing', '0.35', '--fill-thr-pct', str(tpct),
               '--fill-chord-p', str(cprob), '--fill-prob', str(fprob),
               '--size-boost', str(sboost), '--jack-target', str(jt),
               '--disjoint-p', str(djp), '--chord-cluster-p', str(cluster),
               '--raw-table', '1', '--accent-cover', '1', '--fix-playable', '1',
               '--npr-mode', npmode, '--strict', '--npr-scale', str(round(0.85 + (ett - 10) / 18.0 * 0.30, 3)),
               '--chord-boost', str(boost), '--song-id', 'X', '--target-ett', str(ett),
               '--section-swing', swing, '--shape-prior', '1.0', '--same-hand-target', '0.22',
               '--col-balance', '0.4',
               '--section-motif', str({'Jack': 0.95, 'LN': 0.85, 'Stamina': 0.65}.get(name, 0.7)),
               '--motif-mirror', '0.1',
               '--motif-cluster', '0.5' if name in ('Jack', 'LN') else '0',
               '--creator', 'GJ', '--version', f'GJ {name} D{ett}', '--seed', '22', '-o', dest]
        if ln: cmd += ['--ln-cap', '0.18']
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f'{name:8s} D{ett}  {"OK" if r.returncode==0 else "FAIL"}', flush=True)
