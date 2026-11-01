#!/usr/bin/env python3
"""六档 × 三档难度的二维验证（同一首歌）。"""
import json, os, subprocess, sys
TIERS = [
 ('Rice',    'rice/technical/mid',  'rice',   'speed',   0.1, 'out/packs/corpus-speed.json',   18, 0.35, 0.92, 1.15, 0.0,  0.70, 'quota', False),
 ('Stream',  'rice/stream/high',    'stream', 'speed',   0,   'out/packs/corpus-speed.json',   18, 0.35, 0.92, 1.15, 0.0,  0.70, 'quota', False),
 ('Tech',    'rice/technical/high', 'tech',   'tech',    1,   'out/packs/corpus-tech.json',    20, 0.55, 0.74, 1.25, 0.0,  0.70, 'quota', False),
 ('Jack',    'rice/technical/high', 'tech',   'jack',    1,   'out/packs/corpus-jack.json',    35, 0.30, 0.30, 1.45, 0.0,  0.70, 'prop',  False),
 ('LN',      'ln/technical/high',   'ln',     'tech',    1.2, 'out/packs/corpus-tech.json',    25, 0.55, 0.82, 1.20, 0.0,  0.70, 'quota', True),
 ('Stamina', 'rice/technical/high', 'tech',   'stamina', 1,   'out/packs/corpus-stamina.json', 12, 1.00, 0.92, 2.00, 0.30, 0.00, 'prop',  False),
]
SONG='Heart_Fraction'; AUDIO='ref/multisong/Heart Fraction.ogg'
AN=f'out/multisong/{SONG}-fx.json'
os.makedirs('out/matrix2', exist_ok=True)
for ett in (12, 16, 20):
    dps = 0.411*ett + 0.57
    for (name, style, cls, ptype, boost, corpus, tpct, cprob, fprob, sboost,
         djp, cluster, npmode, ln) in TIERS:
        ck = json.load(open(corpus))['types'][ptype]
        cmd = [sys.executable, 'tools/paint-gj.py', '--analysis', AN, '--audio', AUDIO,
               '--lexicon', 'out/rhythm-lexicon16.json', '--style', style, '--style-class', cls,
               '--ptype', ptype, '--pattern-corpus', corpus, '--fill-spacing', '0.5',
               '--fill-thr-pct', str(tpct), '--fill-chord-p', str(cprob), '--fill-prob', '-1',
               '--size-boost', str(sboost), '--jack-target', str(round(ck['jackRate'], 3)),
               '--disjoint-p', str(djp), '--chord-cluster-p', str(cluster), '--raw-table', '1',
               '--accent-cover', '1', '--fix-playable', '1', '--npr-mode', npmode, '--strict',
               '--chord-boost', str(boost),
               '--song-id', 'X', '--target-ett', str(ett), '--creator', 'GJ',
               '--version', f'GJ {name} ETT{ett}', '--seed', '22',
               '-o', f'out/matrix2/{name}-ETT{ett}.osu']
        if name == 'Jack':
            cmd += ['--section-swing', '0.50']
        if ln:
            cmd += ['--ln-cap', '0.18']
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f'{name:8s} ETT{ett} {"OK" if r.returncode==0 else "FAIL"}', flush=True)
