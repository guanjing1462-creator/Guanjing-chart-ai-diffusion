#!/usr/bin/env python3
"""新六档：Jack / Tech / Speed / Stamina / LN / Hybrid，并为每首歌打 .osz。"""
import json, os, subprocess, sys
TIERS = [
 ('Jack',    'rice/technical/high', 'tech', 'jack',    1.0, 'corpus-jack',    35, 0.30, 0.30, 1.45, 0.0,  0.70, 'prop',  False, '0.50'),
 ('Tech',    'rice/technical/high', 'tech', 'tech',    1.0, 'corpus-tech',    20, 0.55, 0.74, 1.25, 0.0,  0.70, 'quota', False, '0'),
 ('Speed',   'rice/stream/high',    'stream','speed',  0.0, 'corpus-speed',   18, 0.35, 0.92, 1.15, 0.0,  0.70, 'quota', False, '0'),
 ('Stamina', 'rice/technical/high', 'tech', 'stamina', 1.0, 'corpus-stamina', 12, 1.00, 0.92, 2.00, 0.30, 0.00, 'prop',  False, '0'),
 ('LN',      'ln/technical/high',   'ln',   'tech',    1.2, 'corpus-tech',    25, 0.55, 0.82, 1.20, 0.0,  0.70, 'quota', True,  '0'),
 ('Hybrid',  'rice/technical/high', 'tech', 'hybrid',  1.0, 'corpus-hybrid',  20, 0.45, 0.85, 1.20, 0.0,  0.70, 'quota', False, '0'),
]
SONGS = {
 'Frums_SuperMix': 'ref/multisong/Frums_SuperMix.wav',
 'Heart_Fraction': 'ref/multisong/Heart Fraction.ogg',
 'Plum_FallenEra': 'ref/multisong/Plum-FallenEra.mp3',
 'REDALiCE_x_t_pazolite___Xterfusion': 'ref/multisong/REDALiCE x t+pazolite - Xterfusion.mp3',
 '_______________Flash_Me_Back': 'ref/multisong/かめりあ - Flash Me Back.mp3',
 'seatrus___purifikation': 'ref/multisong/seatrus - purifikation.ogg',
}
only = sys.argv[1] if len(sys.argv) > 1 else None
out_dir = 'out/tiers-v19'
os.makedirs(out_dir, exist_ok=True)
for song, audio in SONGS.items():
    if only and only != song: continue
    an = f'out/multisong/{song}-fx.json'
    if not os.path.exists(an): print('跳过', song); continue
    for (name, style, cls, ptype, boost, corpus, tpct, cprob, fprob, sboost,
         djp, cluster, npmode, ln, swing) in TIERS:
        ck = json.load(open(f'out/packs/{corpus}.json'))['types'][ptype]
        dest = f'{out_dir}/{song}--{name}.osu'
        cmd = [sys.executable, 'tools/paint-gj.py', '--analysis', an, '--audio', audio,
               '--lexicon', 'out/rhythm-lexicon16.json', '--style', style,
               '--style-class', cls, '--ptype', ptype, '--pattern-corpus', f'out/packs/{corpus}.json',
               '--fill-spacing', '0.35', '--fill-thr-pct', str(tpct),
               '--fill-chord-p', str(cprob), '--fill-prob', str(fprob),
               '--size-boost', str(sboost), '--jack-target', str(round(ck['jackRate'] * (0.68 if name == 'LN' else 0.82 if name == 'Speed' else 0.85 if name == 'Jack' else 1.0), 3)),
               '--disjoint-p', str(djp), '--chord-cluster-p', str(cluster),
               '--raw-table', '1', '--accent-cover', '1', '--fix-playable', '1',
               '--npr-mode', npmode, '--strict', '--dps', '12',
               '--chord-boost', str(boost), '--song-id', 'X', '--target-ett', '18',
               '--section-swing', swing, '--quantize', '32', '--move-boost', '-0.5', '--shape-prior', '1.0', '--same-hand-target', '0.45', '--col-balance', '0.4',
               '--section-motif', str({'Jack': 0.95, 'LN': 0.85, 'Stamina': 0.65}.get(name, 0.7)),
               '--motif-mirror', '0.1',
               '--motif-cluster', '0.5' if name in ('Jack', 'LN') else '0',
               '--creator', 'GJ', '--version', f'GJ {name} D18', '--seed', '22', '-o', dest]
        if ln: cmd += ['--ln-cap', '0.18']
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f'{song[:24]:24s} {name:8s} {"OK" if r.returncode==0 else "FAIL"}', flush=True)
        if r.returncode != 0: print(r.stderr[-300:], flush=True)
