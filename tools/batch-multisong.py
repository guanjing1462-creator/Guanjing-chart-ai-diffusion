#!/usr/bin/env python3
"""批量：6 首歌 × 六档，全部用同一套（已在参考曲上标定好的）参数。"""
import glob, json, os, subprocess, sys

TIERS = [
 ('Rice',    'rice/technical/mid',  'rice',   'speed',   0.1, 'out/packs/corpus-speed.json',   0.5, None, 18, 0.35, 0.92, 1.15, 0.0,  0.70,'quota', False),
 ('Stream',  'rice/stream/high',    'stream', 'speed',   0,   'out/packs/corpus-speed.json',   0.5, None, 18, 0.35, 0.92, 1.15, 0.0,  0.70,'quota', False),
 ('Tech',    'rice/technical/high', 'tech',   'tech',    1,   'out/packs/corpus-tech.json',    0.5, None, 20, 0.55, 0.74, 1.25, 0.0,  0.70,'quota', False),
 ('Jack',    'rice/technical/high', 'tech',   'jack',    1,   'out/packs/corpus-jack.json',    0.5, None, 35, 0.30, 0.30, 1.45, 0.0,  0.70,'prop',  False),
 ('LN',      'ln/technical/high',   'ln',     'tech',    1.2, 'out/packs/corpus-tech.json',    0.5, None, 25, 0.55, 0.82, 1.20, 0.0,  0.70,'quota', True),
 ('Stamina', 'rice/technical/high', 'tech',   'stamina', 1,   'out/packs/corpus-stamina.json', 0.5, None, 12, 1.00, 0.92, 2.00, 0.30, 0.0, 'prop',  False),
]
SONGS = {
 'Frums_SuperMix': 'ref/multisong/Frums_SuperMix.wav',
 'Heart_Fraction': 'ref/multisong/Heart Fraction.ogg',
 'Plum_FallenEra': 'ref/multisong/Plum-FallenEra.mp3',
 'REDALiCE_x_t_pazolite___Xterfusion': 'ref/multisong/REDALiCE x t+pazolite - Xterfusion.mp3',
 '_______________Flash_Me_Back': 'ref/multisong/かめりあ - Flash Me Back.mp3',
 'seatrus___purifikation': 'ref/multisong/seatrus - purifikation.ogg',
}
out_dir = 'out/multisong/v9'
os.makedirs(out_dir, exist_ok=True)
for song, audio in SONGS.items():
    an = f'out/multisong/{song}-fx.json'
    if not os.path.exists(an):
        print('跳过（无分析）', song); continue
    for (name, style, cls, ptype, boost, corpus, sp, _u, tpct, cprob, fprob, sboost,
         djp, cluster, npmode, ln) in TIERS:
        if False:
            continue
        dest = f'{out_dir}/{song}--{name}.osu'
        ck = json.load(open(corpus))['types'][ptype]
        jt = round(ck['jackRate'], 3)
        cmd = [sys.executable, 'tools/paint-gj.py', '--analysis', an, '--audio', audio,
               '--lexicon', 'out/rhythm-lexicon16.json', '--style', style,
               '--style-class', cls, '--ptype', ptype, '--pattern-corpus', corpus,
               '--fill-spacing', str(sp), '--fill-thr-pct', str(tpct),
               '--fill-chord-p', str(cprob), '--fill-prob', str(fprob),
               '--size-boost', str(sboost), '--jack-target', str(jt),
               '--disjoint-p', str(djp), '--chord-cluster-p', str(cluster),
               '--raw-table', '1', '--accent-cover', '1', '--fix-playable', '1',
               '--npr-mode', npmode, '--strict', '--dps', '12',
               '--chord-boost', str(boost), '--song-id', 'X', '--target-ett', '18',
               '--creator', 'GJ', '--version', f'GJ multi {name}', '--seed', '22',
               '-o', dest]
        if name == 'Jack':
            cmd += ['--section-swing', '0.80']
        if ln:
            cmd += ['--ln-cap', '0.18']
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode == 0
        print(f'{song[:22]:22s} {name:8s} {"OK" if ok else "FAIL"}', flush=True)
        if not ok:
            print(r.stderr[-300:], flush=True)
