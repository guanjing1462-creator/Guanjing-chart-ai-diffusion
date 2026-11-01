#!/usr/bin/env python3
"""生成评估用谱面（每档 × D10/D28），用于键型分化判别测试。"""
import json
import os
import subprocess
import sys

SONG = 'Heart_Fraction'
AUDIO = 'ref/multisong/Heart Fraction.ogg'
OUT = 'out/eval'
os.makedirs(OUT, exist_ok=True)

TIERS = {
    'Tech': dict(ptype='tech', corpus='tech', style='rice/technical/high', cls='tech',
                 boost='1.0', tpct='20', cprob='0.55', fprob='0.74', sboost='1.25',
                 djp='0', cluster='0.70', npmode='quota', ln=False, swing='0'),
    'Jack': dict(ptype='jack', corpus='jack', style='rice/technical/high', cls='tech',
                 boost='1.0', tpct='35', cprob='0.30', fprob='0.30', sboost='1.45',
                 djp='0', cluster='0.70', npmode='prop', ln=False, swing='0.50'),
    'Speed': dict(ptype='speed', corpus='speed', style='rice/stream/high', cls='stream',
                  boost='0.0', tpct='18', cprob='0.35', fprob='0.92', sboost='1.15',
                  djp='0', cluster='0.70', npmode='quota', ln=False, swing='0'),
    'Stamina': dict(ptype='stamina', corpus='stamina', style='rice/technical/high', cls='tech',
                    boost='1.0', tpct='12', cprob='1.00', fprob='0.92', sboost='2.00',
                    djp='0.30', cluster='0.0', npmode='prop', ln=False, swing='0'),
}
MUL = {'Jack': 0.85, 'Speed': 0.82, 'LN': 0.68}

ok = 0
for ett in (10, 28):
    for tier, cfg in TIERS.items():
        ck = json.load(open(f"out/packs/corpus-{cfg['corpus']}.json"))['types'][cfg['ptype']]
        jt = round(ck['jackRate'] * MUL.get(tier, 1.0), 3)
        dest = f'{OUT}/{SONG}--{tier}--D{ett}.osu'
        cmd = [sys.executable, 'tools/paint-gj.py',
               '--analysis', f'out/multisong/{SONG}-fx.json', '--audio', AUDIO,
               '--lexicon', 'out/rhythm-lexicon16.json',
               '--style', cfg['style'], '--style-class', cfg['cls'],
               '--ptype', cfg['ptype'], '--pattern-corpus', f"out/packs/corpus-{cfg['corpus']}.json",
               '--fill-spacing', '0.35', '--fill-thr-pct', cfg['tpct'],
               '--fill-chord-p', cfg['cprob'], '--fill-prob', cfg['fprob'],
               '--size-boost', cfg['sboost'], '--jack-target', str(jt),
               '--disjoint-p', cfg['djp'], '--chord-cluster-p', cfg['cluster'],
               '--raw-table', '1', '--shape-prior', '1.0', '--same-hand-target', '0.25',
               '--shape-model', f'out/models/shape-{cfg["corpus"]}.json',
               '--accent-cover', '1', '--fix-playable', '1', '--npr-mode', cfg['npmode'],
               '--strict', '--chord-boost', cfg['boost'], '--song-id', SONG,
               '--target-ett', str(ett), '--section-swing', cfg['swing'],
               '--section-motif', '0.7', '--motif-mirror', '0.1',
               '--creator', 'GJ', '--version', 'Eval', '--seed', '22', '-o', dest]
        if cfg['ln']:
            cmd += ['--ln-cap', '0.18']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            ok += 1
            print(f'{tier} D{ett} OK', flush=True)
        else:
            print(f'{tier} D{ett} FAIL: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""}', flush=True)
print(f'完成 {ok}/8')
