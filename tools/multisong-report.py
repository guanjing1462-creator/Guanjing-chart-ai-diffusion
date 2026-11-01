#!/usr/bin/env python3
"""跨歌验证报告：对每首已生成的谱，输出 BPM/网格/采音/键型四项指标。
1/32 用**格相对**口径（gap <= 0.6 * 1/16 步长），避免慢歌被判 0。
"""
import importlib.util, glob, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location('ac', 'tools/audit-chart.py')
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

corpus = json.load(open('out/packs/corpus-tech.json'))['types']['tech']
sz = corpus['sizes']
TGT = dict(g32=corpus['gapDist'].get('32', 0), hand=corpus['hand']['mean'],
           jack=corpus['jackRate'],
           npr=sum(int(k) * v for k, v in sz.items()) / sum(sz.values()))

print('目标（tech 包）: 1/32 %.2f  hand %.2f  jack %.3f  npr %.2f' %
      (TGT['g32'], TGT['hand'], TGT['jack'], TGT['npr']))
print('%-30s %7s %7s | %7s %7s %7s %7s' %
      ('歌', 'BPM', '1/16ms', '1/32', 'hand', 'jack', 'npr'))
rows = []
for f in sorted(glob.glob('out/multisong/charts/*.osu')):
    n = f.split('/')[-1][:-4]
    an = json.load(open(f'out/multisong/{n}-fx.json'))
    step = an['stepMs']
    notes, _ = ac.parse_osu(f)
    r = ac.rows_of(notes)
    mv = []; g = 0; ss = sj = 0
    for (t0, a), (t1, b) in zip(r, r[1:]):
        mv += [min(abs(c - q) for q in a) for c in b]
        d = t1 - t0
        g += d / step <= 0.6
        if len(a) == 1 and len(b) == 1 and d >= 81:
            ss += 1; sj += a[0] == b[0]
    T = max(1, len(r) - 1)
    row = dict(song=n, bpm=an['bpm'], step=step, g32=g / T, hand=sum(mv) / len(mv),
               jack=sj / max(1, ss), npr=len(notes) / len(r), notes=len(notes))
    rows.append(row)
    print('%-30s %7.1f %7.1f | %7.2f %7.2f %7.3f %7.2f' %
          (n[:30], row['bpm'], row['step'], row['g32'], row['hand'], row['jack'], row['npr']))

import statistics as st
def dev(k): return st.mean(abs(x[k] - TGT[k]) for x in rows)
print('\n平均绝对偏差: 1/32 %.3f   hand %.3f   jack %.3f   npr %.3f' %
      (dev('g32'), dev('hand'), dev('jack'), dev('npr')))
json.dump(dict(targets=TGT, rows=rows), open('out/multisong/report.json', 'w'), indent=1)
