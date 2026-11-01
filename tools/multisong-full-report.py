#!/usr/bin/env python3
import glob, json, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('ac', 'tools/audit-chart.py')
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
T = {'Rice':'speed','Stream':'speed','Tech':'tech','Jack':'jack','LN':'tech','Stamina':'stamina'}
TG = {}
for k, v in T.items():
    d = json.load(open(f'out/packs/corpus-{v}.json'))['types'][v]
    sz = d['sizes']
    TG[k] = dict(g32=d['gapDist'].get('32', 0), hand=d['hand']['mean'], jack=d['jackRate'],
                 npr=sum(int(x)*y for x, y in sz.items())/sum(sz.values()))
print('%-6s %7s %7s %7s %7s' % ('档', '1/32', 'hand', 'jack', 'npr'))
for k in TG:
    t = TG[k]
    print('目标 %-4s %6.2f %7.2f %7.3f %7.2f' % (k, t['g32'], t['hand'], t['jack'], t['npr']))
print()
rows = []
for f in sorted(glob.glob('out/multisong/v6/*.osu')):
    base = f.split('/')[-1][:-4]
    song, tier = base.split('--')
    an = json.load(open(f'out/multisong/{song}-fx.json')); step = an['stepMs']
    notes, _ = ac.parse_osu(f); r = ac.rows_of(notes)
    mv = []; g = 0; ss = sj = 0
    for (t0, a), (t1, b) in zip(r, r[1:]):
        mv += [min(abs(c-q) for q in a) for c in b]
        d = t1 - t0
        g += d/step <= 0.6
        if len(a) == 1 and len(b) == 1 and d >= 81:
            ss += 1; sj += a[0] == b[0]
    n = max(1, len(r)-1)
    m = dict(song=song, tier=tier, g32=g/n, hand=sum(mv)/len(mv), jack=sj/max(1, ss),
             npr=len(notes)/len(r), notes=len(notes), bpm=an['bpm'])
    t = TG[tier]
    m['d'] = dict(g32=abs(m['g32']-t['g32']), hand=abs(m['hand']-t['hand']),
                  jack=abs(m['jack']-t['jack']), npr=abs(m['npr']-t['npr']))
    rows.append(m)
for k in TG:
    sub = [x for x in rows if x['tier'] == k]
    if not sub: continue
    print('%-6s n=%d  1/32 %.2f(±%.2f)  hand %.2f(±%.2f)  jack %.3f(±%.3f)  npr %.2f(±%.2f)' % (
        k, len(sub), np.mean([x['g32'] for x in sub]), np.mean([x['d']['g32'] for x in sub]),
        np.mean([x['hand'] for x in sub]), np.mean([x['d']['hand'] for x in sub]),
        np.mean([x['jack'] for x in sub]), np.mean([x['d']['jack'] for x in sub]),
        np.mean([x['npr'] for x in sub]), np.mean([x['d']['npr'] for x in sub])))
allm = np.mean([[x['d']['g32'], x['d']['hand'], x['d']['jack'], x['d']['npr']] for x in rows], axis=0)
print('\n36 张总体平均绝对偏差: 1/32 %.3f  hand %.3f  jack %.3f  npr %.3f' % tuple(allm))
json.dump(dict(targets=TG, rows=rows), open('out/multisong/v6-report.json','w'), indent=1)
