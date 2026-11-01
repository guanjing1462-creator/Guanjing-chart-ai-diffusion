#!/usr/bin/env python3
"""节奏词汇统计：把相邻行间隔换算成「1/16 格数」，统计每个档位的间隔分布。

用途：对比人类谱与生成谱的节奏词汇差异（混沌度高的根因）。
"""
import argparse, glob, json, collections
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location('ac', 'tools/audit-chart.py')
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

# 允许的简单比例（1/16 单位）：1=1/16, 2=1/8, 3=附点, 4=1/4, 6, 8=1/2, 12, 16=全音符
SIMPLE = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def vocab_of(path, step_ms=None):
    n, _ = ac.parse_osu(path)
    rows = ac.rows_of(n)
    if len(rows) < 40:
        return None
    gaps = np.diff([t for t, _ in rows]).astype(float)
    if step_ms is None:
        step_ms = float(np.median(gaps))
    if step_ms <= 0:
        return None
    units = gaps / step_ms
    units = units[(units > 0.3) & (units < 40)]
    if len(units) < 20:
        return None
    # 吸到最近的简单比例，统计残差（判断这首谱是不是严格贴网格）
    snapped = np.array([min(SIMPLE, key=lambda s: abs(u - s)) for u in units])
    resid = np.abs(units - snapped)
    hist = collections.Counter(snapped.tolist())
    total = sum(hist.values())
    return dict(hist={k: v / total for k, v in sorted(hist.items())},
                resid_med=float(np.median(resid)),
                resid_p90=float(np.percentile(resid, 90)),
                n=len(units))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packs', nargs='*', default=[])
    ap.add_argument('--charts', nargs='*', default=[])
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    out = {}
    groups = []
    for pk in a.packs:
        groups.append((pk, sorted(glob.glob(f'out/packs/{pk}/*.osu'))[:40], None))
    for c in a.charts:
        song = c.split('/')[-1].split('--')[0]
        step = None
        try:
            step = json.load(open(f'out/multisong/{song}-fx.json'))['stepMs']
        except Exception:
            pass
        groups.append((c.split('/')[-1], [c], step))
    for name, files, step in groups:
        agg = collections.Counter()
        resids = []
        for f in files:
            v = vocab_of(f, step)
            if not v:
                continue
            for k, vv in v['hist'].items():
                agg[k] += vv * v['n']
            resids.append(v['resid_med'])
        tot = sum(agg.values()) or 1
        dist = {k: round(v / tot, 3) for k, v in sorted(agg.items())}
        out[name] = dict(dist=dist, resid=round(float(np.mean(resids)), 3) if resids else None,
                         n=len(files))
        if not a.json:
            top = sorted(dist.items(), key=lambda x: -x[1])[:6]
            print('%-26s 残差 %.3f  词汇: %s' % (
                name[:26], out[name]['resid'] or 0,
                ' '.join(f'{k:g}×:{v:.2f}' for k, v in top)))
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
