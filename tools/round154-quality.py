#!/usr/bin/env python3
"""round154 质量自证：两套音频条件数据集（4 带 onset / 32 带 mel 通量）的逐谱对齐强度。

只读 npz + manifest，不解码音频（mel/C 都已经在 npz 里），所以很快。

口径（与 build-audio-cond-fixed.py 的 verify_dataset 一致）
------------------------------------------------------------
* **1/32 格口径**：把每张谱所有窗口按 start_cell 铺回该谱时间轴（重叠处取均值），
  X 四键取均值得「音符占用率」序列，条件取「四带均值」(C) 或「32 带求和」(M)，
  在整首轴上算 Pearson r。这就是「逐谱 X–条件相关」。
* **零假设**：把条件序列在谱内**随机循环平移**（不与音符对齐）重算 r，应 ≈0。
* **人采上界 r250 / r1000**：把占用率与条件强度按 250ms / 1000ms 分箱（格中心时刻
  = offset + shift* + (k+0.5)*step），在分箱序列上算 Pearson r。

用法：
  python3 tools/round154-quality.py            # 新数据集 + 旧数据集对照
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 256
N_RAND = 8


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def chart_series(sel, starts, X, cond, agg='mean'):
    """把一张谱的窗口铺回时间轴。返回 (occ[ T ], cnd[ T ], base_cell, n_cells)。"""
    st = starts[sel].astype(np.int64)
    base = int(st.min())
    T = int((st + WIN).max()) - base
    occ = np.zeros(T, np.float64)
    cnd = np.zeros(T, np.float64)
    cnt = np.zeros(T, np.float64)
    for i, s in zip(sel, st):
        a = int(s - base)
        occ[a:a + WIN] += X[i].astype(np.float32).mean(axis=0)
        c = cond[i].astype(np.float32)
        cnd[a:a + WIN] += c.mean(axis=1) if agg == 'mean' else c.sum(axis=1)
        cnt[a:a + WIN] += 1.0
    m = cnt > 0
    return occ[m] / cnt[m], cnd[m] / cnt[m], base


def dist(a):
    a = np.asarray(a, np.float64)
    if not len(a):
        return {}
    return dict(n=int(len(a)), median=round(float(np.median(a)), 4),
                mean=round(float(a.mean()), 4), min=round(float(a.min()), 4),
                p10=round(float(np.percentile(a, 10)), 4),
                p25=round(float(np.percentile(a, 25)), 4),
                p75=round(float(np.percentile(a, 75)), 4),
                max=round(float(a.max()), 4),
                frac_ge_025=round(float((a >= 0.25).mean()), 4),
                frac_ge_040=round(float((a >= 0.40).mean()), 4))


def binned_r(occ, cnd, base_cell, step_ms, q_ms, bin_ms):
    """按固定时间窗分箱后算 r。格中心时刻 = q + (base+k+0.5)*step。"""
    n = len(occ)
    k = np.arange(n) + base_cell
    t = q_ms + (k + 0.5) * step_ms
    b = np.floor(t / bin_ms).astype(np.int64)
    b -= b.min()
    nb = int(b.max()) + 1
    so = np.bincount(b, weights=occ, minlength=nb)
    sc = np.bincount(b, weights=cnd, minlength=nb)
    cn = np.bincount(b, minlength=nb).astype(np.float64)
    ok = cn > 0
    if ok.sum() < 8:
        return None
    return pearson(so[ok] / cn[ok], sc[ok] / cn[ok])


def analyze(npz_path, man_path, tag, rng_seed=7):
    d = np.load(npz_path)
    X = d['X']
    cid = d['chart_id'].astype(str)
    starts = d['start_cell'].astype(np.int64)
    shift = d['shift_ms'].astype(np.float64)
    cond_key = 'C' if 'C' in d else 'M'
    cond = d[cond_key]
    agg = 'mean' if cond_key == 'C' else 'sum'

    man = {}
    if man_path and os.path.exists(man_path):
        for m in json.load(open(man_path, encoding='utf-8')):
            if m.get('verdict') == 'kept':
                man[m['chart_id']] = m

    rng = np.random.default_rng(rng_seed)
    rows = []
    for c in np.unique(cid):
        sel = np.flatnonzero(cid == c)
        occ, cnd, base = chart_series(sel, starts, X, cond, agg)
        if len(occ) < 20:
            continue
        r = pearson(occ, cnd)
        rr = [pearson(occ, np.roll(cnd, int(k)))
              for k in rng.integers(8, max(9, len(cnd) - 8), N_RAND)]
        m = man.get(str(c), {})
        q = float(m.get('offset_ms', 0.0)) + float(shift[sel[0]])
        step = float(m.get('step_ms', 0.0))
        rec = dict(chart_id=str(c), set_id=int(m.get('SetID', -1)) if m else -1,
                   windows=int(len(sel)), cells=int(len(occ)),
                   r=round(r, 4), r_null=round(float(np.mean(rr)), 4),
                   ln_frac=m.get('ln_frac'), bpm=m.get('bpm'),
                   shift_ms=float(shift[sel[0]]))
        if step > 0:
            rec['r250'] = binned_r(occ, cnd, base, step, q, 250.0)
            rec['r1000'] = binned_r(occ, cnd, base, step, q, 1000.0)
            rec['r250'] = None if rec['r250'] is None else round(rec['r250'], 4)
            rec['r1000'] = None if rec['r1000'] is None else round(rec['r1000'], 4)
        rows.append(rec)

    out = dict(tag=tag, npz=os.path.relpath(npz_path, ROOT),
               cond=cond_key, n_windows=int(len(cid)),
               n_charts=int(len(rows)),
               n_songs=int(len(np.unique(d['song']))),
               r=dist([x['r'] for x in rows]),
               r_null=dist([x['r_null'] for x in rows]),
               split_counts={str(int(s)): int((d['split'] == s).sum())
                             for s in np.unique(d['split'])},
               split_songs={str(int(s)): int(len(np.unique(d['song'][d['split'] == s])))
                            for s in np.unique(d['split'])})
    if any(x.get('r250') is not None for x in rows):
        out['r250'] = dist([x['r250'] for x in rows if x.get('r250') is not None])
        out['r1000'] = dist([x['r1000'] for x in rows if x.get('r1000') is not None])
    out['per_chart'] = rows
    return out


def compare(a, b, tag):
    """4 带 (a) vs 32 带 mel 通量 (b) 的配对对照。"""
    ra = {x['chart_id']: x['r'] for x in a['per_chart']}
    rb = {x['chart_id']: x['r'] for x in b['per_chart']}
    common = [c for c in ra if c in rb]
    va = np.array([ra[c] for c in common])
    vb = np.array([rb[c] for c in common])
    d = vb - va
    return dict(tag=tag, n=int(len(common)),
                r4_median=round(float(np.median(va)), 4),
                r32_median=round(float(np.median(vb)), 4),
                delta_median=round(float(np.median(d)), 4),
                frac_mel_better=round(float((d > 0).mean()), 4),
                corr_of_rs=round(pearson(va, vb), 4),
                frac_mel_ge_025=round(float((vb >= 0.25).mean()), 4),
                frac_4band_ge_025=round(float((va >= 0.25).mean()), 4))


def gate(man_path):
    man = json.load(open(man_path, encoding='utf-8'))
    kept = [m for m in man if m.get('verdict') == 'kept']
    rej = [m for m in man if m.get('verdict') != 'kept']
    c4 = [m for m in man if m.get('reject_reason') != 'non_4k']
    reasons = {}
    for m in c4:
        if m.get('verdict') == 'kept':
            continue
        k = m.get('reject_reason') or 'unknown'
        reasons[k] = reasons.get(k, 0) + 1
    sets_kept = sorted({int(m['SetID']) for m in kept})
    sets_seen = sorted({int(m['SetID']) for m in man})
    windows = int(sum(m['windows'] for m in kept))
    return dict(n_charts_seen=len(man), n_charts_4k=len(c4),
                kept_charts=len(kept), rejected_charts=len(c4) - len(kept),
                non_4k_charts=len(man) - len(c4),
                pass_rate=round(len(kept) / max(1, len(c4)), 4),
                reject_reasons=reasons, windows=windows,
                sets_seen=len(sets_seen), sets_kept=len(sets_kept))


def main():
    fixed2 = 'out/diff-data-audio-fixed2/dataset.npz'
    mel2 = 'out/diff-data-mel2/dataset.npz'
    res = {}
    print('=== 新数据集：4 带 onset（fixed2）===', flush=True)
    a = analyze(fixed2, 'out/diff-data-audio-fixed2/manifest.json', 'new-4band')
    print(json.dumps({k: v for k, v in a.items() if k != 'per_chart'},
                     ensure_ascii=False, indent=1), flush=True)
    print('=== 新数据集：32 带 mel 通量（mel2）===', flush=True)
    b = analyze(mel2, 'out/diff-data-mel2/manifest.json', 'new-mel32')
    print(json.dumps({k: v for k, v in b.items() if k != 'per_chart'},
                     ensure_ascii=False, indent=1), flush=True)
    res['new_4band'] = a
    res['new_mel32'] = b
    res['cmp_new'] = compare(a, b, 'new')
    print('=== 4 带 vs 32 带 mel 通量（新数据集）===')
    print(json.dumps(res['cmp_new'], ensure_ascii=False, indent=1), flush=True)

    res['gate'] = gate('out/diff-data-audio-fixed2/manifest.json')
    print('=== 门禁 ===')
    print(json.dumps(res['gate'], ensure_ascii=False, indent=1), flush=True)

    # 旧数据集同口径对照
    for key, f, m, tag in (
            ('old_4band', 'out/diff-data-audio-fixed/dataset.npz',
             'out/diff-data-audio-fixed/manifest.json', 'old-4band'),
            ('old_mel32', 'out/diff-data-mel/dataset.npz',
             'out/diff-data-mel/manifest.json', 'old-mel32')):
        if os.path.exists(f):
            r = analyze(f, m, tag)
            res[key] = r
            print(f'=== 旧数据集 {tag} ===')
            print(json.dumps({k: v for k, v in r.items() if k != 'per_chart'},
                             ensure_ascii=False, indent=1), flush=True)
    if 'old_4band' in res and 'old_mel32' in res:
        res['cmp_old'] = compare(res['old_4band'], res['old_mel32'], 'old')
        print('=== 4 带 vs 32 带（旧数据集）===')
        print(json.dumps(res['cmp_old'], ensure_ascii=False, indent=1), flush=True)
    if 'old_4band' in res:
        res['gate_old'] = gate('out/diff-data-audio-fixed/manifest.json')

    out_dir = 'out/diffusion/round154-data'
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, 'quality.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    # 逐谱表（便于报告引用）
    json.dump(b['per_chart'], open(os.path.join(out_dir, 'per-chart-mel2.json'),
                                   'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\n写出 {out_dir}/quality.json', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
