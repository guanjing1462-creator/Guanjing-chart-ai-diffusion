#!/usr/bin/env python3
"""人类写谱「对音」选择性分析（round 156）。

数据
----
* `out/diff-data-mel/dataset.npz`：M[N,256,32] mel onset 通量（1/32 格，逐歌 P99
  归一），X[N,4,256] 4 列占用（含 LN 持续段），start_cell / chart_id 可把窗口拼回
  整曲时间轴。
* `out/diff-data-audio-fixed/manifest.json`：chart_id → SetID / chart / offset_ms /
  step_ms / n_cells / bpm / shift_ms。
* `out/audio-dl/<SetID>/<chart>`：原始 .osu（音符头 / LN 头尾 / 红线），用
  build-audio-cond-fixed.parse_osu 解析，得到「头」而不是「占用」。

口径
----
1. 每张谱把 256 格窗口按 start_cell 拼成整曲 flux（重叠区取均值），只统计实际
   被窗口覆盖的格（valid）。flux 强度用 32 带均值；另算「低频带 0–7 / 高频带
   24–31」用于 Q6。
2. 强度分位一律在**单谱内**算（rankdata / n），再跨谱汇总：同时给「跨谱中位」
   与「池化」。这样高密度谱不会把低密度谱的统计压平。
3. 音符时刻 → 格：c = floor((t - (offset_ms+shift_ms)) / step_ms)，与条件网格同
   一映射。头（heads）与占用（occ）分开：LN 体只算占用、不算头。

用法
----
  python3 tools/analyze-human-align.py            # 全量 114 谱，写 metrics.json
  python3 tools/analyze-human-align.py --limit 8  # 冒烟
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEL_NPZ = os.path.join(ROOT, 'out/diff-data-mel/dataset.npz')
MANIFEST = os.path.join(ROOT, 'out/diff-data-audio-fixed/manifest.json')
DL_DIR = os.path.join(ROOT, 'out/audio-dl')
OUT_DIR = os.path.join(ROOT, 'out/diffusion/round156-human-align')

BANDS = 32
LOW_BANDS = slice(0, 8)      # mel 0–? 低带（约 0–1.5 kHz）
HIGH_BANDS = slice(24, 32)   # mel 高带（约 5–8 kHz）


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BFX = load_module('tools/build-audio-cond-fixed.py', 'bfx')


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.median(xs)) if xs else float('nan')


def pct_edge(xs, q):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.percentile(xs, q)) if xs else float('nan')


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 8 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ 数据装载
def stitch_charts(limit=None):
    d = np.load(MEL_NPZ, allow_pickle=True)
    M, X = d['M'], d['X']
    cid = d['chart_id'].astype(str)
    start = d['start_cell']
    man = {m['chart_id']: m for m in json.load(open(MANIFEST, encoding='utf-8'))
           if m.get('verdict') == 'kept' and m.get('chart_id')}
    charts = []
    for c in np.unique(cid):
        if c not in man:
            continue
        meta = man[c]
        n = int(meta['n_cells'])
        w = np.where(cid == c)[0]
        acc = np.zeros((n, BANDS), np.float64)
        cnt = np.zeros(n, np.float64)
        occ = np.zeros((4, n), bool)
        for i in w:
            s = int(start[i])
            e = min(n, s + M.shape[1])
            if s >= n:
                continue
            acc[s:e] += M[i, :e - s].astype(np.float64)
            cnt[s:e] += 1
            occ[:, s:e] |= X[i, :, :e - s].astype(bool)
        cov = cnt > 0
        if cov.sum() < 64:
            continue
        flux = np.zeros((n, BANDS), np.float64)
        flux[cov] = acc[cov] / cnt[cov, None]
        charts.append(dict(cid=c, meta=meta, n=n, flux=flux, cov=cov, occ=occ))
    if limit:
        charts = charts[:limit]
    return d, charts


def add_notes(ch, want_phase=True):
    """解析 .osu，把音符头映射到格。"""
    m = ch['meta']
    p = os.path.join(DL_DIR, str(m['SetID']), m['chart'])
    info = BFX.parse_osu(p)
    origin = float(m['offset_ms']) + float(m['shift_ms'])
    step = float(m['step_ms'])
    n = ch['n']
    heads = np.zeros(n, np.int32)          # 头数（含 LN 头）
    occ_tap = np.zeros((4, n), bool)
    ln_head = []                           # (cell, col, len_cells)
    notes = []                             # (t, col, end, cell)
    for t, col, end in info['notes']:
        c = int(np.floor((t - origin) / step))
        if c < 0 or c >= n:
            continue
        heads[c] += 1
        occ_tap[col, c] = True
        notes.append((t, col, end, c))
        if end > t + 1e-6:
            ln_head.append((c, col, int(round((end - t) / step))))
    ch['heads'] = heads
    ch['occ_tap'] = occ_tap
    ch['notes'] = notes
    ch['ln_head'] = ln_head
    ch['redlines'] = info['redlines']
    ch['step'] = step
    ch['origin'] = origin
    return ch


def flux_pct(ch):
    """单谱内 flux 强度的分位（0..1），用 32 带均值；无效格 → nan。"""
    v = ch['cov']
    I = ch['flux'][v].mean(axis=1)
    r = np.argsort(np.argsort(I)).astype(np.float64)  # 0..n-1
    p = np.full(ch['n'], np.nan)
    p[v] = (r + 0.5) / len(I)
    ch['I'] = I
    ch['Ipct'] = p
    ch['per_ch'] = p
    return ch


# ------------------------------------------------------------------ 各项统计
def stats_q1(ch):
    """强度分位 × 命中率。"""
    v = ch['cov']
    p = ch['Ipct'][v]
    hit = (ch['heads'][v] > 0)
    bands = [(0.90, 1.01, 'top10'), (0.70, 0.90, 'p70-90'),
             (0.40, 0.70, 'p40-70'), (0.00, 0.40, 'low40')]
    out = {}
    for lo, hi, name in bands:
        m = (p >= lo) & (p < hi)
        out[name] = dict(n=int(m.sum()),
                         hit=float(hit[m].mean()) if m.any() else float('nan'))
    # 强起音（局部极大）是否被跳过
    I = ch['Ipct'].copy()
    n = ch['n']
    peak = np.zeros(n, bool)
    if n > 5:
        for c in range(2, n - 2):
            if not np.isfinite(I[c]) or I[c] < 0.10:
                continue
            w = I[c - 2:c + 3]
            if np.isfinite(w).all() and I[c] >= w.max() - 1e-12:
                peak[c] = True
    pk = peak & v
    if pk.any():
        pp = ch['Ipct'][pk]
        ph = ch['heads'][pk] > 0
        thr = np.percentile(pp, 70)
        out['peak'] = dict(n=int(pk.sum()),
                           hit=float(ph.mean()),
                           hit_strong=float(ph[pp >= thr].mean()),
                           n_strong=int((pp >= thr).sum()))
    return out


def stats_q2(ch):
    v = ch['cov']
    p = ch['Ipct'][v]
    h = ch['heads'][v]
    out = {}
    for k in (1, 2, 3, 4):
        m = h == k
        if m.sum() >= 5:
            out[str(k)] = dict(n=int(m.sum()), pct_med=float(np.median(p[m])))
    # 跨手 / 同手双押：用 4 列占用是否跨左右（0,1 左；2,3 右）
    return out


def stats_q3(ch):
    v = ch['cov']
    p = ch['Ipct']
    heads = ch['heads']
    step = ch['step']
    gap_cells = int(np.ceil(2000.0 / step))
    empty = (heads == 0) & v
    n = ch['n']
    gaps = []
    c = 0
    while c < n:
        if empty[c]:
            c0 = c
            while c < n and empty[c]:
                c += 1
            if c - c0 >= gap_cells:
                gaps.append((c0, c))
        else:
            c += 1
    out = dict(gaps=[], gap_frac=0.0, pre=[], post=[], inside=[])
    if not gaps:
        return out
    out['gap_frac'] = float(sum(b - a for a, b in gaps) / v.sum())
    w = int(round(500.0 / step))
    for a, b in gaps:
        inside = p[a:b]
        inside = inside[np.isfinite(inside)]
        pre = p[max(0, a - w):a]
        post = p[b:min(n, b + w)]
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        out['inside'].append(float(inside.mean()) if len(inside) else None)
        out['pre'].append(float(pre.mean()) if len(pre) else None)
        out['post'].append(float(post.mean()) if len(post) else None)
        out['gaps'].append(dict(cells=b - a, dur_s=round((b - a) * step / 1000, 2),
                                inside=out['inside'][-1], pre=out['pre'][-1],
                                post=out['post'][-1],
                                start_pct=float(p[a - 1]) if a > 0 and np.isfinite(p[a - 1]) else None))
    return out


def stats_q4(ch):
    p = ch['Ipct']
    heads = ch['heads']
    step = ch['step']
    n = ch['n']
    ln = ch['ln_head']
    out = dict(n_ln=len(ln), n_heads=int(heads.sum()))
    if not ln:
        return out
    ln_pct = [p[c] for c, _, _ in ln if np.isfinite(p[c])]
    tap_pct = [p[c] for c in np.where(heads > 0)[0]
               if np.isfinite(p[c]) and not any(c == lc for lc, _, _ in ln)]
    out['ln_head_pct_med'] = med(ln_pct)
    out['tap_head_pct_med'] = med(tap_pct)
    # LN 长度 vs 下一个强 onset（pct>=0.85）的距离
    strong = np.isfinite(p) & (p >= 0.85)
    idx = np.where(strong)[0]
    lens, dists = [], []
    for c, _, L in ln:
        if L < 2:
            continue
        k = np.searchsorted(idx, c + 1)
        if k >= len(idx):
            continue
        lens.append(L)
        dists.append((idx[k] - c) * step / 1000.0)
    out['len_vs_next_strong_r'] = spearman(lens, dists)
    out['n_len'] = len(lens)
    # 对照：LN 长度 vs 到下一个音符头的距离
    nh = np.where(heads > 0)[0]
    lens2, dists2 = [], []
    for c, _, L in ln:
        if L < 2:
            continue
        k = np.searchsorted(nh, c + 1)
        if k >= len(nh):
            continue
        lens2.append(L)
        dists2.append((nh[k] - c) * step / 1000.0)
    out['len_vs_next_note_r'] = spearman(lens2, dists2)
    out['len_med'] = med(lens)
    return out


def stats_q5(ch):
    if len(ch['redlines']) != 1:
        return None
    h = ch['heads']
    idx = np.where(h > 0)[0]
    if len(idx) < 64:
        return None
    ph8 = np.bincount(idx % 8, minlength=8).astype(float)
    ph8 /= ph8.sum()
    ph16 = np.bincount(idx % 16, minlength=16).astype(float)
    ph16 /= ph16.sum()
    return dict(ph8=ph8.tolist(), ph16=ph16.tolist(),
                on_beat=float(ph8[0]), on_eighth=float(ph8[0] + ph8[4]),
                on_sixteenth=float(ph8[0] + ph8[2] + ph8[4] + ph8[6]))


def stats_q6(ch):
    """左右手 × 低/高频带。"""
    occ_tap = ch['occ_tap']
    v = ch['cov']
    low = ch['flux'][:, LOW_BANDS].mean(axis=1)
    high = ch['flux'][:, HIGH_BANDS].mean(axis=1)
    d = low - high
    # 每格：左/右手是否有头
    left = occ_tap[0] | occ_tap[1]
    right = occ_tap[2] | occ_tap[3]
    cells = v & (left | right)
    if cells.sum() < 32:
        return None
    dh = d[cells]
    is_right = (right[cells] & ~left[cells]).astype(float)
    both = (right[cells] & left[cells])
    ok = ~both
    # 只在单手上报「高带→右」的倾向
    P_r_given_high = float(is_right[ok & (dh > 0)].mean()) if (ok & (dh > 0)).any() else None
    P_r_given_low = float(is_right[ok & (dh < 0)].mean()) if (ok & (dh < 0)).any() else None
    r = spearman(dh[ok], is_right[ok])
    # 块级（1 拍 = 8 格）
    blocks = []
    for s in range(0, ch['n'] - 7, 8):
        sl = slice(s, s + 8)
        if not v[sl].all():
            continue
        lc = int((occ_tap[0, sl] | occ_tap[1, sl]).sum())
        rc = int((occ_tap[2, sl] | occ_tap[3, sl]).sum())
        if lc + rc < 2:
            continue
        blocks.append((float(d[sl].mean()), (rc - lc)))
    br = spearman([b[0] for b in blocks], [b[1] for b in blocks]) if len(blocks) >= 16 else None
    return dict(P_r_given_high=P_r_given_high, P_r_given_low=P_r_given_low,
                note_r=r, block_r=br, n_cells=int(cells.sum()),
                n_blocks=len(blocks))


# ------------------------------------------------------------------ 汇总
def summarize(charts):
    res = {}
    q1 = [stats_q1(c) for c in charts]
    res['q1'] = dict(
        bands={k: dict(hit_med=med([x[k]['hit'] for x in q1]),
                       hit_p25=pct_edge([x[k]['hit'] for x in q1], 25),
                       hit_p75=pct_edge([x[k]['hit'] for x in q1], 75),
                       cells_med=med([x[k]['n'] for x in q1]))
               for k in ('top10', 'p70-90', 'p40-70', 'low40')},
        peak=dict(hit_med=med([x['peak']['hit'] for x in q1 if 'peak' in x]),
                  hit_strong_med=med([x['peak']['hit_strong'] for x in q1 if 'peak' in x]),
                  n_charts=sum(1 for x in q1 if 'peak' in x)),
    )

    q2s = [stats_q2(c) for c in charts]
    q2 = {}
    for k in ('1', '2', '3', '4'):
        v = [x[k] for x in q2s if k in x]
        q2[k] = dict(n_med=med([x['n'] for x in v]),
                     pct_med=med([x['pct_med'] for x in v]),
                     n_charts=len(v))
    res['q2'] = q2

    q3 = [stats_q3(c) for c in charts]
    res['q3'] = dict(
        gap_frac_med=med([x['gap_frac'] for x in q3]),
        inside_med=med([i for x in q3 for i in x['inside']]),
        pre_med=med([i for x in q3 for i in x['pre']]),
        post_med=med([i for x in q3 for i in x['post']]),
        n_gaps=sum(len(x['gaps']) for x in q3),
        charts_with_gap=sum(1 for x in q3 if x['gaps']),
        mean_gap_s=med([g['dur_s'] for x in q3 for g in x['gaps']]),
    )

    q4 = [stats_q4(c) for c in charts]
    q4 = [x for x in q4 if x.get('n_ln', 0) >= 8]
    res['q4'] = dict(
        charts=len(q4),
        ln_frac=med([x['n_ln'] / max(1, x['n_heads']) for x in q4]),
        ln_head_pct_med=med([x['ln_head_pct_med'] for x in q4]),
        tap_head_pct_med=med([x['tap_head_pct_med'] for x in q4]),
        len_vs_next_strong_r=med([x['len_vs_next_strong_r'] for x in q4]),
        len_vs_next_note_r=med([x['len_vs_next_note_r'] for x in q4]),
        len_med_cells=med([x['len_med'] for x in q4]),
    )

    q5 = [stats_q5(c) for c in charts]
    q5 = [x for x in q5 if x]
    if q5:
        arr8 = np.array([x['ph8'] for x in q5])
        arr16 = np.array([x['ph16'] for x in q5])
        res['q5'] = dict(charts=len(q5),
                         ph8_med=arr8.mean(axis=0).tolist(),
                         ph8_p25=np.percentile(arr8, 25, axis=0).tolist(),
                         ph8_p75=np.percentile(arr8, 75, axis=0).tolist(),
                         ph16_med=arr16.mean(axis=0).tolist(),
                         on_beat=med([x['on_beat'] for x in q5]),
                         on_eighth=med([x['on_eighth'] for x in q5]),
                         on_sixteenth=med([x['on_sixteenth'] for x in q5]))
    q6 = [stats_q6(c) for c in charts]
    q6 = [x for x in q6 if x]
    res['q6'] = dict(
        charts=len(q6),
        P_r_given_high=med([x['P_r_given_high'] for x in q6]),
        P_r_given_low=med([x['P_r_given_low'] for x in q6]),
        note_r_med=med([x['note_r'] for x in q6]),
        note_r_frac_pos=float(np.mean([x['note_r'] > 0 for x in q6])) if q6 else float('nan'),
        block_r_med=med([x['block_r'] for x in q6]),
        block_r_frac_pos=float(np.mean([x['block_r'] > 0 for x in q6 if x['block_r'] is not None])),
    )
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'metrics.json'))
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    d, charts = stitch_charts()
    print(f'[data] {len(charts)} 谱拼接完成', flush=True)
    charts = charts[:a.limit] if a.limit else charts
    for ch in charts:
        add_notes(ch)
        flux_pct(ch)
    print('[data] 音符头 / 分位就绪', flush=True)

    # 完整性：占用（npz）与 osu 头的一致性 + 基线 r
    rs, mism = [], []
    for ch in charts:
        occ_npz = ch['occ'].any(axis=0).astype(float)
        v = ch['cov']
        rs.append(BFX.pearson(ch['flux'][v].mean(axis=1), occ_npz[v]))
        ho = (ch['heads'] > 0).astype(float)
        mism.append(float(np.mean((occ_npz > 0) != (ho > 0))))
    baseline = dict(r_flux_occ_med=med(rs), head_vs_occ_mismatch_med=med(mism),
                    n_charts=len(charts))

    res = summarize(charts)
    res['baseline'] = baseline
    res['n_charts'] = len(charts)
    json.dump(res, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])
    print(f'[out] {a.out}')


if __name__ == '__main__':
    main()
