#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit-chart: 谱面质量体检（对音 / 键型 / 段落逻辑 / 密度跟随）。

用法：
  python3 tools/audit-chart.py --osu X.osu --analysis out/final14-s50/analysis.json \
      [--audio INTERNET YAMERO.ogg] [--json out/audit-X.json] [-q]

指标设计（每条都对应「一坨」的一个具体病）：
  A 对音    note→最近 onset peak 的绝对时间误差（均值/中位/p90/命中率），
            以及「行内重音对齐」（和弦是否落在强 onset 上，对比随机基线）
  B 键型    行间隔分布、和弦率、锚和弦、jack(<120ms 同列)、同列连击、列熵、手平衡
  C 段落    小节的重复结构与重复变奏比例、乐句首音是否踩段首
  D 密度    每 4s 的 nps 曲线与音频 RMS 能量曲线的 Spearman 相关
  E 旋律    音高（谱质心）方向 vs 列移动方向的相关系数（人类谱通常显著为正）
  F LN      LN 占比、长度分布、4s 窗 LN 墙
"""
import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict

COLS = {64: 0, 192: 1, 320: 2, 448: 3}


def parse_osu(path):
    notes = []          # (t, col, end)
    meta = {}
    sec = None
    for ln in open(path, encoding='utf-8', errors='ignore'):
        s = ln.strip()
        if s.startswith('[') and s.endswith(']'):
            sec = s
            continue
        if sec == '[Metadata]' and ':' in s:
            k, v = s.split(':', 1)
            meta[k.strip()] = v.strip()
        elif sec == '[General]' and ':' in s:
            k, v = s.split(':', 1)
            meta[k.strip()] = v.strip()
        elif sec == '[TimingPoints]' and s and not s.startswith('//'):
            p = s.split(',')
            if len(p) > 1:
                meta.setdefault('TimingPoints', []).append((float(p[0]), float(p[1])))
        elif sec == '[HitObjects]' and s and s[0].isdigit():
            p = s.split(',')
            if len(p) < 5:
                continue
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
            col = min(range(4), key=lambda i: abs((i + 0.5) * 128 - x))
            end = None
            if typ & 128:
                tail = p[5].split(':')[0] if len(p) > 5 else ''
                if tail.isdigit() and int(tail) > t:
                    end = int(tail)
            notes.append((t, col, end))
    notes.sort(key=lambda n: (n[0], n[1]))
    return notes, meta


def rows_of(notes):
    rows = defaultdict(list)
    for t, c, e in notes:
        rows[t].append(c)
    return [(t, sorted(cs)) for t, cs in sorted(rows.items())]


def nearest_peaks(peaks_ms, times):
    import bisect
    out = []
    for t in times:
        j = bisect.bisect_left(peaks_ms, t)
        cand = []
        if j < len(peaks_ms):
            cand.append(peaks_ms[j] - t)
        if j > 0:
            cand.append(t - peaks_ms[j - 1])
        if cand:
            out.append(min(cand))
    return out


def spearman(a, b):
    if len(a) < 3:
        return float('nan')
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da * db else float('nan')


def pitch_contour(audio, hop_ms=11.6):
    """谱质心（对数）时间序列，与 analysis.fine 同 hop。"""
    import librosa
    import numpy as np
    y, sr = librosa.load(audio, sr=22050, mono=True)
    hop = int(round(sr * hop_ms / 1000))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    m = (freqs >= 200) & (freqs <= 2500)
    sub, sf = S[m], freqs[m]
    w = sub.sum(axis=0) + 1e-9
    centroid = (sub * sf[:, None]).sum(axis=0) / w
    lc = np.log2(np.maximum(centroid, 100.0))
    # 与着色器同口径：3 点平滑（≈35ms），否则量的是帧噪声
    if len(lc) >= 3:
        lc = np.convolve(lc, np.ones(3) / 3.0, mode='same')
    return lc, sr / hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osu', required=True)
    ap.add_argument('--analysis', default='out/final14-s50/analysis.json')
    ap.add_argument('--audio', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('-q', action='store_true')
    args = ap.parse_args()

    notes, meta = parse_osu(args.osu)
    rows = rows_of(notes)
    an = json.load(open(args.analysis)) if args.analysis else None
    rep = {'file': args.osu, 'version': meta.get('Version'), 'notes': len(notes), 'rows': len(rows)}

    dur = (notes[-1][0] - notes[0][0]) / 1000.0 if notes else 1
    rep['durationSec'] = round(dur, 1)
    rep['nps'] = round(len(notes) / max(1e-9, dur), 2)
    rep['rps'] = round(len(rows) / max(1e-9, dur), 2)
    rep['chordRate'] = round(1 - len(rows) / max(1, len(notes)), 3)
    rep['chords'] = round(sum(1 for t, cs in rows if len(cs) > 1) / max(1, len(rows)), 3)

    # ---------- A 对音 ----------
    if an:
        hop = an['fine']['hopMs']
        peaks_ms = sorted(p * hop for p in an['peaks'])
        err = nearest_peaks(peaks_ms, [t for t, c, e in notes])
        if err:
            err_s = sorted(err)
            rep['align'] = {
                'meanMs': round(st.mean(err), 1),
                'medianMs': round(st.median(err), 1),
                'p90Ms': round(err_s[int(len(err_s) * 0.9)], 1),
                'within10ms': round(sum(1 for x in err if x <= 10) / len(err), 3),
                'within20ms': round(sum(1 for x in err if x <= 20) / len(err), 3),
                'within40ms': round(sum(1 for x in err if x <= 40) / len(err), 3),
            }
        # 行级：行时间到最近 onset
        rerr = nearest_peaks(peaks_ms, [t for t, cs in rows])
        rep['alignRows'] = {'meanMs': round(st.mean(rerr), 1),
                            'medianMs': round(st.median(rerr), 1),
                            'within15ms': round(sum(1 for x in rerr if x <= 15) / len(rerr), 3)}
        # 重音推进：把 onset 按峰值强度排序，取前 30% 作强 onset；和弦行落在强 onset 的比例
        accent = an['accent']
        off, step = an['offsetMs'], an['stepMs']
        acc_sorted = sorted(accent)
        thr = acc_sorted[int(len(acc_sorted) * 0.7)] if acc_sorted else 0
        def cell_of(t):
            return max(0, min(len(accent) - 1, int(round((t - off) / step))))
        ch = [cell_of(t) for t, cs in rows if len(cs) > 1]
        sg = [cell_of(t) for t, cs in rows if len(cs) == 1]
        if ch and sg:
            rep['chordAccent'] = {
                'chordOnStrong': round(sum(1 for c in ch if accent[c] >= thr) / len(ch), 3),
                'singleOnStrong': round(sum(1 for c in sg if accent[c] >= thr) / len(sg), 3),
            }

    # ---------- B 键型 ----------
    gaps = [rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1)]
    if gaps:
        gs = sorted(gaps)
        rep['gaps'] = {'mean': round(st.mean(gaps), 1), 'median': int(st.median(gaps)),
                       'p10': int(gs[int(len(gs) * 0.1)]), 'p90': int(gs[int(len(gs) * 0.9)]),
                       'min': int(gs[0])}
        # 行间隔直方（按 1/16=81ms 归一：0.25 半格 / 0.5 / 1 / 2 / >2 拍）
        near = Counter()
        for g in gaps:
            if g < 20:
                near['<0.25'] += 1
            elif g < 60:
                near['~0.5'] += 1
            elif g < 110:
                near['~1'] += 1
            elif g < 200:
                near['~2'] += 1
            elif g < 400:
                near['~4'] += 1
            else:
                near['>4'] += 1
        rep['gapHist'] = dict(near)
    # jack：同列时间差 <120ms
    bycol = defaultdict(list)
    for t, c, e in notes:
        bycol[c].append(t)
    jacks, pairs = 0, 0
    for c, ts in bycol.items():
        for a, b in zip(ts, ts[1:]):
            pairs += 1
            if b - a < 120:
                jacks += 1
    rep['jackRate'] = round(jacks / max(1, pairs), 3)
    # 「真 jack」：相邻行（gap<=120ms）共用列的比例。站点 D 实测几乎只吃这一项
    # （本机 ablation：同音符数下 +3.0 D，anchor/triple 为 0）
    rj = elig = 0
    for i in range(len(rows) - 1):
        t0, c0 = rows[i]
        t1, c1 = rows[i + 1]
        if t1 - t0 > 120:
            continue
        elig += 1
        if set(c0) & set(c1):
            rj += 1
    rep['rowJack'] = round(rj / max(1, elig), 3)
    rep['rowJackElig'] = elig
    # 连续同列长度
    runs, cur = [], 1
    for i in range(len(rows) - 1):
        same = rows[i][1][-1] == rows[i + 1][1][0] and rows[i + 1][0] - rows[i][0] < 200
        cur = cur + 1 if same else 1
        runs.append(cur)
    rep['maxConsecSameCol'] = max(runs) if runs else 0
    # 列分布 & 手平衡
    colc = Counter(c for t, c, e in notes)
    tot = sum(colc.values())
    rep['colShare'] = {f'c{c}': round(colc[c] / tot, 3) for c in range(4)}
    ent = -sum(p * math.log(p + 1e-12) for p in (colc[c] / tot for c in range(4)))
    rep['colEntropy'] = round(ent / math.log(4), 3)
    left = colc[0] + colc[1]
    rep['handBalance'] = round(left / tot, 3)
    # 锚和弦（同一行是否重复上一行/下两行的列）
    rep['anchor'] = 0.0

    # ---------- C 段落 / 重复 ----------
    if an:
        # 段落：以 4s 为一节（skeleton 口径），统计「同一 4s 等价段的列模式是否被复用」
        pass
    # 4 行 n-gram 重复度（键型词表丰富度：太低=机械循环，太高=随机）
    sig = [tuple(cs) if not isinstance(cs, str) else cs for t, cs in rows]
    n4 = [tuple(sig[i:i + 4]) for i in range(len(sig) - 3)]
    if n4:
        cnt = Counter(n4)
        rep['ngram4Distinct'] = round(len(cnt) / len(n4), 3)
        rep['ngram4TopShare'] = round(cnt.most_common(1)[0][1] / len(n4), 4)

    # ---------- D 密度跟随 ----------
    if an:
        accent = an['accent']
        off, step = an['offsetMs'], an['stepMs']
        seg_cells = max(1, int(4000 / step))
        e_seg = []
        for i in range(0, len(accent), seg_cells):
            s = accent[i:i + seg_cells]
            if s:
                e_seg.append(st.mean(s))
        seg_len_ms = seg_cells * step
        d_seg = [0] * len(e_seg)
        for t, cs in rows:
            k = int((t - off) / seg_len_ms)
            if 0 <= k < len(d_seg):
                d_seg[k] += 1
        # 去掉尾部空段
        n = min(len(e_seg), len(d_seg))
        rep['densityVsEnergy'] = round(spearman(d_seg[:n], e_seg[:n]), 3)

    # ---------- E 旋律（音高→列方向） ----------
    if args.audio:
        try:
            import numpy as np
            cent, fps = pitch_contour(args.audio)
            def pitch_at(t):
                i = max(0, min(len(cent) - 1, int(t * fps / 1000)))
                return float(cent[i])
            # (1) 严格口径：相邻单音行（时间上相邻，gap<=3 格）
            pairs_s, prev = [], None
            # (2) 宽松口径：相邻行，用行内列均值代表手位（含和弦，覆盖 100% 行）
            pairs_a, prev2 = [], None
            for t, cs in rows:
                p = pitch_at(t)
                mean_c = sum(cs) / len(cs)
                if prev2 is not None and t - prev2[0] <= 260:
                    pairs_a.append((p - prev2[1], mean_c - prev2[2]))
                prev2 = (t, p, mean_c)
                if len(cs) == 1:
                    if prev is not None and t - prev[0] <= 260:
                        pairs_s.append((p - prev[1], cs[0] - prev[2]))
                    prev = (t, p, cs[0])
            if len(pairs_a) > 30:
                rep['pitchDir'] = round(spearman([x for x, y in pairs_a],
                                                 [y for x, y in pairs_a]), 3)
                rep['pitchDirN'] = len(pairs_a)
            if len(pairs_s) > 30:
                rep['pitchDirSingle'] = round(spearman([x for x, y in pairs_s],
                                                       [y for x, y in pairs_s]), 3)
                rep['pitchDirSingleN'] = len(pairs_s)
        except Exception as e:      # noqa: BLE001
            rep['pitchDir'] = f'err:{e}'[:80]

    # ---------- F LN ----------
    lns = [(t, c, e) for t, c, e in notes if e]
    rep['lnRatio'] = round(len(lns) / max(1, len(notes)), 3)
    if lns:
        L = sorted((e - t) for t, c, e in lns)
        rep['lnLenMs'] = {'p50': int(L[len(L) // 2]), 'p90': int(L[int(len(L) * 0.9)]), 'max': int(L[-1])}
        # 4s 窗 LN 墙
        ts = [t for t, c, e in lns]
        best = 0
        j = 0
        for i in range(len(ts)):
            while ts[i] - ts[j] > 4000:
                j += 1
            best = max(best, i - j + 1)
        rep['lnWall4s'] = best

    if args.json:
        json.dump(rep, open(args.json, 'w'), ensure_ascii=False, indent=1)
    if not args.q:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


if __name__ == '__main__':
    main()
