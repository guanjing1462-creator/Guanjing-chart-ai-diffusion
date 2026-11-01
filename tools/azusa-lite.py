#!/usr/bin/env python3
"""Azusa-lite v2：严格按开源实现（azusaEstimator.js）逐音符复刻。

每条音符算 dtGlobal/dtSame/dtHand/dtAny、窗密度 d250/d500，
五技能各 4 个衰减窗累加 → power-mean 融合 → 分位数混合 + 结构加成。
未复刻它的最终标定表，因此是相对难度，不是 RC 段位。
"""
import argparse, glob, json, os
import numpy as np

DECAY_MS = [140.0, 280.0, 560.0, 980.0]
DECAY_W = [0.34, 0.30, 0.22, 0.14]
LOCAL_POWER = 2.15
SKILL_KEYS = ['speed', 'stamina', 'chord', 'tech', 'jack']
SKILL_W = dict(speed=0.36, stamina=0.24, chord=0.12, tech=0.16, jack=0.12)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_div(a, b, fb=1.0):
    return a / b if b else fb


def taps_of(path):
    xs = {64: 0, 192: 1, 320: 2, 448: 3}
    by_t = {}
    for ln in open(path, encoding='utf-8', errors='ignore'):
        p = ln.split(',')
        if len(p) < 5:
            continue
        try:
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
        except ValueError:
            continue
        if x not in xs:
            continue
        if typ & 1 or typ & 128:
            by_t.setdefault(t, set()).add(xs[x])
    taps = []
    for t in sorted(by_t):
        cs = sorted(by_t[t])
        for c in cs:
            taps.append((t, c, len(cs)))
    return taps


def curve(path):
    taps = taps_of(path)
    if len(taps) < 80:
        return None
    last_col_t = [-1e9] * 4
    last_hand_t = [-1e9, -1e9]
    prev_any1 = prev_any2 = -1e9
    prev_col = 0
    prev_t = taps[0][0]
    c250 = c500 = 0
    series = {k: np.zeros(len(taps)) for k in SKILL_KEYS}
    dens250 = np.zeros(len(taps)); dens500 = np.zeros(len(taps))
    states = {k: np.zeros(len(DECAY_MS)) for k in SKILL_KEYS}

    for i, (t, c, row_size) in enumerate(taps):
        hand = 0 if c <= 1 else 1
        dt_global = 0.0 if i == 0 else max(0.0, t - prev_t)
        dt_same = max(0.0, t - last_col_t[c])
        dt_hand = max(0.0, t - last_hand_t[hand])
        dt_any = max(0.0, t - prev_any1)
        while c250 < i and t - taps[c250][0] > 250:
            c250 += 1
        while c500 < i and t - taps[c500][0] > 500:
            c500 += 1
        d250 = (i - c250 + 1) / 0.25
        d500 = (i - c500 + 1) / 0.5
        dens250[i] = d250; dens500[i] = d500

        jack = (190.0 / (dt_same + 35.0)) ** 1.16
        stream = (170.0 / (dt_any + 30.0)) ** 1.07
        hand_stream = (185.0 / (dt_hand + 42.0)) ** 1.08
        movement = abs(c - prev_col) / 3.0
        rhythm_ratio = safe_div(max(dt_any, 1.0), max(t - prev_any2, 1.0), 1.0)
        rhythm_chaos = abs(np.log2(clamp(rhythm_ratio, 0.2, 5.0)))
        row_chord = max(0, row_size - 1)
        chord = (row_chord + 1) ** 1.22 - 1.0

        speed_in = 0.60 * stream + 0.30 * hand_stream + 0.10 * jack
        jack_in = jack * (1.0 + 0.15 * chord)
        stam_in = 0.48 * (d500 / 11.0) + 0.27 * (d250 / 15.0) + 0.25 * stream
        chord_in = chord * (1.0 + 0.10 * min(1.5, stream))
        tech_in = (0.45 * rhythm_chaos + 0.30 * movement +
                   0.25 * ((1.0 + 0.3 * row_chord) if row_chord > 0 else 0.0))

        for j, tau in enumerate(DECAY_MS):
            decay = np.exp(-max(0.0, dt_global) / tau)
            states['speed'][j] = states['speed'][j] * decay + speed_in
            states['stamina'][j] = states['stamina'][j] * decay + stam_in
            states['chord'][j] = states['chord'][j] * decay + chord_in
            states['tech'][j] = states['tech'][j] * decay + tech_in
            states['jack'][j] = states['jack'][j] * decay + jack_in
        for k in SKILL_KEYS:
            series[k][i] = float(np.dot(states[k], DECAY_W))

        prev_t = t; prev_any2 = prev_any1; prev_any1 = t; prev_col = c
        last_col_t[c] = t; last_hand_t[hand] = t

    q = lambda a, p: float(np.percentile(a, p))
    tail = lambda a, frac: float(np.mean(np.sort(a)[-max(1, int(frac * len(a))):]))
    peak = sum(SKILL_W[k] * (0.6 * q(series[k], 97) + 0.4 * q(series[k], 90)) for k in SKILL_KEYS)
    sustain = sum(SKILL_W[k] * (0.55 * q(series[k], 75) + 0.45 * tail(series[k], 0.04))
                  for k in SKILL_KEYS)
    mid = sum(SKILL_W[k] * q(series[k], 50) for k in SKILL_KEYS)
    dens = 0.14 * np.log1p(dens250.mean()) + 0.22 * np.log1p(dens500.mean())
    length = min(3.5, (len(taps) / 600.0) ** 0.22)
    raw = 0.52 * peak + 0.26 * sustain + 0.10 * dens + 0.08 * mid + 0.04 * length
    scaled = 0.82 + 0.43 * raw

    chord_rate = float(np.mean([1.0 if rs > 1 else 0.0 for _, _, rs in taps]))
    jack_q95 = q(series['jack'], 95)
    g_chord = clamp((chord_rate - 0.40) * 3.5, 0, 1)
    g_jack = clamp((jack_q95 - 1.25) * 2.8, 0, 1)
    chordjack = clamp(2.5 * g_chord * g_jack, 0, 2.2)
    times = np.array([t for t, _, _ in taps], dtype=float)
    avg_nps = len(taps) / max(1.0, (times[-1] - times[0]) / 1000.0)
    mid_bonus = clamp((avg_nps - 9) * 0.04, 0, 0.35) * clamp((19 - avg_nps) * 0.25, 0, 1)
    return dict(score=float(scaled + chordjack + mid_bonus), peak=float(peak),
                sustain=float(sustain), nps=float(avg_nps), chord_rate=chord_rate,
                chordjack=float(chordjack), notes=len(taps),
                skills={k: q(series[k], 90) for k in SKILL_KEYS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.osu')))
        else:
            files.append(p)
    out = {}
    for f in files:
        r = curve(f)
        if r is None:
            continue
        out[os.path.basename(f)] = r
        if not a.json:
            print('%-36s 分数 %5.2f  NPS %5.2f  技巧 %.2f' % (
                os.path.basename(f)[:36], r['score'], r['nps'], r['skills']['tech']))
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
