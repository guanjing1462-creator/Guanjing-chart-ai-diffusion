#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mc-profile: 读 Malody .mc（观景台谱库格式）或 .osu，用与 GJ 同一套口径出体检。

Malody .mc 是 JSON：{"meta":{"mode_ext":{"column":4}},"time":[{"beat":[bar,num,den],"bpm":..}],
                     "note":[{"beat":[bar,num,den],"column":k,"endbeat":[...]?}]}
beat 是 [小节, 分子, 分母]，绝对拍数 = bar + num/den。

用法：
  python3 tools/mc-profile.py --chart X.mc --audio X.ogg [--json out.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location('ac', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit-chart.py'))
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


def parse_mc(path):
    d = json.load(open(path, encoding='utf-8'))
    bpm = float(d['time'][0]['bpm']) if d.get('time') else 120.0
    ms_per_beat = 60000.0 / bpm
    notes = []      # (time_ms, col, end_ms or None)
    for nt in d.get('note', []):
        b = nt['beat']
        t = (b[0] + b[1] / max(1, b[2])) * ms_per_beat
        col = int(nt.get('column', 0))
        end = None
        if nt.get('endbeat'):
            e = nt['endbeat']
            end = (e[0] + e[1] / max(1, e[2])) * ms_per_beat
        notes.append((t, col, end))
    notes.sort()
    return notes, bpm


def rows_of_mc(notes):
    """把同一时刻的键合并成一行。LN 的尾巴也算一个结束点（与 osu 口径一致）。"""
    rows = {}
    for t, col, end in notes:
        rows.setdefault(round(t), set()).add(col)
    out = [(t, sorted(cs)) for t, cs in sorted(rows.items())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--audio', default=None)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    if args.chart.endswith('.mc'):
        notes, bpm = parse_mc(args.chart)
        rows = rows_of_mc(notes)
        nt = [(t, c, e) for t, c, e in notes]
    else:
        nt, _ = ac.parse_osu(args.chart)
        rows = ac.rows_of(nt)
        bpm = None

    dur = (max(t for t, c in rows) - min(t for t, c in rows)) / 1000.0
    nps = len(nt) / dur
    rps = len(rows) / dur
    npr = len(nt) / len(rows)

    ss = ssj = cc = ccj = 0
    prev = None
    for t, cs in rows:
        if prev is not None and t - prev[0] <= 120:
            if len(prev[1]) == 1 and len(cs) == 1:
                ss += 1
                if prev[1][0] == cs[0]:
                    ssj += 1
            if len(prev[1]) > 1 and len(cs) > 1:
                cc += 1
                if set(prev[1]) & set(cs):
                    ccj += 1
        prev = (t, cs)

    sizes = [len(c) for t, c in rows]
    ln = sum(1 for t, c, e in nt if e is not None)
    rep = {
        'chart': os.path.basename(args.chart), 'bpm': bpm,
        'notes': len(nt), 'rows': len(rows), 'duration_s': round(dur, 1),
        'rows_per_s': round(rps, 2), 'notes_per_row': round(npr, 2),
        'singleJack': round(ssj / max(1, ss), 3),
        'chordShared': round(ccj / max(1, cc), 3),
        'lnRatio': round(ln / max(1, len(nt)), 3),
        'chordRowRate': round(sum(1 for s in sizes if s >= 2) / len(sizes), 3),
        'size1': round(sizes.count(1) / len(sizes), 3),
        'size2': round(sizes.count(2) / len(sizes), 3),
        'size3': round(sizes.count(3) / len(sizes), 3),
        'size4': round(sizes.count(4) / len(sizes), 3),
    }

    # 对音：用谱自带的音频
    if args.audio and os.path.exists(args.audio):
        import librosa
        y, sr = librosa.load(args.audio, sr=22050, mono=True)
        hop = 256
        env = np.maximum(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), 0)
        fps = sr / hop
        p15 = float(np.percentile(env, 15))
        vals, silent = [], 0
        for t, c, e in nt:
            f0 = max(0, int((t - 60) / (1000 / fps)))
            f1 = min(len(env), int((t + 60) / (1000 / fps)) + 1)
            v = float(env[f0:f1].max()) if f1 > f0 else 0.0
            vals.append(v / (env.mean() + 1e-9))
            if v < p15:
                silent += 1
        rep['onsetStrength'] = round(float(np.mean(vals)), 3)
        rep['silentNoteRate'] = round(silent / len(nt), 4)

    print(json.dumps(rep, ensure_ascii=False))
    if args.json:
        json.dump(rep, open(args.json, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
