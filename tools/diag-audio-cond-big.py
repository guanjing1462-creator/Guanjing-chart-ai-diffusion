#!/usr/bin/env python3
"""对齐诊断：解释「250ms 密度-强度相关偏低」到底是数据错位还是指标不敏感。

三组证据，对同一批谱面：
  A. 音频起点 offset：把谱面相对音频做 ±3s 平移扫描，看最佳 r 落在哪个 shift
  B. 音频 onset 命中率：librosa 检测到的 onset 有多少落在音符 ±50ms 内，
     与「谱面随机循环平移」的对照比（这才是真正的对齐度量）
  C. 指标分箱敏感性：250ms / 500ms / 1000ms 分箱下同一批谱面的 r
  D. 本地 6 首人工谱（ref/multisong + out/multisong/charts）作为同口径标定
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 22050
HOP_V = 256


def load_env(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_V)
    return y, env, SR / HOP_V


def bins_of(env, fps, bin_s):
    nb = int(len(env) / fps / bin_s) + 1
    grid = np.arange(nb + 1) * bin_s
    out = np.array([env[int(a * fps):max(int(a * fps) + 1, int(b * fps))].sum()
                    for a, b in zip(grid, grid[1:])], dtype=float)
    return out, grid


def r_of(note_ms, audio, grid, shift_ms=0.0):
    h, _ = np.histogram((note_ms + shift_ms) / 1000.0, bins=grid)
    h = h.astype(float)
    a = audio[:len(h)]
    if h.std() < 1e-9 or a.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(h, a)[0, 1])


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'bacb', os.path.join(ROOT, 'tools', 'build-audio-cond-big.py'))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)

    data = os.path.join(ROOT, 'out', 'diff-data-audio-big')
    man = json.load(open(os.path.join(data, 'manifest.json'), encoding='utf-8'))
    best = {}
    for m in man:
        k = m['SetID']
        if k not in best or m['windows'] > best[k]['windows']:
            best[k] = m
    picks = sorted(best.values(), key=lambda m: -m['windows'])[:8]

    pairs = []
    for m in picks:
        pairs.append((f'{m["SetID"]} {m["title"][:26]}',
                      os.path.join(ROOT, 'out', 'audio-dl', str(m['SetID']), m['chart']),
                      os.path.join(ROOT, 'out', 'audio-dl', str(m['SetID']), m['audio'])))
    # 本地 6 首人工谱
    local = os.path.join(ROOT, 'out', 'multisong', 'charts')
    meta = json.load(open(os.path.join(ROOT, 'out', 'multisong', 'cond-meta.json'),
                          encoding='utf-8'))['songs']
    for k, v in meta.items():
        osu = os.path.join(local, k + '.osu')
        aud = os.path.join(ROOT, v['audio']) if not os.path.isabs(v['audio']) else v['audio']
        if os.path.exists(osu) and os.path.exists(aud):
            pairs.append((f'[本地] {k[:24]}', osu, aud))

    rows = []
    print(f'诊断 {len(pairs)} 组 (谱面, 音频)\n', flush=True)
    print(f'{"来源":34s} {"r250":>7s} {"bestR":>7s} {"bestShift":>9s} '
          f'{"命中@50ms":>9s} {"随机命中":>8s} {"r500":>7s} {"r1000":>7s}', flush=True)
    for name, osu, aud in pairs:
        if not (os.path.exists(osu) and os.path.exists(aud)):
            continue
        ts, offset_ms, beat_len = B.note_times(osu)
        if len(ts) < 32:
            continue
        y, env, fps = load_env(aud)
        dur = len(y) / SR
        a250, g250 = bins_of(env, fps, 0.25)
        r250 = r_of(ts, a250, g250)

        # A. shift 扫描
        shifts = np.arange(-3000, 3001, 100)
        rs = [r_of(ts, a250, g250, s) for s in shifts]
        i = int(np.argmax(rs))
        best_r, best_shift = rs[i], int(shifts[i])

        # B. onset 命中率
        import librosa
        on = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP_V,
                                        units='time', backtrack=False) * 1000.0
        notes_sorted = np.sort(ts)
        def hit(nt):
            if len(on) < 8 or len(nt) < 8:
                return 0.0
            idx = np.searchsorted(notes_sorted, nt)
            idx = np.clip(idx, 1, len(notes_sorted) - 1)
            d = np.minimum(np.abs(nt - notes_sorted[idx - 1]),
                           np.abs(nt - notes_sorted[idx]))
            return float(np.mean(d <= 50.0))
        h_align = hit(on)
        rng = np.random.default_rng(7)
        h_rand = float(np.mean([hit(np.mod(on + s, dur * 1000.0))
                                for s in rng.uniform(0, dur * 1000.0, 8)]))

        a500, g500 = bins_of(env, fps, 0.5)
        a1000, g1000 = bins_of(env, fps, 1.0)
        rows.append(dict(name=name, r250=r250, best_r=best_r, best_shift=best_shift,
                         hit=h_align, hit_rand=h_rand,
                         r500=r_of(ts, a500, g500), r1000=r_of(ts, a1000, g1000)))
        print(f'{name:34s} {r250:+7.3f} {best_r:+7.3f} {best_shift:+6d}ms '
              f'{h_align:9.3f} {h_rand:8.3f} {rows[-1]["r500"]:+7.3f} '
              f'{rows[-1]["r1000"]:+7.3f}', flush=True)
        del y, env

    if not rows:
        return 1
    loc = [x for x in rows if x['name'].startswith('[本地]')]
    new = [x for x in rows if not x['name'].startswith('[本地]')]
    out = {'rows': rows}
    for tag, grp in (('osu.direct', new), ('本地人工谱', loc)):
        if not grp:
            continue
        out[tag] = {
            'n': len(grp),
            'r250_median': round(float(np.median([x['r250'] for x in grp])), 3),
            'best_r_median': round(float(np.median([x['best_r'] for x in grp])), 3),
            'best_shift_median': int(np.median([x['best_shift'] for x in grp])),
            'hit_median': round(float(np.median([x['hit'] for x in grp])), 3),
            'hit_rand_median': round(float(np.median([x['hit_rand'] for x in grp])), 3),
            'r500_median': round(float(np.median([x['r500'] for x in grp])), 3),
            'r1000_median': round(float(np.median([x['r1000'] for x in grp])), 3),
        }
        print(f'\n[{tag}] n={out[tag]["n"]}  r250 中位 {out[tag]["r250_median"]:+.3f} · '
              f'最佳 r 中位 {out[tag]["best_r_median"]:+.3f} @ shift {out[tag]["best_shift_median"]}ms · '
              f'onset 命中 {out[tag]["hit_median"]:.3f}（随机对照 {out[tag]["hit_rand_median"]:.3f}）· '
              f'r500 {out[tag]["r500_median"]:+.3f} · r1000 {out[tag]["r1000_median"]:+.3f}')
    rep = os.path.join(ROOT, 'out', 'diffusion', 'round146-audio-big')
    os.makedirs(rep, exist_ok=True)
    json.dump(out, open(os.path.join(rep, 'diagnosis.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\n→ {os.path.join(rep, "diagnosis.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
