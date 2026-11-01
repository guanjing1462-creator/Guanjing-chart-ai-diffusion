#!/usr/bin/env python3
"""修正「谱面时间轴 vs 音频」的整体偏移，并筛掉对不齐的谱面。

做法：对每张谱，把音符时间按候选位移 s 平移，算「音符密度 vs 音频强度」相关，
取相关最大的 s 作为最佳位移；若 |s| 超过阈值则判定为可疑（可能是不同版本音频）。

输出 out/audio-dl-fixed/index.json：每张谱的 best_shift / r_before / r_after / 判定。
用法：python3 tools/audio-offset-fix.py --limit 60
"""
import argparse, glob, json, os
import numpy as np
import librosa

XS = {64: 0, 192: 1, 320: 2, 448: 3}


def parse(path):
    raw = open(path, encoding='utf-8', errors='ignore').read().split('\n')
    sec, notes, timing = None, [], []
    audio = None
    for ln in raw:
        s = ln.strip()
        if s.startswith('[') and s.endswith(']'):
            sec = s[1:-1]
            continue
        if not s or s.startswith('//'):
            continue
        if sec == 'General' and s.lower().startswith('audiofilename'):
            audio = s.split(':', 1)[1].strip()
        elif sec == 'TimingPoints':
            p = s.split(',')
            if len(p) >= 2:
                try:
                    bl = float(p[1]); un = int(p[6]) if len(p) > 6 else 1
                except ValueError:
                    continue
                if un == 1 and bl > 0:
                    timing.append((float(p[0]), bl))
        elif sec == 'HitObjects':
            p = s.split(',')
            if len(p) < 5:
                continue
            try:
                x, t, typ = int(p[0]), int(p[2]), int(p[3])
            except ValueError:
                continue
            if x in XS and (typ & 1 or typ & 128):
                notes.append(t)
    return sorted(notes), timing, audio


def corr_profile(notes_ms, env, fps, bin_ms=250.0):
    if len(notes_ms) < 100:
        return None
    t = np.array(notes_ms, dtype=float) / 1000.0
    t0, t1 = 0.0, len(env) / fps
    edges = np.arange(t0, t1, bin_ms / 1000.0)
    if len(edges) < 10:
        return None
    h, _ = np.histogram(t, bins=edges)
    af = np.array([env[int(a * fps):max(int(a * fps) + 1, int(b * fps))].sum()
                   for a, b in zip(edges, edges[1:])], dtype=float)
    h = h.astype(float)
    if h.std() < 1e-9 or af.std() < 1e-9:
        return None
    return float(np.corrcoef(h, af[:len(h)])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='out/audio-dl')
    ap.add_argument('--out', default='out/audio-dl-fixed/index.json')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shifts', default='-2500,2500,100')
    ap.add_argument('--max-shift-ms', type=float, default=250.0,
                    help='超过这个位移仍能提升相关，说明谱面被打包时改过 offset，标为可疑')
    a = ap.parse_args()
    lo, hi, step = (float(x) for x in a.shifts.split(','))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    dirs = sorted(d for d in glob.glob(os.path.join(a.root, '*')) if os.path.isdir(d))
    if a.limit:
        dirs = dirs[:a.limit]
    rows = []
    for d in dirs:
        osus = glob.glob(os.path.join(d, '*.osu'))
        if not osus:
            continue
        f = osus[0]
        try:
            notes, timing, audio_name = parse(f)
        except Exception as e:
            rows.append({'dir': d, 'file': f, 'error': f'解析失败 {e}'})
            continue
        if not notes or not timing:
            rows.append({'dir': d, 'file': f, 'error': '无音符或无红线'})
            continue
        aud = None
        for cand in (audio_name, 'audio.mp3', 'audio.ogg'):
            if cand:
                p = os.path.join(d, cand)
                if os.path.exists(p):
                    aud = p; break
        if aud is None:
            cands = [x for x in glob.glob(os.path.join(d, '*')) if x.lower().endswith(('.mp3', '.ogg', '.wav'))]
            aud = cands[0] if cands else None
        if aud is None:
            rows.append({'dir': d, 'file': f, 'error': '无音频'})
            continue
        try:
            y, sr = librosa.load(aud, sr=22050, mono=True, duration=420)
        except Exception as e:
            rows.append({'dir': d, 'file': f, 'error': f'读音频失败 {e}'})
            continue
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
        fps = sr / 256.0
        r0 = corr_profile(notes, env, fps)
        if r0 is None:
            rows.append({'dir': d, 'file': f, 'error': '相关算不出'})
            continue
        best = (0.0, r0)
        for s in np.arange(lo, hi + 1e-9, step):
            if abs(s) < 1e-9:
                continue
            r = corr_profile([n + s for n in notes], env, fps)
            if r is not None and r > best[1]:
                best = (float(s), r)
        shift, r1 = best
        verdict = 'ok' if abs(shift) <= a.max_shift_ms else 'shift_large'
        if r1 < 0.20:
            verdict = 'low_corr'
        rows.append({'dir': d, 'file': os.path.basename(f), 'audio': os.path.basename(aud),
                     'notes': len(notes), 'r_before': round(r0, 4), 'r_after': round(r1, 4),
                     'best_shift_ms': shift, 'verdict': verdict})
    ok = [r for r in rows if r.get('verdict') == 'ok']
    json.dump({'rows': rows, 'total': len(rows), 'ok': len(ok),
               'r_median_after': float(np.median([r['r_after'] for r in ok])) if ok else 0.0},
              open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'处理 {len(rows)} 个包：ok {len(ok)} / 位移过大 {sum(1 for r in rows if r.get("verdict")=="shift_large")} '
          f'/ 相关低 {sum(1 for r in rows if r.get("verdict")=="low_corr")} / 错误 {sum(1 for r in rows if "error" in r)}')
    if ok:
        print(f'ok 谱面的 r 中位 {np.median([r["r_after"] for r in ok]):.3f}')
        big = [r for r in rows if abs(r.get('best_shift_ms', 0)) > 200]
        print(f'位移 >200ms 的 {len(big)} 个（示例：{[ (r["dir"].split("/")[-1], r["best_shift_ms"]) for r in big[:5] ]}）')
    print('已写', a.out)


if __name__ == '__main__':
    main()
