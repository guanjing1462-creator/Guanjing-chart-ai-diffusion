#!/usr/bin/env python3
"""BPM 检测 v2：拍级评分 + 精细搜索 + 八度判别。

相对 v1 的改动：
1. 八度判别不用「步长是否舒适」这种启发式，改用**拍级评分**：
   在给定 BPM 下，把所有预测拍点上的包络强度取平均，并惩罚「空拍」
   （预测有拍但实际没有起音）—— 这能自然排除 2×BPM（因为一半拍点是空的）。
2. BPM 两段搜索：先 0.25 粗搜，再在最优附近 ±1 BPM 内以 0.01 精搜。
3. offset 在 0..period 内以 5ms 步长搜索，再以 1ms 精修。

输出与 chartgen 同构的 analysis JSON。
"""
import argparse
import json
import numpy as np


def onset_env(y, sr, hop):
    import librosa
    env = np.maximum(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), 0)
    return env


def beat_score(env, fps, bpm, off_ms, win_ms=28.0):
    """拍级评分：预测拍点处的包络均值 − 空拍惩罚。"""
    if bpm <= 0:
        return -1e9, 0.0
    period = 60000.0 / bpm
    dur = len(env) / fps * 1000.0
    t = off_ms
    vals = []
    while t < dur:
        f = int(t / 1000.0 * fps)
        w = int(win_ms / 1000.0 * fps)
        a, b = max(0, f - w), min(len(env), f + w + 1)
        vals.append(float(env[a:b].max()) if b > a else 0.0)
        t += period
    if len(vals) < 8:
        return -1e9, 0.0
    v = np.array(vals)
    v = v / (env.max() + 1e-9)
    hit = float(v.mean())
    # 空拍惩罚：低于中位 25% 的拍点比例
    empty = float((v < np.percentile(v, 25)).mean())
    return hit - 0.9 * empty, hit


def best_offset(env, fps, bpm, win_ms=28.0, coarse=5.0, fine=1.0):
    period = 60000.0 / bpm
    best = (-1e9, 0.0, 0.0)
    off = 0.0
    while off < period:
        s, h = beat_score(env, fps, bpm, off, win_ms)
        if s > best[0]:
            best = (s, off, h)
        off += coarse
    o0 = best[1]
    off = max(0.0, o0 - coarse)
    while off <= o0 + coarse:
        s, h = beat_score(env, fps, bpm, off, win_ms)
        if s > best[0]:
            best = (s, off, h)
        off += fine
    return best


def search_bpm(env, fps, lo=60.0, hi=220.0, step=0.25):
    out = []
    bpm = lo
    while bpm <= hi:
        s, off, hit = best_offset(env, fps, bpm, coarse=8.0, fine=2.0)
        out.append((s, bpm, off, hit))
        bpm += step
    out.sort(key=lambda x: -x[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--bpm-only', action='store_true', help='只打印 BPM（供批量测试）')
    a = ap.parse_args()

    import librosa
    y, sr = librosa.load(a.audio, sr=22050, mono=True)
    dur_ms = len(y) / sr * 1000.0
    hop = 256
    fps = sr / hop
    env = onset_env(y, sr, hop)

    # 1) 粗搜
    coarse = search_bpm(env, fps, 60.0, 220.0, 0.25)
    # 2) 取前 3 个候选，各自精搜 ±1 BPM @0.01
    refined = []
    seen = set()
    for s0, bpm0, _, _ in coarse[:6]:
        if round(bpm0) in seen:
            continue
        seen.add(round(bpm0))
        best = (-1e9, bpm0, 0.0, 0.0)
        b = max(50.0, bpm0 - 1.0)
        while b <= min(240.0, bpm0 + 1.0):
            s, off, hit = best_offset(env, fps, b, coarse=4.0, fine=1.0)
            if s > best[0]:
                best = (s, b, off, hit)
            b += 0.01
        refined.append(best)
        if len(refined) >= 3:
            break
    refined.sort(key=lambda x: -x[0])

    # 3) 八度判别：对 1/2、2× 也精搜，比较拍级评分（空拍惩罚会压制 2×）
    cands = []
    for s, bpm, off, hit in refined[:2]:
        for f in (0.5, 1.0, 2.0):
            b = bpm * f
            if not (55.0 <= b <= 240.0):
                continue
            ss, oo, hh = best_offset(env, fps, b, coarse=4.0, fine=1.0)
            cands.append((ss, b, oo, hh))
    cands.sort(key=lambda x: -x[0])
    # 八度最终裁决：
    #   - 若半速/倍速的评分与最优接近（≥85%），说明两种记谱都成立（如 128 也能记成 64）；
    #   - 此时优先取「常规歌曲范围」[95,200] 内的那个（节奏游戏曲目绝大多数落在这里）。
    if cands:
        _best = cands[0][0]
        _ok = [c for c in cands if c[0] >= 0.85 * _best]
        _in = [c for c in _ok if 95.0 <= c[1] <= 200.0]
        if _in:
            cands = sorted(_in, key=lambda x: -x[0]) + [c for c in cands if c not in _in]
    score, bpm, off, hit = cands[0]
    step = 60000.0 / bpm / 4.0
    print(f'清音评分 {score:.3f} 命中率 {hit:.3f}', flush=True)
    for c in cands[:5]:
        print(f'  候选 bpm {c[1]:7.2f} 评分 {c[0]:.3f} 命中 {c[3]:.3f}', flush=True)
    print(f'选定: bpm {bpm:.2f} offset {off:.2f}ms 1/16 {step:.2f}ms', flush=True)

    if a.bpm_only:
        return

    n_cells = int(dur_ms / step) + 2
    cell_env = np.zeros(n_cells)
    for i in range(n_cells):
        t0 = off + i * step
        f0 = int(t0 / 1000.0 * fps)
        f1 = min(len(env), int((t0 + step) / 1000.0 * fps) + 1)
        if f1 > f0 > 0:
            cell_env[i] = float(env[f0:f1].max())
    if cell_env.max() > 0:
        cell_env = cell_env / cell_env.max()

    out = dict(path=a.audio, bpm=round(bpm, 3), offsetMs=round(off, 2),
               stepMs=round(step, 3), grid='1/16', cellEnv=[round(float(x), 4) for x in cell_env])
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
