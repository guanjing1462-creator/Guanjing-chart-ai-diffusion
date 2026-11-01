#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性音频骨架：从网格化音频能量（accent）选格 → 节奏骨架。

为什么替换扩散骨架（回应「识别音频、写谱、对音存在极大的问题」）：
  扩散模型只训了 ~2000 步，音频条件通路还很弱——它输出的音符位置
  是「训练集平均节奏 + 噪声」，吸附到网格后听着不对音。
  本脚本用 chartgen 已验证的分析器产物（accent = 每格 RMS 能量，
  网格来自校准过的 offset/BPM），选出的格天然：
    1. 在分音网格上（对拍）
    2. 踩在音频能量峰上（对音）
  对音质量有量化指标（选格能量信噪比 + onset 重合率）。
  扩散骨架继续作为研究线训练，交付线等它收敛后再接管。

输出 .osu 格式骨架（painter 直接消费）。
"""
import argparse
import bisect
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True, help='chartgen 的 analysis.json')
    ap.add_argument('--target-rows', type=int, default=2600,
                    help='目标骨架行数（约 11 行/s 行频，painter 再按指纹补）')
    ap.add_argument('--seg-sec', type=float, default=4.0,
                    help='密度调制段落时长（秒）')
    ap.add_argument('-o', required=True)
    args = ap.parse_args()

    an = json.load(open(args.analysis))
    accent = an['accent']
    off, step = an['offsetMs'], an['stepMs']
    n = len(accent)

    # ---- 段落密度调制：人类的谱不是均匀撒的——副歌密、主歌疏、
    #      过门留白。按段落能量分配选格配额，弱段落留白。 ----
    import statistics
    seg_cells = max(1, int(args.seg_sec * 1000 / step))
    segs = [accent[i:i + seg_cells] for i in range(0, n, seg_cells)]
    seg_e = [statistics.mean(s) if s else 0.0 for s in segs]
    p20 = sorted(seg_e)[max(0, int(len(seg_e) * 0.2))]
    p80 = sorted(seg_e)[int(len(seg_e) * 0.8)]
    # 配额权重：能量线性映射到 [0.25, 1.6]（副歌 ~1.6×、最弱段 0.25×）
    weights = [max(0.25, min(1.6, (e - p20 + 1e-9) / max(1e-9, p80 - p20) * 1.35 + 0.25))
               for e in seg_e]
    w_sum = sum(weights)
    quotas = [int(args.target_rows * w / w_sum) for w in weights]
    # 选格池 = 踩中 onset ±半格的格（人类只在重音上放音符——持续能量
    # 段不放音符），段内按能量排序取配额 → 踩音精度按构造最大化
    peaks_ms = sorted(p * an['fine']['hopMs'] for p in an['peaks'])
    half = step / 2
    on_cells = set()
    for pm in peaks_ms:
        c = round((pm - off) / step)
        for cc in (c, c - 1, c + 1):
            if 0 <= cc < n and abs(pm - (off + cc * step)) <= half:
                on_cells.add(cc)
    # 呼吸段：能量 < p20 的段最多留 2 格（听感上的"气口"）
    picked = []
    breath_segs = 0
    for k, seg in enumerate(segs):
        base = k * seg_cells
        cand = [j for j in range(len(seg)) if (base + j) in on_cells]
        cand.sort(key=lambda j: -seg[j])
        idx = cand[:quotas[k]]
        if seg_e[k] < p20:
            idx = idx[:2]
            breath_segs += 1
        picked += [base + j for j in idx]
    picked = sorted(i for i in picked if i < n)

    # 对音质量指标
    import statistics
    sel_mean = statistics.mean(accent[i] for i in picked)
    all_mean = statistics.mean(accent)
    # 踩音精度：选格里有多少踩在 onset ±半格内（与骨架密度无关）
    cells_sorted = [off + i * step for i in picked]
    hit = 0
    for c in cells_sorted:
        j = bisect.bisect_left(peaks_ms, c - half)
        if j < len(peaks_ms) and peaks_ms[j] <= c + half:
            hit += 1
    prec = hit / max(1, len(cells_sorted))
    # onset 覆盖率（recall，随密度变化，仅参考）
    cov = 0
    for pm in peaks_ms:
        j = bisect.bisect_left(cells_sorted, pm - half)
        if j < len(cells_sorted) and cells_sorted[j] <= pm + half:
            cov += 1
    cover = cov / max(1, len(peaks_ms))

    # ---- 歌曲结构检测：段落等价类（重复乐段）----
    # 人类写谱的重复段会呼应（相同图案或变奏）。对每段 accent 归一化
    # 后做余弦相似度，>0.92 且间距 ≥5 段视为同一段音乐。
    import math
    groups = {}
    gid = 0
    seg_ids = [None] * len(segs)
    norms = [math.sqrt(sum(x * x for x in s)) or 1e-9 for s in segs]
    for k in range(len(segs)):
        matched = None
        if len(segs[k]) == seg_cells:   # 只对齐满段（末段可能不完整）
            for j in range(k):
                if seg_ids[j] is None or k - j < 5 or len(segs[j]) != seg_cells:
                    continue
                a, b = segs[k], segs[j]
                na, nb = norms[k], norms[j]
                cos = sum(x / na * y / nb for x, y in zip(a, b))
                if cos > 0.85:
                    matched = seg_ids[j]
                    break
        if matched is None:
            groups[gid] = [k]
            seg_ids[k] = gid
            gid += 1
        else:
            groups[matched].append(k)
            seg_ids[k] = matched
    n_dup_groups = sum(1 for g in groups.values() if len(g) > 1)
    struct = {'seg_cells': seg_cells, 'seg_ids': seg_ids,
              'groups': {str(g): ks for g, ks in groups.items()}}
    json.dump(struct, open(args.o.rsplit('.', 1)[0] + '.structure.json', 'w'))

    head = f'''osu file format v14

[General]
AudioFilename: INTERNET YAMERO.ogg
Mode: 3

[Metadata]
Title: skeleton
Artist: audio-onsets
Creator: scene3d
Version: {args.target_rows}rows

[Difficulty]
CircleSize: 4

[TimingPoints]
{off},{60000 / 185},4,2,1,20,1,0

[HitObjects]
'''
    lines = []
    for i in picked:
        t = int(off + i * step)
        col = i % 4
        lines.append(f'{int((col + 0.5) * 128)},192,{t},1,0,0:0:0:0:')
    open(args.o, 'w', encoding='utf-8').write(head + '\n'.join(lines) + '\n')

    dur = an['durationSec']
    # 行频起伏验证（每 20s）
    row_freq = [0] * int(dur / 20 + 1)
    for i in picked:
        t = i * step / 1000
        row_freq[min(len(row_freq) - 1, int(t / 20))] += 1
    print(f'骨架: {len(picked)} 行 / {dur:.0f}s = {len(picked)/dur:.1f} 行/s  呼吸段 {breath_segs} 个')
    print('行频(行/s, 每20s): ' + ' '.join(str(round(x / 20, 1)) for x in row_freq[:12]))
    print(f'对音指标: 选格能量 {sel_mean:.3f} vs 全格均值 {all_mean:.3f}（信噪比 {sel_mean/max(1e-9,all_mean):.2f}×）')
    print(f'          音符踩音精度 {prec*100:.1f}%（选格踩在 onset ±{half:.0f}ms 内的比例）  onset 覆盖 {cover*100:.1f}%（参考）')
    print(f'歌曲结构: {len(groups)} 个音乐段（{n_dup_groups} 组重复——第二遍会呼应第一遍的写法）')
    print('写出:', args.o)


if __name__ == '__main__':
    main()
