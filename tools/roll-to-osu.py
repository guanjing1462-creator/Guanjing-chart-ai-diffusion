#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""钢琴卷 → .osu（osu!mania）转换器。

输入：[4,128] 二值钢琴卷（.npy），或采样脚本打包的 .npz（X=[n,4,128] / P=[n,4,128]）。
输出：合法的 osu!mania 4K 谱面（[General]/[Metadata]/[Difficulty]/[TimingPoints]/[HitObjects]）。

坐标与格式（与 tools/paint-gj.py 一致）
  4K 列 x = 64 / 192 / 320 / 448   ← 通用式 int((列 + 0.5) * 512 / 键数)
  type 1  = 普通音符   x,192,t,1,0,0:0:0:0:
  type 128= 长条 LN    x,192,t,128,0,<end>:0:0:0:0:

时间
  第 k 个 1/16 格 = offset + k * step_ms
  step_ms 默认按 bpm 算：60000 / bpm / 4（136.3 bpm → 110.05 ms/格）

用法
  # 单个窗口
  python3 tools/roll-to-osu.py out/diffusion/samples/roll-s0d1-seed0-00.npy -o a.osu

  # 多个窗口拼成一张完整谱面（窗口按顺序首尾相接）
  python3 tools/roll-to-osu.py out/diffusion/samples/roll-*.npy --concat -o full.osu

  # 不 concat：每个窗口一张谱
  python3 tools/roll-to-osu.py rolls.npz --outdir out/diffusion
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

WIN = 128
MAX_KEYS = 4
LANE_X = {4: [64, 192, 320, 448]}     # 4K 的标准列坐标


def lane_x(keys, c):
    """列号 → osu x 坐标。4K 时正好是 64/192/320/448。"""
    if keys in LANE_X:
        return LANE_X[keys][c]
    return int(round((c + 0.5) * 512.0 / keys))


# ---------------------------------------------------------------------------
# 读入
# ---------------------------------------------------------------------------
def load_rolls(path):
    """读一个文件 → [(名字, float 数组 [keys,128] 或 [n,keys,128])]。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
        return [(os.path.basename(path), arr)]
    if ext == '.npz':
        z = np.load(path)
        keys = [k for k in ('X', 'roll', 'rolls', 'P') if k in z.files]
        if not keys:
            raise SystemExit(f'[错误] {path} 里没有 X/roll/rolls/P 数组（有 {z.files}）')
        # 优先用阈值化后的 X，没有 X 才退到概率 P
        key = 'X' if 'X' in z.files else ('roll' if 'roll' in z.files else
                                          ('rolls' if 'rolls' in z.files else 'P'))
        arr = z[key]
        return [(f'{os.path.basename(path)}:{key}', arr)]
    if ext in ('.txt', '.json'):
        raise SystemExit(f'[错误] 不支持 {ext}；钢琴卷请存成 .npy / .npz')
    raise SystemExit(f'[错误] 不认识的文件类型 {path}')


def normalize(arr, path):
    """统一成 [n, keys, WIN] 的 float 数组。"""
    a = np.asarray(arr)
    if a.dtype == np.bool_ or a.dtype.kind not in 'fiu':
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise SystemExit(f'[错误] {path}：期望 [4,128] 或 [n,4,128]，实际 {a.shape}')
    # 允许转置存法 [n,128,4]（只有一边是 128、另一边 ≤4 时自动转置）
    n, d1, d2 = a.shape
    if d2 != WIN and d1 == WIN and d2 <= MAX_KEYS:
        a = a.transpose(0, 2, 1)
    if a.shape[2] != WIN:
        raise SystemExit(f'[错误] {path}：窗口长度应为 {WIN} 格，实际 {a.shape[2]}')
    if a.shape[1] > MAX_KEYS:
        raise SystemExit(f'[错误] {path}：键数 {a.shape[1]} 超过 {MAX_KEYS}')
    return a.astype(np.float32)


# ---------------------------------------------------------------------------
# 卷 → 音符
# ---------------------------------------------------------------------------
def roll_to_notes(roll, keys, step_ms, t0_ms, ln_mode='none', ln_min=3, max_chord=0,
                  min_gap_ms=0.0):
    """roll: [keys,128] float（>0.5 视为有音符）→ [(time, x, typ, end_time)]。

    ln_mode:
      none  —— 全是 tap（默认；二值卷本身分不清 LN 和密流）
      runs  —— 同一列连续 >= ln_min 格 → 一条 LN（起点 type 128），中间格不再出 tap
    """
    on = np.asarray(roll) > 0.5
    notes = []
    for c in range(keys):
        k = 0
        while k < WIN:
            if not on[c, k]:
                k += 1
                continue
            run = 1
            while k + run < WIN and on[c, k + run]:
                run += 1
            t_start = t0_ms + k * step_ms
            if ln_mode == 'runs' and run >= max(2, ln_min):
                t_end = t0_ms + (k + run) * step_ms          # 最后一格的右边界
                notes.append((t_start, lane_x(keys, c), 128, t_end))
            else:
                for j in range(run):
                    notes.append((t0_ms + (k + j) * step_ms, lane_x(keys, c), 1, 0.0))
            k += run

    if not notes:
        return []

    # 1) 同一时刻最多 max_chord 个音（0=不限）。二值卷没有置信度，按列号交替取舍太随机，
    #    这里直接按「列顺序」保留，结果稳定可复现。
    if max_chord and 0 < max_chord < keys:
        per_t = {}
        for nt in notes:
            per_t.setdefault(int(round(nt[0])), []).append(nt)
        kept = []
        for t in sorted(per_t):
            group = sorted(per_t[t], key=lambda z: z[1])
            kept.extend(group[:max_chord])
        notes = kept

    # 2) 取整 + 去重（同列同毫秒只能有一个音符）
    # 3) 同列 LN 内部/重叠的 tap 丢弃，与 LN 起点同毫秒的 tap 也丢掉
    out, seen = [], set()
    for t, x, typ, te in sorted(notes, key=lambda z: (z[0], z[1])):
        ti = int(round(t))
        if typ == 128:
            tei = int(round(te))
            if tei - ti < 40:          # 太短的假 LN 退化成 tap
                typ, tei = 1, 0
            if typ == 128:
                key = (ti, x, 128)
                if key in seen:
                    continue
                seen.add(key)
                out.append((ti, x, 128, tei))
                continue
        key = (ti, x, 1)
        if key in seen:
            continue
        seen.add(key)
        out.append((ti, x, 1, 0))

    # LN 区间（同列），用于清掉落进体内的 tap
    ln_int = [(t, x, e) for (t, x, typ, e) in out if typ == 128]
    if ln_int:
        fixed = []
        for (t, x, typ, e) in out:
            if typ == 1 and any(xl == x and a < t < b for (a, xl, b) in ln_int):
                continue       # tap 落在同列 LN 体内 → 丢弃（否则游戏里会互卡）
            fixed.append((t, x, typ, e))
        out = fixed

    # 4) 同列最小间隔（可选，默认关）
    if min_gap_ms and min_gap_ms > 0:
        last = {}
        keep = []
        for (t, x, typ, e) in out:
            if typ == 1:
                if x in last and t - last[x] < min_gap_ms:
                    continue
                last[x] = t
            else:
                last[x] = e
            keep.append((t, x, typ, e))
        out = keep

    return out


# ---------------------------------------------------------------------------
# 写 .osu
# ---------------------------------------------------------------------------
def write_osu(path, notes, a, step_ms):
    n_tap = sum(1 for n in notes if n[2] == 1)
    n_ln = sum(1 for n in notes if n[2] == 128)
    last_t = max([n[3] if n[2] == 128 else n[0] for n in notes], default=0)
    first_t = min([n[0] for n in notes], default=0)
    lines = []
    for (t, x, typ, e) in notes:
        if typ == 128:
            lines.append(f'{x},192,{t},128,0,{e}:0:0:0:0:')
        else:
            lines.append(f'{x},192,{t},1,0,0:0:0:0:')

    # 说明：[Difficulty] 里 CircleSize 才是 osu! 认的键数；这里额外写一行 Keys
    # 只是为了让人一眼看出 4K（osu! 会忽略不认识的键，实测 Koishi 解析结果不变）。
    head = f'''osu file format v14

[General]
AudioFilename: {a.audio}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Soft
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
SpecialStyle: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 16
TimelineZoom: 2

[Metadata]
Title:{a.title}
TitleUnicode:{a.title}
Artist:{a.artist}
ArtistUnicode:{a.artist}
Creator:{a.creator}
Version:{a.version}
Source:
Tags:diffusion sampler piano-roll AI
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:5
Keys: {a.keys}
CircleSize:{a.keys}
OverallDifficulty:{a.od}
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
{a.offset:.2f},{60000.0 / a.bpm:.6f},4,2,1,20,1,0

[HitObjects]
'''
    body = '\n'.join(lines)
    if body:
        body += '\n'
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(head + body)

    span = max(1.0, (last_t - first_t) / 1000.0)
    return {
        '文件': path,
        '音符': len(notes),
        'tap': n_tap,
        'LN': n_ln,
        '首音符_ms': first_t,
        '末音符_ms': last_t,
        '时长_s': round(span, 2),
        'NPS': round((n_tap + n_ln) / span, 2),
        'LN占比': round(n_ln / max(1, len(notes)), 4),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='钢琴卷 → osu!mania 4K 谱面')
    ap.add_argument('inputs', nargs='+', help='.npy（[4,128] 或 [n,4,128]）或 .npz')
    ap.add_argument('-o', '--out', default=None, help='输出 .osu（单张时）')
    ap.add_argument('--outdir', default='out/diffusion', help='多张时的输出目录')
    ap.add_argument('--concat', action='store_true',
                    help='把多个窗口首尾相接拼成一张完整谱面（不加则每窗口一张）')
    ap.add_argument('--bpm', type=float, default=136.3, help='BPM（默认 136.3）')
    ap.add_argument('--offset', type=float, default=0.0, help='第一个音符的时间偏移 ms（默认 0）')
    ap.add_argument('--step-ms', type=float, default=None,
                    help='1/16 格时长 ms（默认 60000/bpm/4）')
    ap.add_argument('--keys', type=int, default=4, choices=[1, 2, 3, 4], help='键数（默认 4）')
    ap.add_argument('--threshold', type=float, default=0.5, help='> 该值视为有音符（默认 0.5）')
    ap.add_argument('--ln', choices=['none', 'runs'], default='none',
                    help='LN 推断：none=全 tap（默认）；runs=同列连续 >= --ln-min 格变 LN')
    ap.add_argument('--ln-min', type=int, default=3, help='--ln runs 的最短连续格数（默认 3，小于 2 按 2 处理）')
    ap.add_argument('--max-chord', type=int, default=0, help='同一时刻最多几个音（0=不限）')
    ap.add_argument('--min-gap-ms', type=float, default=0.0, help='同列最小间隔 ms（0=不限制）')
    ap.add_argument('--title', default='AI Diffusion 4K')
    ap.add_argument('--artist', default='unknown')
    ap.add_argument('--creator', default='dsh-diffusion')
    ap.add_argument('--version', default='Diffusion Seed')
    ap.add_argument('--audio', default='song.ogg', help='[General] AudioFilename')
    ap.add_argument('--od', type=float, default=8.0, help='OverallDifficulty（默认 8）')
    ap.add_argument('--json', action='store_true', help='同时打印 JSON 统计')
    a = ap.parse_args()

    if a.bpm <= 0:
        raise SystemExit('[错误] --bpm 必须 > 0')
    if a.keys < 1:
        raise SystemExit('[错误] --keys 必须 >= 1')
    step_ms = a.step_ms if a.step_ms is not None else 60000.0 / a.bpm / 4.0
    if step_ms <= 0:
        raise SystemExit('[错误] --step-ms 必须 > 0')

    # 展开通配符（shell 没展开时兜底）与目录
    paths = []
    for p in a.inputs:
        if os.path.isdir(p):
            paths += sorted(glob.glob(os.path.join(p, '*.npy')) + glob.glob(os.path.join(p, '*.npz')))
        elif os.path.exists(p):
            paths.append(p)
        else:
            hits = sorted(glob.glob(p))
            if not hits:
                raise SystemExit(f'[错误] 找不到输入 {p}')
            paths += hits
    if not paths:
        raise SystemExit('[错误] 没有输入文件')

    # 读全部窗口，按输入顺序排队
    windows = []      # [(名字, [keys,128] float)]
    for p in paths:
        for name, arr in load_rolls(p):
            A = normalize(arr, p)
            for i in range(A.shape[0]):
                windows.append((f'{name}#{i}', A[i]))
    print(f'▶ 读入 {len(windows)} 个窗口（{len(paths)} 个文件），'
          f'{a.keys}K，step={step_ms:.3f}ms（bpm {a.bpm}）')

    bad = [w[1].shape for w in windows if w[1].shape[0] != a.keys]
    if bad:
        raise SystemExit(f'[错误] 输入的键数是 {bad[0][0]}，但 --keys={a.keys}；'
                         f'请让两者一致（4K 用 --keys 4）')

    reports = []
    os.makedirs(a.outdir, exist_ok=True)
    if a.concat:
        notes = []
        for i, (name, roll) in enumerate(windows):
            t0 = a.offset + i * WIN * step_ms          # 第 i 个窗口在时间轴上的起点
            notes += roll_to_notes(roll, a.keys, step_ms, t0, a.ln, a.ln_min,
                                   a.max_chord, a.min_gap_ms)
        if not notes:
            raise SystemExit('[错误] 阈值化后一个音符都没有，不出谱面（检查 --threshold）')
        out = a.out
        if not out:
            stem = os.path.splitext(os.path.basename(paths[0]))[0]
            out = os.path.join(a.outdir, f'{stem}-concat.osu')
        reports.append(write_osu(out, notes, a, step_ms))
    else:
        if a.out and len(windows) == 1:
            out = a.out
            notes = roll_to_notes(windows[0][1], a.keys, step_ms, a.offset, a.ln,
                                  a.ln_min, a.max_chord, a.min_gap_ms)
            if not notes:
                raise SystemExit('[错误] 阈值化后一个音符都没有，不出谱面（检查 --threshold）')
            reports.append(write_osu(out, notes, a, step_ms))
        else:
            base = os.path.splitext(os.path.basename(paths[0]))[0]
            outdir = os.path.dirname(a.out) if a.out else a.outdir
            os.makedirs(outdir or '.', exist_ok=True)
            for i, (name, roll) in enumerate(windows):
                notes = roll_to_notes(roll, a.keys, step_ms, a.offset, a.ln,
                                      a.ln_min, a.max_chord, a.min_gap_ms)
                if not notes:
                    print(f'  [跳过] 窗口 {i}（{name}）阈值化后没有音符')
                    continue
                out = os.path.join(outdir or '.', f'{base}-{i:02d}.osu')
                reports.append(write_osu(out, notes, a, step_ms))

    print('\n=== 写出 ===')
    for r in reports:
        print(f"  {r['文件']}  音符 {r['音符']}（tap {r['tap']} / LN {r['LN']}）  "
              f"时长 {r['时长_s']}s  NPS {r['NPS']}  LN占比 {r['LN占比']}")
    if a.json:
        print(json.dumps(reports, ensure_ascii=False, indent=1))
    last = reports[-1]
    print(f"\n下一步：node tools/gj/batch-koishi.mjs {last['文件']}")


if __name__ == '__main__':
    main()
