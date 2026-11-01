#!/usr/bin/env python3
"""LN 感知的 1/32 音乐卷数据集：占用卷 + **长条体（hold）** + 长条头。

为什么需要这个文件
------------------
`out/diff-data-grouped/dataset.npz` 的 X 只保留 **LN 起点**（type&128 的 endTime 被丢掉，
见 round136 REPORT 的 `ln_duration_lost`）。模型因此不可能学到「人写的 LN 有多长、
什么时候写、怎么写」。本脚本在完全相同的分组划分/网格口径下，额外保存：

    X  [N,4,256] uint8   占用：LN 覆盖的所有格（含头到尾）、普通音符只占起点格
    L  [N,4,256] uint8   LN 体：LN 覆盖的格 = 1（普通音符 = 0）
    H  [N,4,256] uint8   LN 头：LN 起点格 = 1（普通音符 = 0）

口径（与既有数据集对齐）
------------------------
- **谱面清单与 train/val/test 分组原样继承**
  `out/diffusion/round136-data/manifest.json`，`split` 按 group_id 切好，
  本脚本**不做任何窗口级随机划分**；窗口只继承其 chart 的 split。
- **网格 = 真实红线**：只读 `[TimingPoints]`、只取 uninherited=1（复用
  `tools/gj-data-manifest.py::parse_chart` 的同一份解析器），
  `cell(t) = round(8 * beat(t))`，8 格/拍 = 1/32。
  **>3 条红线的变速谱跳过并计数**（`--max-red-lines 3`）。
- **LN 判定**：`type&128` → LN，`endTime` 取 `x,y,time,type,hitsound,endTime:...`
  最后一个逗号字段里冒号前的时间；`type&1` → 普通音符。
- 窗口：256 格、步长 128 格（= 半窗），起点对齐全局格 `k*stride`（不逐窗归零），
  窗口内被占用**行数** ≥ `--min-fill`（与 diff-data32-grouped.py 同口径）。
  窗口覆盖范围用 X 的最左/最右占用格（含 LN 尾），因此 LN 尾不会被切掉；
  跨窗 LN 自然以 carry-in（窗口内 L 段起点不是真头）形式保留。
- LN 长度（格）定义：**覆盖格数 = end_cell - start_cell + 1**（1/32 格），
  另存时长 `end_cell - start_cell` 便于对照拍长口径。

输出
----
out/diff-data-ln/dataset.npz
    X / L / H  [N,4,256] uint8
    style/diff/diff_nps/bpm/chart_id/group_id/split/start_cell
    ln_heads / ln_cells  [N] int16  每窗口的 LN 头数（L 连续段的段数）与 LN 体格数
    window_cells / grid_division 标量
out/diff-data-ln/report.json  全部统计 + 断言实测值

用法
----
  python3 tools/diff-data-ln.py                       # 全量
  python3 tools/diff-data-ln.py --limit 30            # 每风格前 30 张（冒烟）
  python3 tools/diff-data-ln.py --max-red-lines 3     # 默认 3
"""
import argparse
import array
import bisect
import collections
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np

STYLES = ("tech", "jack", "speed", "stamina")
SID = {s: i for i, s in enumerate(STYLES)}
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SPLIT_NAME = {v: k for k, v in SPLIT_CODE.items()}
GRID_DIV = 8          # 每拍 8 格 → 1/32
WIN = 256             # 窗口长（格）= 32 拍 = 8 小节 4/4
STRIDE = 128          # 滑窗步长（格）= 半窗
NLANES = 4

PARSER_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gj-data-manifest.py")
PARSER_VERSION = "ln-1/32-real-red-timing-v1"


# ---------------------------------------------------------------------------
# 复用 manifest 的严格解析器（同一份实现，避免两套 .osu 语义）
# ---------------------------------------------------------------------------
def load_parser_module(path=PARSER_MODULE):
    spec = importlib.util.spec_from_file_location("gj_data_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 拍坐标 ⇄ 绝对格 ⇄ 毫秒（真实红线）
# ---------------------------------------------------------------------------
def beat_at(segments, time_ms):
    offsets = [s["offset_ms"] for s in segments]
    i = max(0, bisect.bisect_right(offsets, time_ms) - 1)
    s = segments[i]
    return s["beat_start"] + (time_ms - s["offset_ms"]) / s["beat_length_ms"]


def cell_of(segments, time_ms):
    """cell(t) = round(8 * beat(t))。"""
    return int(round(GRID_DIV * beat_at(segments, time_ms)))


def segment_of_cell(segments, cell):
    beat = cell / float(GRID_DIV)
    i = max(0, bisect.bisect_right([s["beat_start"] for s in segments], beat) - 1)
    return segments[i]


def ms_of_cell(segments, cell):
    s = segment_of_cell(segments, cell)
    return s["offset_ms"] + (cell / float(GRID_DIV) - s["beat_start"]) * s["beat_length_ms"]


# ---------------------------------------------------------------------------
# 单张谱 → 每键的 tap 格 / LN 区间格（按下标可直接二分检索）
# ---------------------------------------------------------------------------
class ChartEvents:
    """一张谱里每键的 tap 格与 LN 区间，全部为 int64 数组，供窗口切片。"""

    __slots__ = ("taps", "ln_s", "ln_e", "ln_emax", "degenerate", "overlap",
                 "adjacent")

    def __init__(self):
        self.taps = [np.empty(0, dtype=np.int64) for _ in range(NLANES)]
        self.ln_s = [np.empty(0, dtype=np.int64) for _ in range(NLANES)]
        self.ln_e = [np.empty(0, dtype=np.int64) for _ in range(NLANES)]
        self.ln_emax = [np.empty(0, dtype=np.int64) for _ in range(NLANES)]
        self.degenerate = 0
        self.overlap = 0    # 同键 LN 互相覆盖的个数（非法谱里偶发，0.07%）
        self.adjacent = 0   # 同键 LN 首尾相接（s == 上一段 e+1）→ L 连成一段


def chart_events(parsed, segments):
    """解析结果 → ChartEvents。

    LN 量化后 end<=start 的（退化）记 1 格覆盖并计数；LN 区间按 start 排序，
    并给出 end 的累积最大值（emax 单调不减），这样「与窗口相交的 LN」可以二分定位。
    """
    ev = ChartEvents()
    raw_taps = [[] for _ in range(NLANES)]
    raw_lns = [[] for _ in range(NLANES)]
    for lane, start, end in parsed["notes"]:
        s = cell_of(segments, start)
        if end > start:
            e = cell_of(segments, end)
            if e <= s:
                e = s + 1
                ev.degenerate += 1
            raw_lns[lane].append((s, e))
        else:
            raw_taps[lane].append(s)
    for lane in range(NLANES):
        ev.taps[lane] = np.sort(np.asarray(raw_taps[lane], dtype=np.int64))
        if raw_lns[lane]:
            arr = np.asarray(sorted(raw_lns[lane]), dtype=np.int64)
            ev.ln_s[lane] = arr[:, 0]
            ev.ln_e[lane] = arr[:, 1]
            ev.ln_emax[lane] = np.maximum.accumulate(arr[:, 1])
            if len(arr) > 1:
                ev.overlap += int((arr[1:, 0] <= arr[:-1, 1]).sum())
                ev.adjacent += int((arr[1:, 0] == arr[:-1, 1] + 1).sum())
    return ev


def span_and_rows(ev):
    """返回 (cmin, cmax, row_prefix)。

    row_prefix[i] = [cmin, cmin+i) 内「有任一键占用」的格数（union，前缀和）。
    占用 = tap 格 ∪ LN 覆盖格；LN 尾也计入，所以窗口边界不会切掉 LN 尾。
    """
    lo_all, hi_all = [], []
    for lane in range(NLANES):
        if len(ev.taps[lane]):
            lo_all.append(int(ev.taps[lane][0]))
            hi_all.append(int(ev.taps[lane][-1]))
        if len(ev.ln_s[lane]):
            lo_all.append(int(ev.ln_s[lane].min()))
            hi_all.append(int(ev.ln_e[lane].max()))
    cmin, cmax = min(lo_all), max(hi_all)
    span = cmax - cmin + 1
    diff = np.zeros(span + 1, dtype=np.int16)
    for lane in range(NLANES):
        t = ev.taps[lane]
        if len(t):
            diff[t - cmin] += 1
            diff[t - cmin + 1] -= 1
        for s, e in zip(ev.ln_s[lane].tolist(), ev.ln_e[lane].tolist()):
            diff[s - cmin] += 1
            diff[e - cmin + 1] -= 1
    occ = np.cumsum(diff[:span]) > 0
    prefix = np.zeros(span + 1, dtype=np.int32)
    np.cumsum(occ, out=prefix[1:])
    return cmin, cmax, prefix


def window_starts(cmin, cmax, stride):
    lo = int(math.floor(cmin / stride))
    hi = int(math.floor(cmax / stride))
    return [k * stride for k in range(lo, hi + 1)]


def rows_in(prefix, cmin, lo, hi):
    """[lo,hi) 内被占用行数；越界裁剪。"""
    span = len(prefix) - 1
    a = max(0, min(span, lo - cmin))
    b = max(0, min(span, hi - cmin))
    return int(prefix[b] - prefix[a]) if b > a else 0


def build_window(ev, start, cells):
    """把 [start,start+cells) 铺成 (X, L, H) 三个 [4,cells] uint8。"""
    x = np.zeros((NLANES, cells), dtype=np.uint8)
    l = np.zeros((NLANES, cells), dtype=np.uint8)
    h = np.zeros((NLANES, cells), dtype=np.uint8)
    lo, hi = start, start + cells
    for lane in range(NLANES):
        t = ev.taps[lane]
        if len(t):
            a = int(np.searchsorted(t, lo, side="left"))
            b = int(np.searchsorted(t, hi, side="left"))
            if b > a:
                x[lane, t[a:b] - lo] = 1
        s_arr, e_arr, emax = ev.ln_s[lane], ev.ln_e[lane], ev.ln_emax[lane]
        if len(s_arr):
            i = int(np.searchsorted(emax, lo, side="left"))      # 第一个 e >= lo
            j = int(np.searchsorted(s_arr, hi, side="left"))     # 第一个 s >= hi
            for k in range(i, j):
                s = int(s_arr[k])
                e = int(e_arr[k])
                a = max(s, lo) - lo
                b = min(e, hi - 1) - lo
                x[lane, a:b + 1] = 1
                l[lane, a:b + 1] = 1
                if lo <= s < hi:
                    h[lane, s - lo] = 1
    return x, l, h


def ln_runs(ln_window):
    """每键上 L=1 的连续段长度列表（头 = 每段起点）。"""
    runs = []
    for lane in range(NLANES):
        row = ln_window[lane]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        runs.extend((ends - starts).tolist())
    return runs


def bucket(value, cuts):
    for i, c in enumerate(cuts):
        if value < c:
            return i
    return len(cuts)


def dist_summary(values):
    """分布摘要（单位：1/32 格）。"""
    if not len(values):
        return {"n": 0}
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "均值": round(float(v.mean()), 2),
        "中位数": round(float(np.median(v)), 2),
        "p25": round(float(np.percentile(v, 25)), 2),
        "p75": round(float(np.percentile(v, 75)), 2),
        "p90": round(float(np.percentile(v, 90)), 2),
        "p95": round(float(np.percentile(v, 95)), 2),
        "p99": round(float(np.percentile(v, 99)), 2),
        "最小": int(v.min()),
        "最大": int(v.max()),
        "中位数_拍": round(float(np.median(v)) / GRID_DIV, 3),
        "p95_拍": round(float(np.percentile(v, 95)) / GRID_DIV, 3),
    }


def load_manifest(path):
    raw = open(path, "rb").read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="out/diffusion/round136-data/manifest.json")
    ap.add_argument("--out", default="out/diff-data-ln/dataset.npz")
    ap.add_argument("--report", default="out/diff-data-ln/report.json")
    ap.add_argument("--window-cells", type=int, default=WIN)
    ap.add_argument("--stride-cells", type=int, default=STRIDE)
    ap.add_argument("--min-fill", type=int, default=20,
                    help="窗口内被占用的行数下限（与 diff-data32-grouped.py 同口径）")
    ap.add_argument("--min-notes", type=int, default=64, help="整张谱最少音符数")
    ap.add_argument("--max-red-lines", type=int, default=3,
                    help="红线（uninherited=1）条数上限；超过视为变速谱，跳过并计数")
    ap.add_argument("--limit", type=int, default=0, help="每个风格包只取前 N 张（冒烟）")
    ap.add_argument("--audit-sample", type=int, default=192,
                    help="抽样重解析源谱，重建 X/L/H 验证可复现")
    args = ap.parse_args(argv)

    parser = load_parser_module()
    man, manifest_sha = load_manifest(args.manifest)
    charts = man["charts"]
    if args.limit:
        keep = collections.Counter()
        sel = []
        for c in charts:
            st = c["primary_style"]
            if keep[st] < args.limit:
                keep[st] += 1
                sel.append(c)
        charts = sel

    W = args.window_cells
    skip = collections.Counter()
    red_skip_by_split = collections.Counter()
    degenerate_total = 0
    overlap_total = 0
    adjacent_total = 0
    notes_total = taps_total = 0
    ln_len_global = array.array("i")        # 全局 LN 覆盖格数
    ln_dur_global = array.array("i")        # 全局 LN 时长（end-start）
    ln_len_by_style = {s: array.array("i") for s in STYLES}

    # ---- pass1：解析 → 事件 + 窗口起点（只留紧凑的格坐标，不留窗口）----
    records = []
    win_count = 0
    hash_ok = 0
    for idx, chart in enumerate(charts):
        path = chart["canonical_path"]
        style = chart["primary_style"]
        split = chart["split"]
        if split not in SPLIT_CODE:
            raise AssertionError(f"非法 split {split!r} 于 {path}")
        try:
            parsed = parser.parse_chart(path)
        except OSError as exc:
            skip[f"read_error:{type(exc).__name__}"] += 1
            continue
        if parsed["errors"]:
            skip["parser_errors"] += 1
            continue
        if parsed["content_sha256"] != chart["content_sha256"]:
            raise AssertionError(f"解析结果与 manifest 不一致：{path}")
        hash_ok += 1

        n_notes = len(parsed["notes"])
        if n_notes < args.min_notes:
            skip["too_few_notes"] += 1
            continue
        segments = parsed["timing_segments"]
        if not segments:
            skip["no_red_timing"] += 1
            continue
        if len(segments) != len(chart["timing_segments"]):
            raise AssertionError(f"红线条数与 manifest 不一致：{path}")
        if len(segments) > args.max_red_lines:
            skip[f"red_lines_gt_{args.max_red_lines}"] += 1
            red_skip_by_split[split] += 1
            continue

        ev = chart_events(parsed, segments)
        degenerate_total += ev.degenerate
        overlap_total += ev.overlap
        adjacent_total += ev.adjacent
        if not any(len(ev.taps[l]) or len(ev.ln_s[l]) for l in range(NLANES)):
            skip["empty_after_quantise"] += 1
            continue

        cmin, cmax, prefix = span_and_rows(ev)
        starts = []
        for start in window_starts(cmin, cmax, args.stride_cells):
            if rows_in(prefix, cmin, start, start + args.window_cells) < args.min_fill:
                skip["too_empty"] += 1
                continue
            starts.append(start)
        if not starts:
            skip["no_window"] += 1
            continue

        notes_total += n_notes
        for lane in range(NLANES):
            taps_total += len(ev.taps[lane])
            for s, e in zip(ev.ln_s[lane].tolist(), ev.ln_e[lane].tolist()):
                ln_len_global.append(e - s + 1)
                ln_dur_global.append(e - s)
                ln_len_by_style[style].append(e - s + 1)

        records.append({"chart": chart, "segments": segments, "style": style,
                        "split": split, "starts": starts, "events": ev})
        win_count += len(starts)
        if (idx + 1) % 400 == 0:
            print(f"[pass1] {idx + 1}/{len(charts)} 张，窗口 {win_count}", file=sys.stderr)

    if not records:
        raise SystemExit("[错误] 一张谱都没通过，检查 --min-fill / --min-notes / --max-red-lines")

    N = win_count
    X = np.empty((N, NLANES, W), dtype=np.uint8)
    L = np.empty((N, NLANES, W), dtype=np.uint8)
    H = np.empty((N, NLANES, W), dtype=np.uint8)
    style_arr = np.empty(N, dtype=np.uint8)
    diff_arr = np.empty(N, dtype=np.uint8)
    diffnps_arr = np.empty(N, dtype=np.uint8)
    bpm_arr = np.empty(N, dtype=np.float32)
    split_arr = np.empty(N, dtype=np.int8)
    cell_arr = np.empty(N, dtype=np.int64)
    ln_heads_arr = np.empty(N, dtype=np.int16)
    ln_cells_arr = np.empty(N, dtype=np.int16)
    chart_ids, group_ids = [], []

    # ---- pass2：铺窗口 + 统计 ----
    lens_window = array.array("i")       # 窗口内 LN 段长度（按窗裁剪）
    lens_window_by_style = {s: array.array("i") for s in STYLES}
    per_style_windows = collections.Counter()
    fill = 0
    for rec in records:
        chart, segments, style, split = rec["chart"], rec["segments"], rec["style"], rec["split"]
        cid, g = chart["chart_id"], chart["group_id"]
        b_rps = bucket(chart["statistics"]["row_rate_rps"], (8.0, 12.0, 16.0))
        b_nps = bucket(chart["statistics"]["note_head_rate_nps"], (16.0, 20.0, 24.0))
        for start in rec["starts"]:
            x, l, h = build_window(rec["events"], start, W)
            X[fill] = x
            L[fill] = l
            H[fill] = h
            style_arr[fill] = SID[style]
            diff_arr[fill] = b_rps
            diffnps_arr[fill] = b_nps
            bpm_arr[fill] = round(
                60000.0 / segment_of_cell(segments, start)["beat_length_ms"], 3)
            split_arr[fill] = SPLIT_CODE[split]
            cell_arr[fill] = start
            chart_ids.append(cid)
            group_ids.append(g)
            runs = ln_runs(l)
            ln_heads_arr[fill] = len(runs)
            ln_cells_arr[fill] = int(l.sum())
            per_style_windows[style] += 1
            for ln in runs:
                lens_window.append(ln)
                lens_window_by_style[style].append(ln)
            fill += 1
        if fill // 20000 != (fill - len(rec["starts"])) // 20000:
            print(f"[pass2] 已铺 {fill}/{N} 窗", file=sys.stderr)
    assert fill == N, f"窗口数不一致：铺了 {fill}，预留 {N}"

    chart_arr = np.array(chart_ids, dtype=f"<U{max(len(s) for s in chart_ids)}")
    group_arr = np.array(group_ids, dtype=f"<U{max(len(s) for s in group_ids)}")

    # ==================================================================
    # 硬性校验
    # ==================================================================
    # 1. LN 体 ⊆ 占用，LN 头 ⊆ LN 体
    assert not np.any(L > X), "存在 L=1 但 X=0 的格（LN 体不在占用卷里）"
    assert not np.any(H > L), "存在 H=1 但 L=0 的格（LN 头不在 LN 体里）"
    # 2. H 的每个格必须是某段 L 的起点（窗口首格也算起点；真头可能落在窗外，
    #    所以只断言 H ⊆ 「L 段起点」，不断言两者个数相等）
    pad = np.zeros((N, NLANES, 1), dtype=np.uint8)
    lpad = np.concatenate([pad, L], axis=2)          # 窗口首格=L 也算段起点
    rising = (lpad[:, :, 1:] > lpad[:, :, :-1]).astype(np.uint8)
    # 同键 LN 互相覆盖时（非法谱，实测 0.07%），两个头会并成一段 L；
    # 这类 H 格不是段起点，只计数不断言为 0。
    heads_not_rising = int((H & (1 - rising)).sum())
    assert int(rising.sum()) == int(ln_heads_arr.sum()), "L 段数统计与 ln_heads 不一致"

    # 3. 抽样重解析源谱：X/L/H 必须逐格复现
    rng = np.random.default_rng(147)
    audit_idx = rng.choice(N, size=min(args.audit_sample, N), replace=False)
    audit_ok = 0
    chart_meta = {c["chart_id"]: c for c in man["charts"]}
    for i in sorted(audit_idx.tolist()):
        cid = chart_arr[i]
        reparsed = parser.parse_chart(chart_meta[cid]["canonical_path"])
        if reparsed["errors"]:
            raise AssertionError(f"复核时源谱解析失败：{cid}")
        ev2 = chart_events(reparsed, reparsed["timing_segments"])
        x2, l2, h2 = build_window(ev2, int(cell_arr[i]), W)
        if not (np.array_equal(x2, X[i]) and np.array_equal(l2, L[i])
                and np.array_equal(h2, H[i])):
            raise AssertionError(f"窗口 {i} 无法从源谱复现（{cid} @ {cell_arr[i]}）")
        audit_ok += 1

    # 4. 来源/划分一致性（与 diff-data32-grouped.py 同一套断言）
    bad = 0
    for cid, gid, sp in zip(chart_arr.tolist(), group_arr.tolist(), split_arr.tolist()):
        meta = chart_meta.get(cid)
        if meta is None or meta["group_id"] != gid or SPLIT_CODE[meta["split"]] != int(sp):
            bad += 1
    assert bad == 0, f"{bad} 个窗口的 group/split 与 manifest 不一致"

    pairs = collections.defaultdict(set)
    for cid, sp in zip(chart_arr.tolist(), split_arr.tolist()):
        pairs[cid].add(int(sp))
    multi = {c: sorted(s) for c, s in pairs.items() if len(s) > 1}
    assert not multi, f"{len(multi)} 张谱的窗口跨了 split，例如 {list(multi.items())[:3]}"

    def sets_for(code):
        mask = split_arr == code
        return set(chart_arr[mask].tolist()), set(group_arr[mask].tolist())

    per = {name: sets_for(code) for name, code in SPLIT_CODE.items()}
    inter = {}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        inter[f"{a_}/{b_}"] = {"chart_id": len(per[a_][0] & per[b_][0]),
                               "group_id": len(per[a_][1] & per[b_][1])}
    worst = max(max(v.values()) for v in inter.values())
    assert worst == 0, f"集间交集不为 0：{inter}"

    key = list(zip(chart_arr.tolist(), cell_arr.tolist()))
    assert len(key) == len(set(key)), "存在 (chart_id, start_cell) 重复窗口"

    # 同组跨 split 的时间重叠必须为 0（把格坐标换算成真实毫秒）
    ms_lo = np.empty(N)
    ms_hi = np.empty(N)
    for i, (cid, sc) in enumerate(key):
        seg = chart_meta[cid]["timing_segments"]
        ms_lo[i] = ms_of_cell(seg, sc)
        ms_hi[i] = ms_of_cell(seg, sc + W)
    by_group = collections.defaultdict(list)
    for i, gid in enumerate(group_arr.tolist()):
        by_group[gid].append(i)
    cross_split_same_group = 0
    for gid, idxs in by_group.items():
        idxs.sort(key=lambda j: (ms_lo[j], ms_hi[j]))
        active = []
        for j in idxs:
            active = [a for a in active if ms_hi[a] > ms_lo[j]]
            for a in active:
                if chart_arr[a] != chart_arr[j] and split_arr[a] != split_arr[j]:
                    cross_split_same_group += 1
            active.append(j)
    assert cross_split_same_group == 0, \
        f"同组跨 split 有 {cross_split_same_group} 对窗口时间重叠"

    # ==================================================================
    # 统计
    # ==================================================================
    note_heads = notes_total
    ln_heads = int(len(ln_len_global))
    if taps_total + ln_heads != notes_total:
        raise AssertionError("音符人数不一致：taps + LN 头 ≠ 音符头总数")
    ln_cells_total = int(sum(ln_len_global))
    occ_cells_total = int(X.sum())
    split_masks = {SPLIT_NAME[c]: split_arr == c for c in sorted(SPLIT_NAME)}
    style_masks = {s: style_arr == SID[s] for s in STYLES}
    gcount = collections.Counter(group_arr.tolist())
    gcharts = collections.defaultdict(set)
    for cid, gid in zip(chart_arr.tolist(), group_arr.tolist()):
        gcharts[gid].add(cid)

    def per_win(arr, mask, nd=3):
        return round(float(arr[mask].mean()), nd) if int(mask.sum()) else None

    stats = {
        "数据集": args.out,
        "版本": PARSER_VERSION,
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "网格": {"cell": "1/32", "grid_division": GRID_DIV,
                 "window_cells": args.window_cells, "stride_cells": args.stride_cells,
                 "step_note": "1/32 音符 = 1 格 = 1/8 拍",
                 "max_red_lines": args.max_red_lines},
        "通道": {"X": "占用（LN 覆盖全部格，普通音符只取起点）",
                 "L": "LN 体（LN 覆盖格 = 1）",
                 "H": "LN 头（LN 起点格 = 1）"},
        "LN长度定义": "覆盖格数 = end_cell - start_cell + 1（单位 1/32 格）；LN时长分布另给 end-start",
        "窗口数": int(N),
        "形状": list(X.shape),
        "占位率": round(float(X.mean()), 4),
        "LN体格占占位格比例": round(ln_cells_total / max(1, occ_cells_total), 5),
        "每split窗口": {SPLIT_NAME[c]: int((split_arr == c).sum()) for c in sorted(SPLIT_NAME)},
        "每split占位率": {k: round(float(X[m].mean()), 4) for k, m in split_masks.items()},
        "风格分布": np.bincount(style_arr, minlength=4).tolist(),
        "难度分布_RPS": np.bincount(diff_arr, minlength=4).tolist(),
        "难度分布_NPS": np.bincount(diffnps_arr, minlength=4).tolist(),
        "每split谱数": {SPLIT_NAME[c]: len(per[SPLIT_NAME[c]][0]) for c in sorted(SPLIT_NAME)},
        "每split组数": {SPLIT_NAME[c]: len(per[SPLIT_NAME[c]][1]) for c in sorted(SPLIT_NAME)},
        "每style窗口": dict(per_style_windows),
        "用到的谱": int(len(set(chart_arr.tolist()))),
        "谱面音符头数(清单谱合计)": int(note_heads),
        "普通音符数(清单谱合计)": int(taps_total),
        "LN头数(清单谱合计)": ln_heads,
        "LN占音符比例(头/音符头)": round(ln_heads / max(1, note_heads), 5),
        "LN体格数(清单谱合计)": ln_cells_total,
        "LN长度分布_全局": dist_summary(ln_len_global),
        "LN时长分布_全局": dist_summary(ln_dur_global),
        "LN长度分布_按风格": {s: dist_summary(ln_len_by_style[s]) for s in STYLES},
        "LN长度分布_窗口裁剪": dist_summary(lens_window),
        "每窗口平均LN头数": round(float(ln_heads_arr.mean()), 3),
        "每窗口平均LN体格数": round(float(ln_cells_arr.mean()), 3),
        "每split每窗口平均LN头数": {k: per_win(ln_heads_arr, m) for k, m in split_masks.items()},
        "每split每窗口平均LN体格数": {k: per_win(ln_cells_arr, m) for k, m in split_masks.items()},
        "每style每窗口平均LN头数": {s: per_win(ln_heads_arr, m) for s, m in style_masks.items()},
        "每style每窗口平均LN体格数": {s: per_win(ln_cells_arr, m) for s, m in style_masks.items()},
        "有LN的窗口占比": round(float((ln_heads_arr > 0).mean()), 4),
        "每style有LN窗口占比": {s: round(float((ln_heads_arr[m] > 0).mean()), 4)
                                if int(m.sum()) else None for s, m in style_masks.items()},
        "每style_LN长度分布_窗口裁剪": {s: dist_summary(lens_window_by_style[s])
                                        for s in STYLES},
        "每窗口LN头数分布": {
            "0窗占比": round(float((ln_heads_arr == 0).mean()), 4),
            "p50": float(np.percentile(ln_heads_arr, 50)),
            "p90": float(np.percentile(ln_heads_arr, 90)),
            "p99": float(np.percentile(ln_heads_arr, 99)),
            "最大": int(ln_heads_arr.max()),
        },
        "bpm_中位数": round(float(np.median(bpm_arr)), 3),
        "组规模": {"组数": len(gcount),
                  "最大组窗口数": int(max(gcount.values())),
                  "最大组谱数": int(max(len(v) for v in gcharts.values())),
                  "窗口中位数": int(np.median(list(gcount.values())))},
        "量化塌陷LN数": degenerate_total,
        "同键重叠LN数(非法谱)": overlap_total,
        "同键相邻LN数(首尾相接,连成一段L)": adjacent_total,
        "LN头格但不是L段起点的格数": heads_not_rising,
        "LN头非段起点_原因": "同键相邻/重叠的多个 LN 在 L 通道里连成一段（L 只表达覆盖，不表达"
                             "「换了长条」）；每张谱的同一头会被它所在的每个窗口各数一次。"
                             "模型采样也按「L 连续段」计头，两边口径一致。",
        "重新解析hash一致谱数": hash_ok,
        "跳过": dict(skip),
        "跳过_红线超限_按split": dict(red_skip_by_split),
        "断言": {
            "Ln体是占用卷的子集": True,
            "Ln头是Ln体的子集": True,
            "Ln头都是Ln段起点": heads_not_rising == 0,
            "Ln头非段起点_计数口径": "同键相邻/重叠 LN 连成一段 + 每个头按窗口重复计数",
            "Ln头格数": int(H.sum()),
            "Ln段数(窗口内)": int(ln_heads_arr.sum()),
            "集间交集(实测)": inter,
            "all_pairwise_intersections_zero": True,
            "同一chart的窗口只在单一split": True,
            "每窗口group_split继承自chart": True,
            "重复(chart,start_cell)窗口": 0,
            "同组跨split窗口时间重叠对数": cross_split_same_group,
            "抽样重解析重建X/L/H一致窗口数": audit_ok,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out,
                        X=X, L=L, H=H,
                        style=style_arr, diff=diff_arr, diff_nps=diffnps_arr,
                        bpm=bpm_arr, chart_id=chart_arr, group_id=group_arr,
                        split=split_arr, start_cell=cell_arr,
                        ln_heads=ln_heads_arr, ln_cells=ln_cells_arr,
                        window_cells=np.int64(args.window_cells),
                        grid_division=np.int64(GRID_DIV))
    stats["文件字节"] = os.path.getsize(args.out)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)

    d = stats["LN长度分布_全局"]
    dw = stats["LN长度分布_窗口裁剪"]
    print(f"数据集：{N} 窗口  X/L/H 各 {NLANES}键 × {W}格（1/32）"
          f"  {os.path.getsize(args.out) / 1e6:.1f}MB")
    print(f"  每split窗口 {stats['每split窗口']}；用到的谱 {stats['用到的谱']}")
    print(f"  LN 头 {ln_heads}（占音符头 {100 * ln_heads / max(1, note_heads):.2f}%）"
          f"  LN 体格 {ln_cells_total}（占占位格 {100 * ln_cells_total / max(1, occ_cells_total):.2f}%）")
    print(f"  LN 长度（格，覆盖格数）：中位 {d['中位数']}  p25 {d['p25']}  p75 {d['p75']}"
          f"  p95 {d['p95']}  max {d['最大']}")
    print(f"  窗口内 LN 段长度：中位 {dw['中位数']}  p95 {dw['p95']}  max {dw['最大']}"
          f"  n={dw['n']}")
    print(f"  每窗口 LN 头 {stats['每窗口平均LN头数']}  体格 {stats['每窗口平均LN体格数']}"
          f"  有LN窗口 {100 * stats['有LN的窗口占比']:.1f}%")
    print(f"  跳过：{stats['跳过']}")
    print(f"  断言：集间交集 {inter}；同组跨split重叠 0 对；抽样重建 {audit_ok} 窗一致")
    print(f"  报告 → {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
