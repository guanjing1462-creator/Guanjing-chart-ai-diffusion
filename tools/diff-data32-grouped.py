#!/usr/bin/env python3
"""按 round136 来源清单重建 **带来源、按组划分、无泄漏** 的 1/32 音乐卷数据集。

与旧工具的关系
--------------
tools/diff-data32.py 用「相邻行间隔中位数」估计 1/16 步长再折半（已知缺陷），
且输出只有 X/style/diff，**无法追溯窗口来自哪张谱、属于哪个同源组、被分到哪个
split**，所以它切出来的任何划分都不能证明训练/验证没有同曲或重叠窗口泄漏。

本脚本改为：

1. **网格 = 真实 TimingPoints 红线**。只读 [TimingPoints] 段，只认 uninherited=1
   的行（与 tools/gj-data-manifest.py 的严格解析器完全同一份实现，直接 import），
   用连续拍坐标 beat(t) 定位：
       cell(t) = round(8 * beat(t))       # 8 格/拍 = 1/32 网格
   逐窗不再把首音符归零，而是落在该谱**固定 timing 原点**的绝对格上。
2. **分组与划分原样继承 manifest**（seed=136，175 组，train/val/test 按组切）。
   窗口只继承其 chart 的 split，脚本里不做任何窗口级随机划分。
3. 每个窗口额外记录 `chart_id` / `group_id` / `split`(int8 0/1/2) / `start_cell`
   / `bpm`（窗口起点所在红线的 BPM），并在构建完立刻做无泄漏断言。

输出
----
out/diff-data-grouped/dataset.npz
    X         [N,4,256] uint8    二值钢琴卷（1/32 格）
    style     [N] uint8          0=tech 1=jack 2=speed 3=stamina（manifest primary_style）
    diff      [N] uint8          行率 RPS 4 档（<8 / 8~12 / 12~16 / >16），与旧 diff 同口径
    diff_nps  [N] uint8          音符头率 NPS 4 档（<16 / 16~20 / 20~24 / >24），真实难度口径
    bpm       [N] float32        窗口起点红线的 BPM（变速谱=局部 BPM）
    chart_id  [N] U68            manifest 的内容寻址谱 ID（gjc_ + sha256）
    group_id  [N] U68            manifest 同源组 ID（gjs_ + sha256）
    split     [N] int8           0=train 1=val 2=test（从 chart 继承，非窗口随机）
    start_cell[N] int64          窗口起点在该谱绝对 1/32 格上的坐标（半开区间）
    window_cells / grid_division 标量（256 / 8），把「格」的口径写进文件
out/diff-data-grouped/report.json   全部统计 + 断言实测值 + 解析修正对照

用法
----
  python3 tools/diff-data32-grouped.py                    # 全量
  python3 tools/diff-data32-grouped.py --limit 30         # 每包前 30 张（冒烟）
  python3 tools/diff-data32-grouped.py --window-cells 256 --stride-cells 128
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
STRIDE = 128          # 滑窗步长（格）= 半窗，50% 重叠；旧脚本用 64「行」且相位随音符漂移

PARSER_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gj-data-manifest.py")
PARSER_VERSION = "grouped-1/32-real-red-timing-v1"


# ---------------------------------------------------------------------------
# 复用 manifest 的严格解析器（同一份实现，避免两套 .osu 语义）
# ---------------------------------------------------------------------------
def load_parser_module(path=PARSER_MODULE):
    spec = importlib.util.spec_from_file_location("gj_data_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 拍坐标 ⇄ 绝对格 ⇄ 毫秒（真实红线，支持变速）
# ---------------------------------------------------------------------------
def cell_of(segments, time_ms):
    """cell(t) = round(8 * beat(t))，beat 由红线分段决定（支持变速）。"""
    return int(round(GRID_DIV * beat_at(segments, time_ms)))


def beat_at(segments, time_ms):
    offsets = [s["offset_ms"] for s in segments]
    i = max(0, bisect.bisect_right(offsets, time_ms) - 1)
    s = segments[i]
    return s["beat_start"] + (time_ms - s["offset_ms"]) / s["beat_length_ms"]


def segment_of_cell(segments, cell):
    """窗口起点所在的红线段（取其 BPM 作为该窗口的 bpm 标签）。"""
    beat = cell / float(GRID_DIV)
    i = max(0, bisect.bisect_right([s["beat_start"] for s in segments], beat) - 1)
    return segments[i]


def ms_of_cell(segments, cell):
    """格 → 毫秒（beat_at 的逆），用于把窗口区间换算成真实音频时间。"""
    s = segment_of_cell(segments, cell)
    return s["offset_ms"] + (cell / float(GRID_DIV) - s["beat_start"]) * s["beat_length_ms"]


# ---------------------------------------------------------------------------
def build_roll(rows_dict, start_cell, window_cells):
    """把 [start_cell, start_cell+window_cells) 内的行铺成 [4, window_cells] 0/1。"""
    m = np.zeros((4, window_cells), dtype=np.uint8)
    lo, hi = start_cell, start_cell + window_cells
    for cell in range(lo, hi):
        mask = rows_dict.get(cell)
        if mask is None:
            continue
        col = cell - lo
        if mask & 1:
            m[0, col] = 1
        if mask & 2:
            m[1, col] = 1
        if mask & 4:
            m[2, col] = 1
        if mask & 8:
            m[3, col] = 1
    return m


def gap_p132(m):
    """p132（口径同 tools/eval-generated.py）：相邻占用行间隔 = 1/32 格 的比例。

    eval-generated 在**毫秒**上算 units = gap_ms / step16，统计 0.35<=u<0.75；
    本数据集已是 1/32 格，1/16 步长 = 2 格，因此等价的精确判据是「间隔 == 1 格」。
    """
    occ = np.where(m.sum(axis=0) > 0)[0]
    if len(occ) < 2:
        return 0.0, 0.0
    g = np.diff(occ)
    return float((g == 1).mean()), float(len(occ))


def p132_ms(onsets_ms, step16_ms):
    """eval-generated 的原始毫秒口径（0.35<=u<0.75），返回 (命中数, 间隔数)。

    与格口径的差别正是「量化」本身：1/48 三连音等非 1/32 内容会被吸到 1 格间隔上，
    所以两个口径必须分开报，不能互相冒充。
    """
    if len(onsets_ms) < 2:
        return 0, 0
    g = np.diff(np.asarray(onsets_ms, dtype=float))
    u = g / step16_ms
    return int(((u >= 0.35) & (u < 0.75)).sum()), int(len(u))


def bucket(value, cuts):
    for i, c in enumerate(cuts):
        if value < c:
            return i
    return len(cuts)


# ---------------------------------------------------------------------------
def load_manifest(path):
    raw = open(path, "rb").read()
    man = json.loads(raw.decode("utf-8"))
    return man, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="out/diffusion/round136-data/manifest.json")
    ap.add_argument("--out", default="out/diff-data-grouped/dataset.npz")
    ap.add_argument("--report", default="out/diff-data-grouped/report.json")
    ap.add_argument("--window-cells", type=int, default=WIN)
    ap.add_argument("--stride-cells", type=int, default=STRIDE)
    ap.add_argument("--min-fill", type=int, default=20,
                    help="窗口内被占用的格数下限（与旧脚本 20 可比）")
    ap.add_argument("--min-notes", type=int, default=64, help="整张谱最少音符数")
    ap.add_argument("--limit", type=int, default=0, help="每个风格包只取前 N 张（冒烟）")
    ap.add_argument("--audit-sample", type=int, default=256,
                    help="随机抽样重建窗口，验证 X 与来源可复现")
    args = ap.parse_args(argv)

    parser = load_parser_module()
    man, manifest_sha = load_manifest(args.manifest)

    pack_of = {}
    for style in STYLES:
        pack_of[style] = sorted(
            os.path.join("out/packs", style, f)
            for f in os.listdir(os.path.join("out", "packs", style))
            if f.endswith(".osu"))

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

    X_list, style_list, diff_list, diffnps_list, bpm_list = [], [], [], [], []
    chart_list, group_list, split_list, cell_list = [], [], [], []

    skip = collections.Counter()
    per_style = collections.Counter()
    per_split = collections.Counter()
    var_tempo_windows = 0
    var_tempo_charts = 0
    neg_cell_charts = 0
    quant_resid = []
    gap1_num = gap_den = 0
    odd_num = occ_den = 0
    step_err = []          # 旧「中位间隔」步长估计 vs 真实 1/32 格长的相对误差
    p132_ms_num = p132_ms_den = 0    # 毫秒口径 p132（eval-generated 原口径，池化）
    p132_ms_per_chart = []           # 毫秒口径 p132（逐谱，取中位数）
    off_grid_charts = 0              # 非 1/32 格点内容占比 >10% 的谱（三连音/1/48 等）
    off_grid_notes = 0
    chart_windows = collections.Counter()
    hash_ok = 0
    n_windows_per_chart = []
    n_notes_per_chart = []
    collision_cells = 0
    resid_sample = array.array("f")   # 抽样量化残差（每 8 个音符取 1 个）
    resid_cnt = near_tie = 0
    resid_sum = 0.0

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
        # 来源自证：重新解析出的玩法哈希必须等于 manifest 的 chart_id 后缀
        if parsed["content_sha256"] != chart["content_sha256"]:
            raise AssertionError(f"解析结果与 manifest 不一致：{path}")
        hash_ok += 1

        notes = parsed["notes"]
        if len(notes) < args.min_notes:
            skip["too_few_notes"] += 1
            continue

        segments = chart["timing_segments"]
        if not segments:
            skip["no_red_timing"] += 1
            continue
        variable = bool(chart["tempo_changes"])
        if variable:
            var_tempo_charts += 1
        beat0 = segments[0]["beat_length_ms"]

        # 旧缺陷量化：相邻行间隔中位数 / 2 与真实 1/32 格长的相对误差
        onset_ms = sorted({n[1] for n in notes})
        g = np.diff(np.array(onset_ms, dtype=float))
        g = g[(g > 40) & (g < 400)]
        if len(g) >= 20:
            step26 = float(np.median(g)) / 2.0
            step_err.append(abs(step26 - beat0 / 8.0) / (beat0 / 8.0))
        nm, dn = p132_ms(onset_ms, beat0 / 4.0)
        p132_ms_num += nm
        p132_ms_den += dn
        if dn:
            p132_ms_per_chart.append(nm / dn)

        # 行：cell -> 4 键位掩码（同行多键合并）
        rows_dict = {}
        for lane, start, _end in notes:
            raw = GRID_DIV * beat_at(segments, start)
            c = int(round(raw))
            r = abs(raw - c)
            resid_cnt += 1
            resid_sum += r
            if abs(r - 0.5) < 0.02:
                near_tie += 1
            if resid_cnt % 8 == 0:
                resid_sample.append(r)
            if c in rows_dict and not (rows_dict[c] >> lane) & 1:
                collision_cells += 1
            rows_dict[c] = rows_dict.get(c, 0) | (1 << lane)
        chart_notes = sum(1 for _ in notes)
        chart_off = sum(1 for lane, start, _e in notes
                        if abs(GRID_DIV * beat_at(segments, start)
                               - round(GRID_DIV * beat_at(segments, start))) > 0.25)
        off_grid_notes += chart_off
        if chart_notes and chart_off / chart_notes > 0.10:
            off_grid_charts += 1
        cells_sorted = sorted(rows_dict)
        if len(cells_sorted) < args.min_notes // 16 or not cells_sorted:
            skip["empty_after_quantise"] += 1
            continue
        min_cell, max_cell = cells_sorted[0], cells_sorted[-1]
        if min_cell < 0:
            neg_cell_charts += 1

        del cells_sorted

        # 绝对格滑窗：起点对齐到全局网格（k*stride），相位恒定，不逐窗归零
        lo = int(math.floor(min_cell / args.stride_cells))
        hi = int(math.floor(max_cell / args.stride_cells))
        n_chart_windows = 0
        for k in range(lo, hi + 1):
            start = k * args.stride_cells
            m = build_roll(rows_dict, start, args.window_cells)
            filled = int(np.count_nonzero(m.sum(axis=0)))
            if filled < args.min_fill:
                skip["too_empty"] += 1
                continue
            X_list.append(m)
            style_list.append(SID[style])
            diff_list.append(bucket(chart["statistics"]["row_rate_rps"], (8.0, 12.0, 16.0)))
            diffnps_list.append(bucket(chart["statistics"]["note_head_rate_nps"],
                                       (16.0, 20.0, 24.0)))
            bpm_list.append(round(60000.0 / segment_of_cell(segments, start)["beat_length_ms"], 3))
            chart_list.append(chart["chart_id"])
            group_list.append(chart["group_id"])
            split_list.append(SPLIT_CODE[split])
            cell_list.append(start)
            n_chart_windows += 1
            if variable:
                var_tempo_windows += 1
            # 统计
            frac, n_occ = gap_p132(m)
            if n_occ:
                gap1_num += frac * (n_occ - 1)
                gap_den += (n_occ - 1)
            occ = np.where(m.sum(axis=0) > 0)[0]
            odd_num += int((occ % 2 == 1).sum())
            occ_den += int(len(occ))
        per_style[style] += n_chart_windows
        per_split[split] += n_chart_windows
        chart_windows[chart["chart_id"]] = n_chart_windows
        n_windows_per_chart.append(n_chart_windows)
        n_notes_per_chart.append(len(notes))
        if (idx + 1) % 200 == 0:
            print(f"[build] {idx + 1}/{len(charts)} 张，窗口 {len(X_list)}", file=sys.stderr)

    if not X_list:
        raise SystemExit("[错误] 一个窗口都没切出来，检查 --min-fill / --min-notes")

    X = np.stack(X_list).astype(np.uint8)
    style_arr = np.array(style_list, dtype=np.uint8)
    diff_arr = np.array(diff_list, dtype=np.uint8)
    diffnps_arr = np.array(diffnps_list, dtype=np.uint8)
    bpm_arr = np.array(bpm_list, dtype=np.float32)
    chart_arr = np.array(chart_list, dtype=f"<U{max(len(s) for s in chart_list)}")
    group_arr = np.array(group_list, dtype=f"<U{max(len(s) for s in group_list)}")
    split_arr = np.array(split_list, dtype=np.int8)
    cell_arr = np.array(cell_list, dtype=np.int64)
    del X_list

    # ------------------------------------------------------------------
    # 硬性校验 1：来源一致性（每个窗口的 group/split 必须等于其 chart 的）
    # ------------------------------------------------------------------
    chart_meta = {c["chart_id"]: c for c in man["charts"]}
    bad = 0
    for cid, gid, sp in zip(chart_arr.tolist(), group_arr.tolist(), split_arr.tolist()):
        meta = chart_meta.get(cid)
        if meta is None or meta["group_id"] != gid or SPLIT_CODE[meta["split"]] != int(sp):
            bad += 1
    assert bad == 0, f"{bad} 个窗口的 group/split 与 manifest 不一致"

    # ------------------------------------------------------------------
    # 硬性校验 2：同一 chart 的所有窗口只出现在一个 split
    # ------------------------------------------------------------------
    pairs = collections.defaultdict(set)
    for cid, sp in zip(chart_arr.tolist(), split_arr.tolist()):
        pairs[cid].add(int(sp))
    multi = {c: sorted(s) for c, s in pairs.items() if len(s) > 1}
    assert not multi, f"{len(multi)} 张谱的窗口跨了 split，例如 {list(multi.items())[:3]}"

    # ------------------------------------------------------------------
    # 硬性校验 3：train/val/test 在 group_id 与 chart_id 上交集全为 0
    # ------------------------------------------------------------------
    def sets_for(code):
        mask = split_arr == code
        return set(chart_arr[mask].tolist()), set(group_arr[mask].tolist())

    per = {name: sets_for(code) for name, code in SPLIT_CODE.items()}
    inter = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter[f"{a}/{b}"] = {
            "chart_id": len(per[a][0] & per[b][0]),
            "group_id": len(per[a][1] & per[b][1]),
            "windows_from_shared_chart": int(np.isin(chart_arr, list(per[a][0] & per[b][0])).sum()),
        }
    worst = max(max(v["chart_id"], v["group_id"]) for v in inter.values())
    assert worst == 0, f"集间交集不为 0：{inter}"

    # ------------------------------------------------------------------
    # 硬性校验 4：重复窗口（同一 chart 同一 start_cell）为 0
    # ------------------------------------------------------------------
    key = list(zip(chart_arr.tolist(), cell_arr.tolist()))
    dup = len(key) - len(set(key))
    assert dup == 0, f"{dup} 个 (chart_id, start_cell) 重复窗口"

    # ------------------------------------------------------------------
    # 窗口重叠审计（用 start_cell 与窗口长精确判断）
    #   · 同组（同一批谱共用音源候选）→ 换算成真实毫秒区间，跨 split 必须 0；
    #   · 跨组 → 各谱 timing 原点独立，格坐标不可当作音频时间，单独报数。
    # ------------------------------------------------------------------
    ms_lo = np.empty(len(cell_arr))
    ms_hi = np.empty(len(cell_arr))
    for i, (cid, sc) in enumerate(key):
        seg = chart_meta[cid]["timing_segments"]
        ms_lo[i] = ms_of_cell(seg, sc)
        ms_hi[i] = ms_of_cell(seg, sc + args.window_cells)

    order = np.argsort(ms_lo, kind="stable")
    cross_split_same_group = 0
    same_split_same_group = 0
    same_chart_pairs = 0
    # 同组内扫描（组内元素通常少，直接按组扫）
    by_group = collections.defaultdict(list)
    for i, gid in enumerate(group_arr.tolist()):
        by_group[gid].append(i)
    overlapping_groups = []
    for gid, idxs in by_group.items():
        idxs.sort(key=lambda j: (ms_lo[j], ms_hi[j]))
        active = []
        for j in idxs:
            active = [a for a in active if ms_hi[a] > ms_lo[j]]
            for a in active:
                if chart_arr[a] == chart_arr[j]:
                    same_chart_pairs += 1
                elif split_arr[a] != split_arr[j]:
                    cross_split_same_group += 1
                    overlapping_groups.append((gid, int(a), int(j)))
                else:
                    same_split_same_group += 1
            active.append(j)

    # 跨组的格坐标相交（结构性计数：不同谱的格原点独立，不是时间重叠）
    cell_sorted = np.sort(cell_arr)
    total_cell_pairs = int(np.searchsorted(cell_sorted, cell_sorted + args.window_cells,
                                           side="left").sum() - len(cell_sorted))
    same_chart_cell_pairs = 0
    by_chart_cells = collections.defaultdict(list)
    for cid, sc in key:
        by_chart_cells[cid].append(sc)
    for cid, arr in by_chart_cells.items():
        arr = np.array(sorted(arr), dtype=np.int64)
        same_chart_cell_pairs += int(np.searchsorted(arr, arr + args.window_cells,
                                                     side="left").sum() - len(arr))
    same_group_cell_pairs = 0
    for gid, idxs in by_group.items():
        g = np.sort(cell_arr[idxs])
        same_group_cell_pairs += int(np.searchsorted(g, g + args.window_cells,
                                                     side="left").sum() - len(g))
    cross_group_cell_pairs = max(0, total_cell_pairs - same_group_cell_pairs)

    assert cross_split_same_group == 0, (
        f"同一组内仍有 {cross_split_same_group} 对窗口跨 split 重叠：{overlapping_groups[:5]}")

    # ------------------------------------------------------------------
    # 抽样重建：X 必须能由来源谱 + start_cell 精确复现
    # ------------------------------------------------------------------
    audit = []
    rng = np.random.default_rng(137)
    sample = rng.choice(len(cell_arr), size=min(args.audit_sample, len(cell_arr)),
                        replace=False)
    for i in sorted(sample.tolist()):
        cid = chart_arr[i]
        parsed = parser.parse_chart(chart_meta[cid]["canonical_path"])
        rows_dict = {}
        for lane, start, _end in parsed["notes"]:
            c = cell_of(chart_meta[cid]["timing_segments"], start)
            rows_dict[c] = rows_dict.get(c, 0) | (1 << lane)
        rebuilt = build_roll(rows_dict, int(cell_arr[i]), args.window_cells)
        if not np.array_equal(rebuilt, X[i]):
            raise AssertionError(f"窗口 {i} 无法从来源复现（{cid} @ {cell_arr[i]}）")
        audit.append({"window": int(i), "chart_id": cid, "start_cell": int(cell_arr[i]),
                      "reproduced": True})

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    occ = float(X.mean())
    p132_grid = gap1_num / max(1, gap_den)
    odd_ratio = odd_num / max(1, occ_den)
    # 分桶统计（直接读最终数组，避免与循环内累加口径不一致）
    win_num = np.zeros(len(cell_arr))
    win_den = np.zeros(len(cell_arr))
    for i in range(len(cell_arr)):
        o = np.where(X[i].sum(axis=0) > 0)[0]
        if len(o) > 1:
            win_num[i] = int((np.diff(o) == 1).sum())
            win_den[i] = len(o) - 1
    def p132_of(mask):
        return round(float(win_num[mask].sum() / max(1, win_den[mask].sum())), 4)
    split_masks = {SPLIT_NAME[c]: split_arr == c for c in sorted(SPLIT_NAME)}
    gcount = collections.Counter(group_arr.tolist())
    gcharts = collections.defaultdict(set)
    for cid, gid in zip(chart_arr.tolist(), group_arr.tolist()):
        gcharts[gid].add(cid)

    stats = {
        "数据集": args.out,
        "版本": PARSER_VERSION,
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "parser": parser.__dict__.get("__name__", "gj-data-manifest"),
        "网格": {"cell": "1/32", "grid_division": GRID_DIV,
                 "window_cells": args.window_cells, "stride_cells": args.stride_cells,
                 "step_note": "1/32 音符 = 1 格 = 1/8 拍"},
        "窗口数": int(X.shape[0]),
        "形状": list(X.shape),
        "占位率": round(occ, 4),
        "p132": round(p132_grid, 4),
        "p132_口径": "相邻占用行间隔 == 1 格（= 1/32）的比例；等价于 eval-generated 的 0.35<=gap/step16<0.75",
        "p132_毫秒口径_池化": round(p132_ms_num / max(1, p132_ms_den), 4),
        "p132_毫秒口径_逐谱中位数": round(float(np.median(p132_ms_per_chart)), 4) if p132_ms_per_chart else None,
        "p132_毫秒口径_间隔数": p132_ms_den,
        "非格点内容": {
            "口径": "|8*beat-round(8*beat)| > 0.25 的音符（1/48 三连音、1/64 等非 1/32 内容）",
            "非格点音符占比": round(off_grid_notes / max(1, resid_cnt), 4),
            "非格点占比>10%的谱数": off_grid_charts,
        },
        "半格内容占占用格比例": round(odd_ratio, 4),
        "间隔直方图_单位格": None,
        "风格分布": np.bincount(style_arr, minlength=4).tolist(),
        "难度分布_RPS": np.bincount(diff_arr, minlength=4).tolist(),
        "难度分布_NPS": np.bincount(diffnps_arr, minlength=4).tolist(),
        "每split窗口": {SPLIT_NAME[c]: int((split_arr == c).sum()) for c in sorted(SPLIT_NAME)},
        "每split占位率": {k: round(float(X[m].mean()), 4) for k, m in split_masks.items()},
        "每split_p132": {k: p132_of(m) for k, m in split_masks.items()},
        "风格xsplit窗口": {st: {SPLIT_NAME[c]: int(((style_arr == SID[st]) & (split_arr == c)).sum())
                              for c in sorted(SPLIT_NAME)} for st in STYLES},
        "组规模": {"组数": len(gcount),
                  "最大组窗口数": int(max(gcount.values())),
                  "最大组谱数": int(max(len(v) for v in gcharts.values())),
                  "窗口中位数": int(np.median(list(gcount.values())))},
        "每split谱数": {SPLIT_NAME[c]: len(per[SPLIT_NAME[c]][0]) for c in sorted(SPLIT_NAME)},
        "每split组数": {SPLIT_NAME[c]: len(per[SPLIT_NAME[c]][1]) for c in sorted(SPLIT_NAME)},
        "每style窗口": dict(per_style),
        "用到的谱": int(len(chart_windows)),
        "每谱窗口数_均值": round(float(np.mean(n_windows_per_chart)), 2),
        "每谱音符数_均值": round(float(np.mean(n_notes_per_chart)), 1),
        "bpm_中位数": round(float(np.median(bpm_arr)), 3),
        "变速谱_谱数": var_tempo_charts,
        "变速谱_窗口数": var_tempo_windows,
        "负格谱数(音符早于首红线)": neg_cell_charts,
        "量化_同行同格合并次数": collision_cells,
        "量化残差": {
            "口径": "|8*beat(t) - round(8*beat(t))|，单位=1/32 格；抽样每 8 音符取 1",
            "残差_中位数": round(float(np.median(resid_sample)), 4) if len(resid_sample) else None,
            "残差_均值": round(resid_sum / max(1, resid_cnt), 4),
            "残差_p90": round(float(np.percentile(resid_sample, 90)), 4) if len(resid_sample) else None,
            "残差<0.05占比": round(float((np.array(resid_sample) < 0.05).mean()), 4) if len(resid_sample) else None,
            "残差<0.10占比": round(float((np.array(resid_sample) < 0.10).mean()), 4) if len(resid_sample) else None,
            "近半格(.5±0.02)占比": round(near_tie / max(1, resid_cnt), 5),
        },
        "重新解析hash一致谱数": hash_ok,
        "跳过": dict(skip),
        "旧步长估计相对误差_中位数": round(float(np.median(step_err)), 4) if step_err else None,
        "旧步长估计相对误差_p90": round(float(np.percentile(step_err, 90)), 4) if step_err else None,
        "断言": {
            "集间交集(实测)": inter,
            "all_pairwise_intersections_zero": True,
            "同一chart的窗口只在单一split": True,
            "每窗口group_split继承自chart": True,
            "重复(chart,start_cell)窗口": 0,
            "同组跨split窗口时间重叠对数": cross_split_same_group,
            "同组同split窗口时间重叠对数": same_split_same_group,
            "同chart相邻窗口重叠对数(设计内,50%)": same_chart_pairs,
            "跨组格坐标相交对数(非时间重叠)": cross_group_cell_pairs,
            "窗口可由来源谱精确复现(抽样)": len(audit),
        },
        "重叠审计口径": {
            "跨split": "同组窗口换算成真实毫秒区间后必须不相交；实测 0 对",
            "同split": "同组同split窗口按设计重叠（步长 128 / 窗长 256），不影响集间泄漏",
            "跨组": "不同组各谱 timing 原点独立，格坐标相交不代表同一首歌的时间重叠；"
                    "音频文件缺失（audio_file_unavailable=2601），无法证明不同组不是同一首歌",
        },
    }
    # 间隔直方图（单位 = 1/32 格）
    hist = collections.Counter()
    for i in range(len(X) if len(X) < 4000 else 4000):
        occ_i = np.where(X[i].sum(axis=0) > 0)[0]
        if len(occ_i) < 2:
            continue
        for u in np.diff(occ_i):
            hist[int(u)] += 1
    stats["间隔直方图_单位格"] = {str(k): round(v / max(1, sum(hist.values())), 4)
                             for k, v in sorted(hist.items())[:10]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, style=style_arr, diff=diff_arr,
                        diff_nps=diffnps_arr, bpm=bpm_arr, chart_id=chart_arr,
                        group_id=group_arr, split=split_arr, start_cell=cell_arr,
                        window_cells=np.int64(args.window_cells),
                        grid_division=np.int64(GRID_DIV))
    stats["文件字节"] = os.path.getsize(args.out)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)

    print(f"数据集：{X.shape[0]} 窗口  {X.shape[1]}键 × {X.shape[2]}行（1/32 格）"
          f"  {os.path.getsize(args.out) / 1e6:.1f}MB")
    print(f"  每split窗口 {stats['每split窗口']}")
    print(f"  占位率 {occ:.4f}  p132 {p132_grid:.4f}  半格内容 {odd_ratio:.4f}")
    print(f"  难度RPS {stats['难度分布_RPS']}  难度NPS {stats['难度分布_NPS']}")
    print(f"  断言：集间交集 {inter}；同组跨split重叠 {cross_split_same_group} 对；"
          f"抽样重建 {len(audit)} 窗全部一致")
    print(f"  报告 → {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
