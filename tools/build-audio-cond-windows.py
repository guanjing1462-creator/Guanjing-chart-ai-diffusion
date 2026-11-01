#!/usr/bin/env python3
"""为「有音频的歌」配 mania 谱面，切 1/32 网格窗口 + 音频条件切片（round143）。

做三件事：

1. **验证**离线下载的 osu set（`_dl/ex/<SetID>/` 里解出来的 .osu + 音频）是否真的对应
   本地那 6 首歌：标题/艺术家人工确认（表里写死）、音频时长差、BPM 差。
   BPM 判据先按严格口径（相对差 ≤2%）；若只在整倍/半倍/1.5 倍这种 **tempo 级别**上
   不一致、且音频时长差 ≤3s，则记为 tier B（tempo-multiple），原因与倍数都写进报告。
2. **切窗口**：网格 = 谱面真实红线 `[TimingPoints]` 里 uninherited=1 的 beatLength
   （1 拍 8 格 = 1/32；变速谱按红线分段做逐段线性拍坐标，分段计数）。
   窗口 256 格、stride 128，最少 20 个占位格。
3. **切条件**：把 `out/multisong/<歌>-cond.npz` 的 `feat [T,4]` 按**绝对时间**映射到每个
   1/32 格：k = floor((t_cell - cond_offset) / cond_step)，一个 1/16 格覆盖它时间范围内的
   所有 1/32 格。这样条件与谱面共享「时间」这个唯一坐标，与两边 BPM 是否成倍数无关。

输出 `out/diff-data-audio/dataset.npz`（X/C/song/chart_id/split/start_cell + 溯源键）
和 `report.json`（逐 set 验证表 + 数据集统计）。

用法：
  python3 tools/build-audio-cond-windows.py --dry-verify      # 只做验证表
  python3 tools/build-audio-cond-windows.py                   # 验证 + 建库
"""
import argparse
import bisect
import collections
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np

GRID_DIV = 8                       # 每拍 8 格 → 1/32
WIN = 256
STRIDE = 128
MIN_FILL = 20
MIN_NOTES = 64
PARSER_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gj-data-manifest.py")

SONGS = [
    ("Heart_Fraction", "Heart Fraction"),
    ("seatrus___purifikation", "seatrus - purifikation"),
    ("Plum_FallenEra", "Plum - FallenEra"),
    ("REDALiCE_x_t_pazolite___Xterfusion", "REDALiCE x t+pazolite - Xterfusion"),
    ("_______________Flash_Me_Back", "かめりあ - Flash Me Back"),
    ("Frums_SuperMix", "Frums - SuperMix"),
]
SONG_ID = {k: i for i, (k, _n) in enumerate(SONGS)}

# SetID -> (song_key, 人工确认的标题/艺术家匹配说明)
SETS = {
    2614015: ("REDALiCE_x_t_pazolite___Xterfusion", "artist/title 完全一致"),
    2628774: ("REDALiCE_x_t_pazolite___Xterfusion", "artist/title 完全一致（写为 REDALiCEt+pazolite）"),
    1835578: ("_______________Flash_Me_Back", "artist/title 完全一致（ranked）"),
    2538176: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    2625633: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    2187330: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    1828871: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    1826508: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    1831601: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    1832304: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    2503929: ("_______________Flash_Me_Back", "artist/title 完全一致"),
    # 以下 set 搜索命中但下载端点反复 503（镜像侧缺缓存）：无本地文件 → 无法验证 → 拒绝
    2630531: ("Plum_FallenEra", "Plum - FallenEra (2022 Remake)，mania=4，BPM 180"),
    2501284: ("REDALiCE_x_t_pazolite___Xterfusion", "title/artist 一致，BPM 128"),
    1955928: ("_______________Flash_Me_Back", "title/artist 一致，BPM 175"),
    1856132: ("_______________Flash_Me_Back", "title/artist 一致，BPM 175"),
    1836984: ("_______________Flash_Me_Back", "title/artist 一致，BPM 175"),
}

# tempo 级别允许的倍数（BPM 检测器落在不同拍级时的常见关系）
TEMPO_MULTS = [1.0, 2.0, 0.5, 1.5, 2.0 / 3.0, 4.0 / 3.0, 3.0 / 4.0]
BPM_TOL = 0.02
DUR_TOL = 3.0


# ---------------------------------------------------------------------------
def load_parser(path=PARSER_MODULE):
    spec = importlib.util.spec_from_file_location("gj_data_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ffprobe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败 {path}: {out.stderr.strip()[:200]}")
    return float(out.stdout.strip())


# ---- 拍坐标 ⇄ 绝对格 ⇄ 毫秒（与 tools/diff-data32-grouped.py 同一套口径）----
def beat_at(segments, time_ms):
    offsets = [s["offset_ms"] for s in segments]
    i = max(0, bisect.bisect_right(offsets, time_ms) - 1)
    s = segments[i]
    return s["beat_start"] + (time_ms - s["offset_ms"]) / s["beat_length_ms"]


def ms_of_cell(segments, cell):
    beat = cell / float(GRID_DIV)
    i = max(0, bisect.bisect_right([s["beat_start"] for s in segments], beat) - 1)
    s = segments[i]
    return s["offset_ms"] + (beat - s["beat_start"]) * s["beat_length_ms"]


def dominant_bpm(segments, end_ms):
    """按时间覆盖最长的红线取 BPM（变速谱 = 主段 BPM），并返回各段覆盖。"""
    spans = []
    for i, s in enumerate(segments):
        lo = s["offset_ms"]
        hi = segments[i + 1]["offset_ms"] if i + 1 < len(segments) else max(end_ms, lo + 1.0)
        hi = min(hi, max(end_ms, lo + 1.0))
        spans.append((max(0.0, hi - lo), 60000.0 / s["beat_length_ms"]))
    spans.sort(reverse=True)
    return spans[0][1], spans


def build_roll(rows, start, window_cells, lanes=4):
    m = np.zeros((lanes, window_cells), dtype=np.uint8)
    lo, hi = start, start + window_cells
    for cell in range(max(lo, min(rows)), min(hi, max(rows) + 1)):
        mask = rows.get(cell)
        if mask is None:
            continue
        col = cell - lo
        for lane in range(lanes):
            if mask & (1 << lane):
                m[lane, col] = 1
    return m


# ---------------------------------------------------------------------------
def verify_sets(osz_root, fx_root, parser, log=print):
    """逐个 SetID 校验：解析每张 .osu，量音频时长，算 BPM 差，给出 tier/拒绝原因。"""
    rows = []
    for set_id in sorted(SETS):
        song_key, match_note = SETS[set_id]
        fx = json.load(open(os.path.join(fx_root, f"{song_key}-fx.json")))
        exp_bpm = float(fx["bpm"])
        exp_dur = float(fx["durationSec"])
        d = os.path.join(osz_root, f"ex/{set_id}")
        if not os.path.isdir(d):
            rows.append({"set_id": set_id, "song": song_key, "status": "missing_dir"})
            continue
        osus = sorted(f for f in os.listdir(d) if f.lower().endswith(".osu"))
        audios = [f for f in os.listdir(d) if f.lower().endswith((".mp3", ".ogg", ".wav"))]
        dur = None
        audio_name = None
        if audios:
            audio_name = sorted(audios)[0]
            try:
                dur = ffprobe_duration(os.path.join(d, audio_name))
            except Exception as exc:            # noqa: BLE001
                log(f"  [警告] ffprobe 失败 {set_id}: {exc}")
        for f in osus:
            p = os.path.join(d, f)
            try:
                parsed = parser.parse_chart(p)
            except Exception as exc:            # noqa: BLE001
                rows.append({"set_id": set_id, "song": song_key, "osu": f,
                             "status": "parse_error", "why": str(exc)[:120]})
                continue
            meta = parsed["metadata"]
            rec = {
                "set_id": set_id, "song": song_key, "song_show": dict(SONGS)[song_key],
                "osu": f, "mode": parsed["mode"], "keys": parsed["keys"],
                "osu_artist": meta.get("artist", ""), "osu_title": meta.get("title", ""),
                "creator": meta.get("creator", ""), "match_note": match_note,
                "audio_file": audio_name, "audio_dur_s": None if dur is None else round(dur, 3),
                "local_dur_s": round(exp_dur, 3), "dur_diff_s": None,
                "local_bpm": round(exp_bpm, 3), "bpm_chart": None,
                "bpm_rel_diff": None, "tempo_mult": 1.0, "tier": None, "reject": [],
                "notes": len(parsed["notes"]), "segments": len(parsed["timing_segments"]),
                "variable_tempo": False,
                "parse_errors": parsed["errors"][:6],
            }
            if parsed["errors"]:
                rec["reject"].append("parser_" + ",".join(parsed["errors"][:3]))
            if parsed["mode"] != 3:
                rec["reject"].append(f"mode={parsed['mode']}（非 mania）")
            if parsed["keys"] != 4:
                rec["reject"].append(f"keys={parsed['keys']}（非 4K，X 是 [N,4,256]）")
            if parsed["timing_segments"] and parsed["notes"]:
                end_ms = max(n[2] for n in parsed["notes"])
                bpm_chart, spans = dominant_bpm(parsed["timing_segments"], end_ms)
                rec["bpm_chart"] = round(bpm_chart, 3)
                bpm_set = sorted({round(60000.0 / s["beat_length_ms"], 2)
                                  for s in parsed["timing_segments"]})
                rec["bpm_set"] = bpm_set[:8]
                rec["n_distinct_bpm"] = len(bpm_set)
                rec["bpm_main_share"] = round(spans[0][0] / max(1e-9, sum(s for s, _ in spans)), 3)
                rel = abs(bpm_chart - exp_bpm) / bpm_chart
                rec["bpm_rel_diff"] = round(rel, 5)
                if rel > BPM_TOL:
                    best = min(TEMPO_MULTS,
                               key=lambda m: abs(bpm_chart - exp_bpm * m) / bpm_chart)
                    rel_m = abs(bpm_chart - exp_bpm * best) / bpm_chart
                    rec["tempo_mult"] = best
                    rec["bpm_rel_diff_tempo_norm"] = round(rel_m, 5)
                    rec["bpm_tempo_ok"] = rel_m <= BPM_TOL
                else:
                    rec["bpm_rel_diff_tempo_norm"] = round(rel, 5)
                    rec["bpm_tempo_ok"] = True
            if dur is not None:
                rec["dur_diff_s"] = round(abs(dur - exp_dur), 3)
                if rec["dur_diff_s"] > DUR_TOL:
                    rec["reject"].append(f"音频时长差 {rec['dur_diff_s']:.2f}s > {DUR_TOL}s")
            else:
                rec["reject"].append("无音频文件，无法验证时长")
            if rec.get("bpm_tempo_ok") is False:
                rec["reject"].append(
                    f"BPM 差 {rec['bpm_rel_diff']*100:.2f}% > {BPM_TOL*100:.0f}%"
                    f"（tempo 倍数 {rec['tempo_mult']} 也不符）")
            if rec.get("n_distinct_bpm"):
                rec["variable_tempo"] = rec["n_distinct_bpm"] > 1
            if not rec["reject"]:
                rec["tier"] = "A" if rec["bpm_rel_diff"] <= BPM_TOL else "B"
                rec["status"] = "accept"
            else:
                rec["status"] = "reject"
            rows.append(rec)
            log(f"  {set_id} {f[:44]:44s} mode={rec['mode']} keys={rec['keys']} "
                f"dur={rec['audio_dur_s']} d={rec['dur_diff_s']} bpm={rec['bpm_chart']}"
                f" rel={rec['bpm_rel_diff']} tier={rec['tier']} {rec['reject']}")
    return rows


# ---------------------------------------------------------------------------
def cond_lookup(feat, offset_ms, step_ms, t_ms):
    k = int(math.floor((t_ms - offset_ms) / step_ms + 1e-9))
    if 0 <= k < feat.shape[0]:
        return k
    return -1


def p132_of(m):
    occ = np.where(m.sum(axis=0) > 0)[0]
    if len(occ) < 2:
        return 0, 0
    g = np.diff(occ)
    return int((g == 1).sum()), int(len(g))


def build_dataset(verify_rows, osz_root, cond_root, out_path, report_path,
                  window_cells=WIN, stride_cells=STRIDE, min_fill=MIN_FILL,
                  parser=None):
    cond_meta = json.load(open(os.path.join(cond_root, "cond-meta.json")))["songs"]
    fx = {k: json.load(open(os.path.join(cond_root, f"{k}-fx.json"))) for k, _ in SONGS}

    accepted = [r for r in verify_rows if r.get("status") == "accept"]
    # 同一 gameplay 内容的两张谱（例如 [sv] / [NO SV] 只差视觉位移）只能算一张，
    # 否则同一谱会同时落进 train 和 val（泄漏）且窗口重复计数。
    seen_sha = {}
    charts = []
    for r in accepted:
        song_key = r["song"]
        cm = cond_meta[song_key]
        feat = np.load(os.path.join(cond_root, f"{song_key}-cond.npz"))["feat"].astype(np.float32)
        p = os.path.join(osz_root, f"ex/{r['set_id']}", r["osu"])
        parsed = parser.parse_chart(p)
        segments = parsed["timing_segments"]
        notes = parsed["notes"]
        sha = parsed["content_sha256"]
        if sha in seen_sha:
            r["status"] = "reject"
            r["reject"] = list(r.get("reject", [])) + [
                f"gameplay 内容与 {seen_sha[sha]} 完全相同（只差 SV/视觉），去重"]
            continue
        seen_sha[sha] = os.path.basename(p)
        if len(notes) < MIN_NOTES:
            r["status"] = "reject"
            r["reject"] = list(r.get("reject", [])) + [f"音符数 {len(notes)} < {MIN_NOTES}"]
            continue
        rows = {}
        for lane, start, _end in notes:
            c = int(round(GRID_DIV * beat_at(segments, start)))
            rows[c] = rows.get(c, 0) | (1 << lane)
        cells = sorted(rows)
        charts.append({
            "verify": r, "song": song_key, "parsed": parsed, "segments": segments,
            "rows": rows, "cells": cells, "feat": feat, "cm": cm,
            "chart_content_sha": parsed["content_sha256"],
        })

    # ---- split：按谱面分组，同谱不跨集（每首歌内按内容哈希排序后轮流分配）----
    by_song = collections.defaultdict(list)
    for ch in charts:
        by_song[ch["song"]].append(ch)
    SPLIT_NAME = {0: "train", 1: "val", 2: "test"}
    # 每首歌的第一张（按内容哈希）必进 train，保证两首歌都在训练集里；
    # 剩下的谱按 (歌, 哈希) 顺序轮流进 val / test。任何谱整体只属于一个 split。
    rest = []
    for song_key, group in by_song.items():
        group.sort(key=lambda c: c["chart_content_sha"] or "")
        group[0]["split"] = 0
        rest.extend(group[1:])
    rest.sort(key=lambda c: (c["song"], c["chart_content_sha"] or ""))
    for i, ch in enumerate(rest):
        ch["split"] = 1 if i % 2 == 0 else 2

    X_l, C_l, song_l, cid_l, split_l, cell_l, bpm_l, t0_l, cov_l = ([] for _ in range(9))
    per_chart = []
    var_tempo_charts = 0
    gap_num = gap_den = 0

    for ch in charts:
        r, segments, rows, feat = ch["verify"], ch["segments"], ch["rows"], ch["feat"]
        cm = ch["cm"]
        off_c, step_c = float(cm["offset_ms"]), float(cm["step_ms"])
        steps_c = int(cm["steps"])
        dur_ms = float(fx[ch["song"]]["durationSec"]) * 1000.0
        distinct_bpm = {round(60000.0 / sg["beat_length_ms"], 2) for sg in segments}
        if len(distinct_bpm) > 1:
            var_tempo_charts += 1
        cells = ch["cells"]
        min_cell, max_cell = cells[0], cells[-1]
        cid = "gja_" + (ch["chart_content_sha"] or "")[:20]
        cid_l_u = np.array(cid, dtype=f"U{len(cid)}")
        lo = int(math.floor((min_cell - window_cells) / stride_cells))
        hi = int(math.floor(max_cell / stride_cells))
        n_win = 0
        for k in range(lo, hi + 1):
            start = k * stride_cells
            m = build_roll(rows, start, window_cells)
            filled = int(np.count_nonzero(m.sum(axis=0)))
            if filled < min_fill:
                continue
            t0 = ms_of_cell(segments, start)
            t1 = ms_of_cell(segments, start + window_cells)
            if t0 < -100.0 or t1 > dur_ms + 100.0:
                continue
            # 条件切片：逐格按绝对时间取 1/16 格特征
            cvec = np.zeros((window_cells, 4), dtype=np.float32)
            valid = 0
            for j in range(window_cells):
                kk = cond_lookup(feat, off_c, step_c, ms_of_cell(segments, start + j))
                if kk >= 0:
                    cvec[j] = feat[kk]
                    valid += 1
            cov = valid / float(window_cells)
            if cov < 0.5:
                continue
            gnum, gden = p132_of(m)
            gap_num += gnum
            gap_den += gden
            X_l.append(m)
            C_l.append(cvec)
            song_l.append(SONG_ID[ch["song"]])
            cid_l.append(cid_l_u)
            split_l.append(ch["split"])
            cell_l.append(start)
            bpm_l.append(60000.0 / segments[max(0, bisect.bisect_right(
                [s["beat_start"] for s in segments],
                start / float(GRID_DIV)) - 1)]["beat_length_ms"])
            t0_l.append(t0)
            cov_l.append(cov)
            n_win += 1
        per_chart.append({
            "chart_id": cid, "set_id": r["set_id"], "song": ch["song"],
            "osu": r["osu"], "split": SPLIT_NAME[ch["split"]],
            "tier": r["tier"], "notes": len(ch["parsed"]["notes"]),
            "windows": n_win, "n_distinct_bpm": len(distinct_bpm),
            "bpm_set": sorted(distinct_bpm)[:6], "variable_tempo": len(distinct_bpm) > 1,
            "bpm_chart": r["bpm_chart"],
        })

    X = np.stack(X_l).astype(np.uint8) if X_l else np.zeros((0, 4, window_cells), np.uint8)
    C = np.stack(C_l).astype(np.float16) if C_l else np.zeros((0, window_cells, 4), np.float16)
    meta = {
        "X": X, "C": C,
        "song": np.array(song_l, dtype=np.int8),
        "chart_id": np.array(cid_l, dtype=cid_l[0].dtype if cid_l else "U24"),
        "split": np.array(split_l, dtype=np.int8),
        "start_cell": np.array(cell_l, dtype=np.int32),
        "songs": np.array([k for k, _ in SONGS], dtype="U48"),
        "bpm": np.array(bpm_l, dtype=np.float32),
        "t0_ms": np.array(t0_l, dtype=np.float32),
        "cond_coverage": np.array(cov_l, dtype=np.float32),
        "window_cells": np.array(window_cells, dtype=np.int32),
        "stride_cells": np.array(stride_cells, dtype=np.int32),
        "grid_division": np.array(GRID_DIV, dtype=np.int32),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **meta)

    # ---- 统计 ----
    splits = collections.Counter(SPLIT_NAME[int(s)] for s in split_l)
    per_song = collections.defaultdict(lambda: {"windows": 0, "charts": 0})
    for c in per_chart:
        per_song[c["song"]]["charts"] += 1
    for s in song_l:
        per_song[SONGS[s][0]]["windows"] += 1
    occ = float(X.mean()) if X.size else 0.0
    stats = {
        "windows": int(X.shape[0]),
        "charts": len(per_chart),
        "songs_with_windows": sorted(per_song),
        "split_counts": dict(splits),
        "occupancy": round(occ, 4),
        "p132_pooled": round(gap_num / gap_den, 4) if gap_den else None,
        "p132_gap_den": gap_den,
        "cond_coverage_mean": round(float(np.mean(cov_l)), 4) if cov_l else None,
        "variable_tempo_charts": var_tempo_charts,
        "notes_per_window_mean": round(float(X.sum(axis=(1, 2)).mean()), 1) if X.size else 0.0,
        "per_song": {k: v for k, v in sorted(per_song.items())},
        "per_chart": per_chart,
        "window_cells": window_cells, "stride_cells": stride_cells, "min_fill": min_fill,
        "grid_division": GRID_DIV,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"stats": stats, "verify": verify_rows}, fh, ensure_ascii=False, indent=1)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osz-root", default="out/diff-data-audio/_dl")
    ap.add_argument("--cond-root", default="out/multisong")
    ap.add_argument("--out", default="out/diff-data-audio/dataset.npz")
    ap.add_argument("--report", default="out/diff-data-audio/report.json")
    ap.add_argument("--window-cells", type=int, default=WIN)
    ap.add_argument("--stride-cells", type=int, default=STRIDE)
    ap.add_argument("--min-fill", type=int, default=MIN_FILL)
    ap.add_argument("--dry-verify", action="store_true")
    args = ap.parse_args(argv)

    parser = load_parser()
    print("[1] 逐 set 验证（标题/艺术家 → 时长 → BPM）")
    rows = verify_sets(args.osz_root, args.cond_root, parser)
    acc = [r for r in rows if r.get("status") == "accept"]
    print(f"    谱面 {len(rows)} 张：接受 {len(acc)}，拒绝 {len(rows)-len(acc)}")
    if args.dry_verify:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"verify": rows}, fh, ensure_ascii=False, indent=1)
        return 0
    print("[2] 切窗口 + 条件切片")
    stats = build_dataset(rows, args.osz_root, args.cond_root, args.out,
                          args.report, args.window_cells, args.stride_cells,
                          args.min_fill, parser)
    print(f"[3] 写出 {args.out}")
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("per_chart",)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
