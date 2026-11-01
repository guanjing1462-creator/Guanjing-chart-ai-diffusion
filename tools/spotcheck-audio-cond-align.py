#!/usr/bin/env python3
"""抽查音频条件切片与谱面/音频的时间对齐（round143 数据任务第 3 步）。

对 `out/diff-data-audio/dataset.npz` 抽 3 个窗口，做三个独立口径的对齐证据：

A. **窗口时间范围 vs 音频 onset 峰值**（任务要求的打印项）
   - 打印窗口 [t0, t1) 毫秒范围，以及该范围内 `-cond.npz` 的 onset 通道（ch3）
     在原始 1/16 网格上的峰值时刻（毫秒），并给出「峰值 → 最近谱面音符时刻」的距离。
B. **条件切片 C 与谱面的相对位移扫描**：把 C 的 onset 通道与窗口内 X 的占位列做
   -8..+8 格的移码相关，看峰值是否落在 0 格（无整体偏移）。
C. **谱面音符时刻 vs fx.json onset（另一条数据管线）**：在整首歌上做 -12..12 格位移
   相关，独立验证「.osu 绝对时间 ↔ 音频分析网格」没偏。

输出 out/diff-data-audio/spotcheck.json，并把要点打到 stdout。
"""
import argparse
import bisect
import importlib.util
import json
import math
import os
import sys

import numpy as np

GRID_DIV = 8
PARSER_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gj-data-manifest.py")


def load_parser(path=PARSER_MODULE):
    spec = importlib.util.spec_from_file_location("gj_data_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / d if d > 1e-12 else 0.0


def peaks_of(vec, thr):
    """局部极大值索引（v[i] >= thr 且 v[i] 是左右邻域最大）。"""
    idx = []
    for i in range(1, len(vec) - 1):
        if vec[i] >= thr and vec[i] >= vec[i - 1] and vec[i] > vec[i + 1]:
            idx.append(i)
    idx.sort(key=lambda i: -vec[i])
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/diff-data-audio/dataset.npz")
    ap.add_argument("--cond-root", default="out/multisong")
    ap.add_argument("--osz-root", default="out/diff-data-audio/_dl")
    ap.add_argument("--report", default="out/diff-data-audio/report.json")
    ap.add_argument("--out", default="out/diff-data-audio/spotcheck.json")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args(argv)

    parser = load_parser()
    d = np.load(args.data)
    songs = [str(s) for s in d["songs"]]
    X, C = d["X"], d["C"].astype(np.float32)
    t0_ms = d["t0_ms"].astype(np.float64)
    cov = d["cond_coverage"]
    rep = json.load(open(args.report, encoding="utf-8"))
    chart_meta = {c["chart_id"]: c for c in rep["stats"]["per_chart"]}
    cond_meta = json.load(open(os.path.join(args.cond_root, "cond-meta.json")))["songs"]

    n = X.shape[0]
    # 抽样：按时间分位取 3 个（覆盖早/中/晚），保证不同歌各一个（若可能）
    picks = []
    order = np.argsort(t0_ms)
    seen = set()
    for frac in (0.12, 0.5, 0.86):
        i = int(order[min(n - 1, int(frac * n))])
        j = i
        # 优先选没出现过的歌
        while songs[int(d["song"][j])] in seen and len(seen) < len(set(songs[s] for s in d["song"])):
            j = (j + 1) % n
        picks.append(j)
        seen.add(songs[int(d["song"][j])])
    picks = picks[:args.n]

    out = {"picked": [int(p) for p in picks], "checks": []}
    for p in picks:
        s_idx = int(d["song"][p])
        song = songs[s_idx]
        cm = cond_meta[song]
        off, step, steps = float(cm["offset_ms"]), float(cm["step_ms"]), int(cm["steps"])
        feat = np.load(os.path.join(args.cond_root, f"{song}-cond.npz"))["feat"].astype(np.float32)
        fx = json.load(open(os.path.join(args.cond_root, f"{song}-fx.json"), encoding="utf-8"))
        cid = str(d["chart_id"][p])
        meta = chart_meta.get(cid, {})
        start = int(d["start_cell"][p])
        win = int(d["window_cells"]) if "window_cells" in d else 256
        # 该窗口的 1/32 格长度：用 C 里非零段没法反推，改从 .osu 红线算
        # （谱面分段已知：重新解析一次拿到 segments；这里用 chart 元数据里的 bpm 近似）
        bpm = float(d["bpm"][p])
        cell_ms = 60000.0 / bpm / GRID_DIV

        t0 = float(t0_ms[p])
        t1 = t0 + win * cell_ms
        # ---- A: 条件特征在窗口内的 onset 峰值时刻（回到原始 1/16 网格） ----
        k_lo = max(0, int(math.floor((t0 - off) / step)))
        k_hi = min(steps, int(math.ceil((t1 - off) / step)) + 1)
        seg = feat[k_lo:k_hi, 3]
        thr = float(np.percentile(seg, 85)) if seg.size else 0.0
        pk = peaks_of(seg, max(thr, 0.35))
        pk_times = [(off + (k_lo + i) * step) for i in pk[:6]]
        # 窗口内 C 的 onset 列（ch3）峰值格（相对格号）
        c_on = C[p, :, 3]
        c_pk = peaks_of(c_on, max(float(np.percentile(c_on, 85)), 0.2))[:6]
        c_pk_ms = [t0 + i * cell_ms for i in c_pk]      # 近似（等速假设）
        # ---- B: C.onset vs X 占位列 的位移扫描 ----
        xcol = (X[p].sum(axis=0) > 0).astype(np.float32)
        scan = {}
        for s in range(-8, 9):
            if s >= 0:
                a, b = c_on[s:], xcol[:len(xcol) - s]
            else:
                a, b = c_on[:s], xcol[-s:]
            scan[s] = corr(a, b) if len(a) > 8 and a.std() > 1e-6 and b.std() > 1e-6 else 0.0
        best_s = max(scan, key=lambda k: scan[k])
        # ---- C: 谱面音符时刻 vs fx.json onset（整首歌，独立管线）----
        fx_on = np.asarray(fx["onset"], dtype=np.float32)
        fx_off, fx_step = float(fx["offsetMs"]), float(fx["stepMs"])
        note_times = None
        # 从 .osu 重新解析（chart_id → 文件）
        osu_path = None
        for root, _dirs, files in os.walk(args.osz_root):
            if meta.get("set_id") and os.path.basename(root) == str(meta["set_id"]):
                for f in files:
                    if f == meta.get("osu"):
                        osu_path = os.path.join(root, f)
        if osu_path:
            parsed = parser.parse_chart(osu_path)
            note_times = np.array([n[1] for n in parsed["notes"]], dtype=np.float64)
        gscan = {}
        if note_times is not None and fx_on.size:
            occ = np.zeros_like(fx_on)
            kk = np.floor((note_times - fx_off) / fx_step).astype(int)
            kk = kk[(kk >= 0) & (kk < len(occ))]
            np.add.at(occ, kk, 1.0)
            m = min(len(occ), fx_on.size)
            for s in range(-12, 13):
                if s >= 0:
                    a, b = occ[s:], fx_on[:m - s]
                else:
                    a, b = occ[:s], fx_on[-s:]
                gscan[s] = corr(a, b) if len(a) > 32 and a.std() > 1e-6 and b.std() > 1e-6 else 0.0
        gbest = max(gscan, key=lambda k: gscan[k]) if gscan else None

        rec = {
            "window_index": int(p), "song": song, "song_show": meta.get("song_show"),
            "chart_id": cid, "set_id": meta.get("set_id"), "osu": meta.get("osu"),
            "split": int(d["split"][p]), "start_cell": start,
            "window_t0_ms": round(t0, 1), "window_t1_ms": round(t1, 1),
            "window_span_s": round((t1 - t0) / 1000.0, 2),
            "bpm": bpm, "cell_ms": round(cell_ms, 3),
            "cond_coverage": round(float(cov[p]), 4),
            "cond_onset_peak_times_ms": [round(t, 1) for t in pk_times],
            "cond_onset_peak_values": [round(float(seg[i]), 3) for i in pk[:6]],
            "C_onset_peak_cells": c_pk,
            "C_onset_peak_times_ms_approx": [round(t, 1) for t in c_pk_ms],
            "shift_scan_C_vs_X": {str(k): round(v, 4) for k, v in scan.items()},
            "shift_best": int(best_s), "shift_corr_at_0": round(scan[0], 4),
            "shift_corr_at_best": round(scan[best_s], 4),
            "chart_notes_in_window": int(X[p].sum()),
        }
        # 峰值 → 最近音符时刻
        if note_times is not None:
            inwin = note_times[(note_times >= t0) & (note_times < t1)]
            rec["notes_in_window"] = int(len(inwin))
            dists = []
            for t in pk_times:
                if len(inwin):
                    dists.append(float(np.min(np.abs(inwin - t))))
                else:
                    dists.append(None)
            rec["peak_to_nearest_note_ms"] = [None if v is None else round(v, 1) for v in dists]
            rec["median_peak_to_note_ms"] = (round(float(np.median([v for v in dists if v is not None])), 1)
                                             if any(v is not None for v in dists) else None)
            if gscan:
                rec["global_scan_notes_vs_fx_onset"] = {
                    "best_shift_cells16": int(gbest), "corr_at_0": round(gscan[0], 4),
                    "corr_at_best": round(gscan[gbest], 4),
                    "corr_minus12": round(gscan[-12], 4), "corr_plus12": round(gscan[12], 4),
                    "n_notes": int(len(note_times)),
                }
        out["checks"].append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    for rec in out["checks"]:
        print(f"\n=== 窗口 #{rec['window_index']} {rec['song']} {rec['chart_id']} "
              f"start_cell={rec['start_cell']} ===")
        print(f"  窗口时间范围 {rec['window_t0_ms']:.1f} ~ {rec['window_t1_ms']:.1f} ms "
              f"({rec['window_span_s']}s)  BPM={rec['bpm']} 1/32格={rec['cell_ms']}ms "
              f"条件覆盖={rec['cond_coverage']}")
        print(f"  音频 onset 峰值时刻(ms): {rec['cond_onset_peak_times_ms']}")
        print(f"  峰值→最近谱面音符距离(ms): {rec.get('peak_to_nearest_note_ms')} "
              f"中位 {rec.get('median_peak_to_note_ms')}")
        print(f"  C.onset vs X 位移扫描: best={rec['shift_best']}格 "
              f"corr0={rec['shift_corr_at_0']} corrBest={rec['shift_corr_at_best']}")
        if "global_scan_notes_vs_fx_onset" in rec:
            g = rec["global_scan_notes_vs_fx_onset"]
            print(f"  全曲 音符 vs fx.onset: best_shift={g['best_shift_cells16']} "
                  f"corr0={g['corr_at_0']} corrBest={g['corr_at_best']} "
                  f"控制(±12格)={g['corr_minus12']}/{g['corr_plus12']}")
    print(f"\n写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
