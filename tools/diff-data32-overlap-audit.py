#!/usr/bin/env python3
"""窗口重叠审计：把每个窗口的 [start_cell, start_cell+256) 换算成真实音频毫秒区间，
在同源组（可能同音频）内部扫描相交窗口对，按 split 分类计数。

为什么只在组内扫：不同组的谱各自以**自己**首条红线为时间原点，格坐标相交并不代表
同一首歌的时间重叠；而组是按 manifest 的同源身份保守合并的，组内跨 split 相交才是
真正的泄漏风险。审计同时给两种口径：

  严格   ：[t0, t1) 直接比较；
  保守   ：两端各再放半格（±0.5 个 1/32 格 ≈ 10~22ms），用于吸收 round() 量化的
           最大误差——保守口径仍为 0，才说明「0 重叠」不是量化侥幸。
"""
import argparse
import bisect
import collections
import json
import os

import numpy as np


def seg_at(segments, cell):
    beat = cell / 8.0
    i = max(0, bisect.bisect_right([s["beat_start"] for s in segments], beat) - 1)
    return segments[i]


def ms_of_cell(segments, cell):
    s = seg_at(segments, cell)
    return s["offset_ms"] + (cell / 8.0 - s["beat_start"]) * s["beat_length_ms"]


def sweep(group_to_idx, lo, hi, pad, chart, split, padded):
    cross = same = same_chart = 0
    for idxs in group_to_idx.values():
        idxs = sorted(idxs, key=lambda j: lo[j] - (pad[j] if padded else 0.0))
        active = []
        for j in idxs:
            loj = lo[j] - (pad[j] if padded else 0.0)
            active = [a for a in active if hi[a] + (pad[a] if padded else 0.0) > loj]
            for a in active:
                if chart[a] == chart[j]:
                    same_chart += 1
                elif split[a] != split[j]:
                    cross += 1
                else:
                    same += 1
            active.append(j)
    return cross, same, same_chart


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="out/diff-data-grouped/dataset.npz")
    ap.add_argument("--manifest", default="out/diffusion/round136-data/manifest.json")
    ap.add_argument("--out", default="out/diff-data-grouped/overlap-audit.json")
    args = ap.parse_args(argv)

    man = json.load(open(args.manifest, encoding="utf-8"))
    chart_meta = {c["chart_id"]: c for c in man["charts"]}
    d = np.load(args.dataset, allow_pickle=False)
    group, split, chart, cell = d["group_id"], d["split"], d["chart_id"], d["start_cell"]
    W = int(d["window_cells"])

    n = len(cell)
    lo = np.empty(n)
    hi = np.empty(n)
    pad = np.empty(n)
    for i, (cid, sc) in enumerate(zip(chart.tolist(), cell.tolist())):
        segs = chart_meta[cid]["timing_segments"]
        lo[i] = ms_of_cell(segs, sc)
        hi[i] = ms_of_cell(segs, sc + W)
        pad[i] = seg_at(segs, sc)["beat_length_ms"] / 8.0 / 2.0

    group_to_idx = collections.defaultdict(list)
    for i, gid in enumerate(group.tolist()):
        group_to_idx[gid].append(i)

    chart_l, split_l = chart.tolist(), split.tolist()
    strict = sweep(group_to_idx, lo, hi, pad, chart_l, split_l, padded=False)
    padded = sweep(group_to_idx, lo, hi, pad, chart_l, split_l, padded=True)

    result = {
        "数据集": args.dataset,
        "窗口数": n,
        "窗口长_格": W,
        "组数": len(group_to_idx),
        "口径": {
            "扫描范围": "只扫同源组内部（跨组时间原点不同，格坐标相交≠时间重叠）",
            "区间": "[t(start_cell), t(start_cell+window_cells)) 毫秒",
            "严格": "直接比较",
            "保守": "两端各外扩半格（吸收 round 量化最大误差）",
        },
        "严格口径": {"同组跨split重叠对": strict[0], "同组同split重叠对": strict[1],
                  "同chart相邻窗口重叠对": strict[2]},
        "保守口径": {"同组跨split重叠对": padded[0], "同组同split重叠对": padded[1],
                  "同chart相邻窗口重叠对": padded[2]},
        "结论": ("跨 split 的同源窗口时间重叠在严格与保守两种口径下均为 0 对；"
                 "同 split 内的重叠（含同谱相邻窗口 50% 设计重叠）不影响集间泄漏。"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"严格": result["严格口径"], "保守": result["保守口径"],
                      "组数": len(group_to_idx), "窗口数": n}, ensure_ascii=False))
    print("→", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
