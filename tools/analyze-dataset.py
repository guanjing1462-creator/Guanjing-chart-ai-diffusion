#!/usr/bin/env python3
"""
analyze-dataset.py — 数据集画像（给「能不能训」提供证据）

用法：
    python tools/analyze-dataset.py
输出：
    out/dataset-profile.png   四联图：星数 / BPM / 时长 / 密度
    out/dataset-summary.json  汇总数字
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out" / "dataset.jsonl"
INDEX = ROOT / "out" / "charts" / "_index.json"


def load():
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    idx = {}
    if INDEX.exists():
        idx = json.loads(INDEX.read_text(encoding="utf-8"))
    return rows, idx


def main():
    rows, idx = load()
    if not rows:
        raise SystemExit("dataset.jsonl 是空的，先跑 build-dataset.mjs")

    # 每谱统计
    stats = []
    for r in rows:
        toks = r["tokens"]
        frames = r["frames"]
        notes = sum(bin(t).count("1") for t in toks)
        dur = frames * r["stepMs"] / 1000.0
        meta = idx.get(str(r["id"])) or idx.get(r["id"]) or {}
        stats.append({
            "id": r["id"],
            "frames": frames,
            "notes": notes,
            "dur": dur,
            "nps": notes / dur if dur > 0 else 0,
            "bpm": r["bpm"],
            "stars": meta.get("stars"),
        })

    total_frames = sum(s["frames"] for s in stats)
    total_notes = sum(s["notes"] for s in stats)
    stars = [s["stars"] for s in stats if s["stars"] is not None]

    summary = {
        "charts": len(stats),
        "frames": total_frames,
        "notes": total_notes,
        "avg_duration_s": round(sum(s["dur"] for s in stats) / len(stats), 1),
        "avg_nps": round(sum(s["nps"] for s in stats) / len(stats), 2),
        "star_min": round(min(stars), 2) if stars else None,
        "star_max": round(max(stars), 2) if stars else None,
        "star_avg": round(sum(stars) / len(stars), 2) if stars else None,
        "need_for_20m": 5_000_000,
        "gap_factor": round(5_000_000 / max(total_frames, 1), 1),
    }

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="#0b0f18")
    fig.suptitle("osu!mania 4K dataset profile", color="#e8f2ff", fontsize=14, y=0.97)

    def style(ax, title, xlabel):
        ax.set_facecolor("#0f1626")
        ax.set_title(title, color="#9fb3cc", fontsize=11)
        ax.set_xlabel(xlabel, color="#7d90ad", fontsize=9)
        ax.tick_params(colors="#7d90ad", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#22314a")
        ax.grid(alpha=0.15, color="#5a96dc")

    ax = axes[0][0]
    if stars:
        ax.hist(stars, bins=24, color="#38e8ff", alpha=0.85)
    style(ax, f"Star rating  (n={len(stars)})", "★")

    ax = axes[0][1]
    ax.hist([s["bpm"] for s in stats], bins=24, color="#ff4fd8", alpha=0.85)
    style(ax, "BPM", "BPM")

    ax = axes[1][0]
    ax.hist([s["dur"] for s in stats], bins=24, color="#ffc23d", alpha=0.85)
    style(ax, "Duration", "seconds")

    ax = axes[1][1]
    ax.hist([s["nps"] for s in stats], bins=24, color="#4dffa6", alpha=0.85)
    style(ax, "Density", "notes / second")

    txt = (f"charts {summary['charts']}   frames {summary['frames']:,}   notes {summary['notes']:,}\n"
           f"avg {summary['avg_duration_s']}s / {summary['avg_nps']} nps   "
           f"star {summary['star_avg']} (range {summary['star_min']}–{summary['star_max']})\n"
           f"need ~5,000,000 frames for 20M params  →  gap {summary['gap_factor']}x")
    fig.text(0.5, 0.015, txt, ha="center", color="#7d90ad", fontsize=9.5)
    plt.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(ROOT / "out" / "dataset-profile.png", dpi=130, facecolor="#0b0f18")
    (ROOT / "out" / "dataset-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
