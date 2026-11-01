#!/usr/bin/env python3
"""Round147 · LN 版采样统计 vs 人类分布对照。

读三样东西：
  1. out/diff-data-ln/report.json（人类 LN 分布，数据侧口径）
  2. 若干个 tools/sample-diffusion32-ln.py 产出的 stats.json（每个模型可以有多个采样批次）
  3. 训练日志 jsonl（可选，拿 val 损失）

按「窗口」合并同一模型的多个采样批次，输出：
  LN 头数/窗、LN 体格数/窗、LN 长度分布（中位/p25/p75/p95/max）、有 LN 洞占比、
  与人类值的比值 / ±50% 判定。

用法
----
  python3 tools/round147-ln-compare.py \
    --model l1.0=out/diffusion/round147-ln/samples/l1.0-s3/stats.json,\
out/diffusion/round147-ln/samples/l1.0-s0/stats.json,... \
    --out out/diffusion/round147-ln/compare.json
"""
import argparse
import glob
import json
import os

import numpy as np

STYLES = ("tech", "jack", "speed", "stamina")


def load_model_stats(paths):
    """合并若干 stats.json：返回 (窗口级数组, 池化统计)。"""
    heads, cells, occ_w, lens = [], [], [], []
    for p in paths:
        st = json.load(open(p, encoding="utf-8"))["stats"]
        heads.extend(st["LN头数_每窗"])
        cells.extend(st["LN体格数_每窗"])
        occ_w.append(st["占位率"])
        for row in st["LN长度_每窗"]:
            lens.extend(row)
    n = len(heads)
    return {
        "窗口数": n,
        "LN头数_每窗": heads,
        "LN体格数_每窗": cells,
        "LN长度": lens,
        "占位率_各批": occ_w,
    }


def dist(vals):
    if not len(vals):
        return {"n": 0}
    v = np.asarray(vals, dtype=float)
    return {"n": int(v.size), "均值": round(float(v.mean()), 2),
            "中位数": round(float(np.median(v)), 2),
            "p25": round(float(np.percentile(v, 25)), 2),
            "p75": round(float(np.percentile(v, 75)), 2),
            "p95": round(float(np.percentile(v, 95)), 2),
            "最大": int(v.max())}


def within(a, b, tol):
    """a 是否落在 b 的 ±tol 内；b=0 时只有 a=0 才算通过。"""
    if b == 0:
        return a == 0
    return abs(a - b) / b <= tol


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="out/diff-data-ln/report.json")
    ap.add_argument("--model", action="append", default=[], required=True,
                    help="名字=stats.json[,stats2.json...]（可重复）")
    ap.add_argument("--tol", type=float, default=0.5, help="判定容差（任务书 ±50%）")
    ap.add_argument("--out", default="out/diffusion/round147-ln/compare.json")
    a = ap.parse_args(argv)

    human = json.load(open(a.human, encoding="utf-8"))
    h_len = human["LN长度分布_全局"]
    h_lenw = human["LN长度分布_窗口裁剪"]
    h_ref = {
        "每窗口平均LN头数": human["每窗口平均LN头数"],
        "每窗口平均LN体格数": human["每窗口平均LN体格数"],
        "LN长度中位_全局": h_len["中位数"],
        "LN长度p95_全局": h_len["p95"],
        "LN长度中位_窗口裁剪": h_lenw["中位数"],
        "LN长度p95_窗口裁剪": h_lenw["p95"],
        "占位率": human["占位率"],
        "有LN窗口占比": human["有LN的窗口占比"],
    }
    # 风格均衡参考（20 窗 = 每风格 5 窗，与人类按窗口数加权的比例不同）
    h_style_heads = human["每style每窗口平均LN头数"]
    h_style_cells = human["每style每窗口平均LN体格数"]
    h_style_med = {s: human["每style_LN长度分布_窗口裁剪"][s].get("中位数")
                   for s in STYLES}
    h_style_p95 = {s: human["每style_LN长度分布_窗口裁剪"][s].get("p95")
                   for s in STYLES}
    h_ref["风格均衡_每窗口平均LN头数"] = round(
        float(np.mean([h_style_heads[s] for s in STYLES])), 3)
    h_ref["风格均衡_每窗口平均LN体格数"] = round(
        float(np.mean([h_style_cells[s] for s in STYLES])), 3)

    res = {"人类参考": h_ref,
           "人类按风格": {"每style每窗口平均LN头数": h_style_heads,
                        "每style每窗口平均LN体格数": h_style_cells,
                        "每style_LN长度中位_窗口裁剪": h_style_med,
                        "每style_LN长度p95_窗口裁剪": h_style_p95},
           "判定容差": a.tol,
           "模型": {}}

    for item in a.model:
        if "=" not in item:
            raise SystemExit(f"[错误] --model 需要 名字=路径 形式：{item!r}")
        name, paths = item.split("=", 1)
        plist = []
        for p in paths.split(","):
            p = p.strip()
            if not p:
                continue
            plist.extend(sorted(glob.glob(p)))
        if not plist:
            raise SystemExit(f"[错误] {name}: 没有匹配到 stats.json")
        m = load_model_stats(plist)
        heads = m["LN头数_每窗"]
        cells = m["LN体格数_每窗"]
        nwin = len(heads)
        merged = {
            "采样批次": [os.path.basename(os.path.dirname(p)) for p in plist],
            "窗口数": nwin,
            "LN头数_每窗均值": round(float(np.mean(heads)), 3) if nwin else 0.0,
            "LN体格数_每窗均值": round(float(np.mean(cells)), 3) if nwin else 0.0,
            "有LN窗口占比": round(float(np.mean([h > 0 for h in heads])), 4) if nwin else 0.0,
            "LN长度分布": dist(m["LN长度"]),
            "占位率_各批": m["占位率_各批"],
            "LN头数_每窗": heads,
            "LN体格数_每窗": cells,
        }
        merged["对照"] = {
            "LN头数_比值": round(merged["LN头数_每窗均值"] / max(1e-9, h_ref["每窗口平均LN头数"]), 3),
            "LN体格数_比值": round(merged["LN体格数_每窗均值"] / max(1e-9, h_ref["每窗口平均LN体格数"]), 3),
            "LN长度中位_比值": (round(merged["LN长度分布"]["中位数"]
                                / max(1e-9, h_ref["LN长度中位_全局"]), 3)
                           if merged["LN长度分布"]["n"] else None),
            "LN长度p95_比值": (round(merged["LN长度分布"]["p95"]
                                / max(1e-9, h_ref["LN长度p95_全局"]), 3)
                           if merged["LN长度分布"]["n"] else None),
            "LN头数_在±tol内": within(merged["LN头数_每窗均值"],
                                    h_ref["每窗口平均LN头数"], a.tol),
            "LN长度中位_在±tol内": (within(merged["LN长度分布"]["中位数"],
                                      h_ref["LN长度中位_全局"], a.tol)
                                 if merged["LN长度分布"]["n"] else False),
        }
        res["模型"][name] = merged

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)

    print(f"人类参考（池化）：LN 头 {h_ref['每窗口平均LN头数']}/窗，"
          f"体格 {h_ref['每窗口平均LN体格数']}/窗，长度中位 {h_ref['LN长度中位_全局']} 格，"
          f"p95 {h_ref['LN长度p95_全局']} 格，有LN窗口 {100*h_ref['有LN窗口占比']:.1f}%")
    print(f"          风格均衡：LN 头 {h_ref['风格均衡_每窗口平均LN头数']}/窗，"
          f"体格 {h_ref['风格均衡_每窗口平均LN体格数']}/窗")
    print(f"{'模型':<14}{'窗':>4}{'LN头/窗':>9}{'体格/窗':>9}{'长度中位':>9}"
          f"{'p95':>7}{'有LN%':>7}  判定(±{int(a.tol*100)}%)")
    for name, m in res["模型"].items():
        d = m["LN长度分布"]
        print(f"{name:<14}{m['窗口数']:>4}{m['LN头数_每窗均值']:>9.2f}"
              f"{m['LN体格数_每窗均值']:>9.2f}{d.get('中位数', 0):>9.1f}"
              f"{d.get('p95', 0):>7.1f}{100*m['有LN窗口占比']:>7.1f}  "
              f"头{'✓' if m['对照']['LN头数_在±tol内'] else '✗'} "
              f"长度{'✓' if m['对照']['LN长度中位_在±tol内'] else '✗'}")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
