#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot-chart: 把谱面画成「折叠钢琴卷」+ 密度/能量对照图（纯 PIL，无依赖）。

为什么需要它：谱面的「一坨感」很难用手指指标说清，但画出来一眼就能看见——
人类谱的折叠图有明显重复的图案块（动机），随机谱是一片均匀的噪点。

用法：
  python3 tools/plot-chart.py --osu X.osu [--analysis ...] --strip-sec 20 \
      -o out/chart.png [--width 1400]
"""
import argparse
import json
import statistics as st

from PIL import Image, ImageDraw

COLS = {64: 0, 192: 1, 320: 2, 448: 3}


def parse(path):
    notes, meta, sec = [], {}, None
    for ln in open(path, encoding='utf-8', errors='ignore'):
        s = ln.strip()
        if s.startswith('['):
            sec = s
            continue
        if sec == '[Metadata]' and ':' in s:
            k, v = s.split(':', 1)
            meta[k.strip()] = v.strip()
        elif sec == '[HitObjects]' and s and s[0].isdigit():
            p = s.split(',')
            if len(p) >= 5:
                col = min(range(4), key=lambda i: abs((i + 0.5) * 128 - int(p[0])))
                t = int(p[2])
                end = None
                if p[3] == '128' and len(p) > 5:
                    e = p[5].split(':')[0]
                    if e.isdigit() and int(e) > t:
                        end = int(e)
                notes.append((t, col, end))
    notes.sort(key=lambda n: (n[0], n[1]))
    return notes, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osu', required=True)
    ap.add_argument('--analysis', default='out/final14-s50/analysis.json')
    ap.add_argument('--strip-sec', type=float, default=20.0)
    ap.add_argument('--width', type=int, default=1500)
    ap.add_argument('-o', required=True)
    args = ap.parse_args()
    notes, meta = parse(args.osu)
    t0 = notes[0][0]
    t1 = notes[-1][0]
    dur = max(1000.0, t1 - t0)
    n_strip = max(1, int(dur / (args.strip_sec * 1000) + 0.999))
    strip_ms = dur / n_strip

    W, H = args.width, int(args.width * 0.78)
    panel_h = H - 100
    left_w = int(W * 0.62)
    strip_w = left_w // n_strip if n_strip else left_w
    lane_w = max(4, strip_w // 4)          # 每条带子内部 4 条轨
    pad = 10
    img = Image.new('RGB', (W, H), (18, 18, 22))
    d = ImageDraw.Draw(img)
    lane_cols = [(90, 160, 255), (255, 210, 90), (255, 130, 190), (120, 230, 160)]

    # ---- 折叠钢琴卷 ----
    for k in range(n_strip):
        x0 = k * strip_w
        d.rectangle([x0, 40, x0 + strip_w - 1, 40 + panel_h], outline=(40, 40, 48))
        for c in range(5):
            d.line([x0 + c * lane_w, 40, x0 + c * lane_w, 40 + panel_h],
                   fill=(28, 28, 34))
    for t, c, end in notes:
        k = min(n_strip - 1, int((t - t0) / strip_ms))
        y = 40 + int((t - t0 - k * strip_ms) / strip_ms * panel_h)
        x = k * strip_w + int(c * lane_w) + 2
        w = max(2, int(lane_w) - 4)
        col = lane_cols[c]
        if end:
            y2 = 40 + int((min(end, t0 + (k + 1) * strip_ms) - t0 - k * strip_ms) / strip_ms * panel_h)
            d.rectangle([x, y, x + w, max(y + 2, y2)], fill=(col[0] // 2, col[1] // 2, col[2] // 2))
            d.rectangle([x, y, x + w, y + 2], fill=col)
        else:
            d.rectangle([x, y, x + w, y + max(2, panel_h // 400)], fill=col)

    # ---- 右侧：密度 vs 能量 ----
    rx0 = left_w + 24
    rx1 = W - 24
    seg = 2000.0
    nseg = int(dur / seg) + 1
    dens = [0] * nseg
    for t, c, e in notes:
        k = min(nseg - 1, int((t - t0) / seg))
        dens[k] += 1
    d.rectangle([rx0, 40, rx1, 40 + panel_h], outline=(40, 40, 48))
    maxd = max(dens) or 1
    pts = []
    for k, v in enumerate(dens):
        x = rx0 + k / max(1, nseg - 1) * (rx1 - rx0)
        y = 40 + panel_h - v / maxd * (panel_h * 0.45)
        pts.append((x, y))
    if len(pts) > 1:
        d.line(pts, fill=(120, 230, 160), width=2)
    # 能量曲线
    try:
        an = json.load(open(args.analysis))
        acc = an['accent']
        off, step = an['offsetMs'], an['stepMs']
        win = max(1, int(2000 / step))
        em = [st.mean(acc[i:i + win]) for i in range(0, len(acc), win)]
        me = max(em) or 1
        pts2 = []
        for k, v in enumerate(em):
            x = rx0 + k / max(1, len(em) - 1) * (rx1 - rx0)
            y = 40 + panel_h - v / me * (panel_h * 0.45) - panel_h * 0.5
            pts2.append((x, y))
        if len(pts2) > 1:
            d.line(pts2, fill=(90, 160, 255), width=2)
    except Exception:      # noqa: BLE001
        pass

    # ---- 文字 ----
    title = f"{meta.get('Title', '')} [{meta.get('Version', '')}]  notes {len(notes)}"
    d.text((12, 12), title, fill=(230, 230, 240))
    d.text((12, H - 46), 'folding piano roll: 4 lanes × %d strips × %.0fs' % (n_strip, strip_ms / 1000),
           fill=(150, 150, 165))
    d.text((rx0, H - 46), 'green: rows / 2s     blue: audio accent / 2s', fill=(150, 150, 165))
    img.save(args.o)
    print('写出', args.o, img.size)


if __name__ == '__main__':
    main()
