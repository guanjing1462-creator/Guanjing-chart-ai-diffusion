#!/usr/bin/env python3
"""节奏规整化：把音符时间吸附到 {1/4,1/8,1/16,1/32} 网格上（保留细分变化，去掉怪间隔）。"""
import argparse, json, sys

def snap(src, dst, step_ms, offset_ms, allow=(1, 2, 4, 8)):
    grid = sorted({round(step_ms * k, 3) for k in allow})
    lines = open(src, encoding='utf-8').read().split('\n')
    out, in_hit = [], False
    for ln in lines:
        if ln.strip() == '[HitObjects]':
            in_hit = True; out.append(ln); continue
        if in_hit and ln.count(',') >= 5:
            p = ln.split(',')
            t = int(p[2])
            rel = t - offset_ms
            # 找最近的允许网格步长
            best = min(grid, key=lambda g: abs(rel - round(rel / g) * g))
            p[2] = str(int(round(offset_ms + round(rel / best) * best)))
            out.append(','.join(p))
        else:
            out.append(ln)
    open(dst, 'w', encoding='utf-8').write('\n'.join(out))
    return len(grid)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--grids', default='1,2,4,8')
    a = ap.parse_args()
    an = json.load(open(a.analysis))
    allow = tuple(int(x) for x in a.grids.split(','))
    n = snap(a.src, a.dst, an['stepMs'], an['offsetMs'], allow)
    print(f'规整化完成：{a.dst}（{n} 种网格）')
