#!/usr/bin/env python3
"""结构后处理：给扩散生成的谱面加「段落动机复用」。

人类谱 8 行窗重复率 0.259，扩散生成是 0.000（无段落结构）。
做法：把谱面切成 8/16/32 行的段落，选若干段整段复制到别的段位置（时间平移），
密度和键型分布不变，只增加长程重复。

用法：python3 tools/structure-post.py --in a.osu --out b.osu --repeat 0.35
"""
import argparse, copy, random


def load_osu(path):
    lines = open(path, encoding='utf-8').read().split('\n')
    head, body, mode = [], [], 0
    for ln in lines:
        if ln.strip() == '[HitObjects]':
            mode = 1
            head.append(ln)
            continue
        if mode:
            if ln.strip():
                body.append(ln)
        else:
            head.append(ln)
    return head, body


def parse(body):
    rows = {}
    for ln in body:
        p = ln.split(',')
        if len(p) < 5:
            continue
        rows.setdefault(int(p[2]), []).append(ln)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--repeat', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    head, body = load_osu(a.src)
    rows = parse(body)
    times = sorted(rows)
    if len(times) < 64:
        open(a.dst, 'w', encoding='utf-8').write('\n'.join(head + body) + '\n')
        print('行数太少，跳过')
        return

    segs, i = [], 0
    while i < len(times):
        k = random.choice([8, 16, 16, 32])
        segs.append((i, min(i + k, len(times))))
        i += k

    n_rep = max(1, int(len(segs) * a.repeat))
    idxs = list(range(len(segs)))
    random.shuffle(idxs)
    src_pool = idxs[:n_rep]
    dst_pool = idxs[n_rep:2 * n_rep] if len(idxs) >= 2 * n_rep else idxs[-n_rep:]

    replaced = 0
    for s, d in zip(src_pool, dst_pool):
        s0, s1 = segs[s]
        d0, d1 = segs[d]
        src_t = times[s0:s1]
        dst_t = times[d0:d1]
        if not src_t or not dst_t:
            continue
        # 源段可能已被之前的复制覆盖掉一部分，跳过这些时间点
        src_t = [t for t in src_t if t in rows]
        if len(src_t) < 4:
            continue
        # 先把源段内容拷出来（避免边改边读）
        snapshot = [(t, copy.copy(rows[t])) for t in src_t if t in rows]
        shift = dst_t[0] - src_t[0]
        for t in dst_t:
            rows.pop(t, None)
        for k, (t, lines) in enumerate(snapshot):
            if k >= len(dst_t):
                break
            new_t = t + shift
            bucket = rows.setdefault(new_t, [])
            for ln in lines:
                p = ln.split(',')
                p[2] = str(new_t)
                bucket.append(','.join(p))
        replaced += 1

    out = []
    for t in sorted(rows):
        for ln in rows[t]:
            p = ln.split(',')
            p[2] = str(t)
            out.append(','.join(p))
    open(a.dst, 'w', encoding='utf-8').write('\n'.join(head + out) + '\n')
    print(f'结构后处理：{len(segs)} 段，替换 {replaced} 段 → {a.dst}')


if __name__ == '__main__':
    main()
