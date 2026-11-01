#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch-osu-packs: 按分类（jack/tech/speed/stamina）从 osu.direct 拉 4K mania 谱面。

只下 .osu 单文件（几十 KB），不下整包音频。按分类建独立语料，供键型统计单独训练。

用法：
  python3 tools/fetch-osu-packs.py --type jack --sets 25
  python3 tools/fetch-osu-packs.py --type all --sets 25
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

UA = 'gj-chart-research/0.1 (personal chart study)'
BASE = 'https://osu.direct'
QUERIES = {
    'jack':    ['jack pack', 'jack', 'chordjack', 'jack speed', 'minijack'],
    'tech':    ['tech pack', 'technical', 'tech map', 'technical pack'],
    'speed':   ['speed pack', 'speed', 'vibro', 'stream speed'],
    'stamina': ['stamina pack', 'stamina', 'marathon', 'endurance'],
}


def get(url, binary=False, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            return data if binary else data.decode('utf-8', 'ignore')
        except Exception as e:
            if i == tries - 1:
                print(f'    ! {type(e).__name__} {url}', flush=True)
                return None
            time.sleep(1.5 * (i + 1))


def search(q, limit=50):
    url = f'{BASE}/api/v2/search?{urllib.parse.urlencode({"q": q, "m": 3, "limit": limit})}'
    txt = get(url)
    if not txt:
        return []
    try:
        return json.loads(txt)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default='jack')
    ap.add_argument('--sets', type=int, default=20, help='每个 query 最多取多少个 set')
    ap.add_argument('--min-notes', type=int, default=400)
    ap.add_argument('--out', default='out/packs')
    ap.add_argument('--max-per-type', type=int, default=500)
    args = ap.parse_args()

    kinds = list(QUERIES) if args.type == 'all' else [args.type]
    for kind in kinds:
        outdir = os.path.join(args.out, kind)
        os.makedirs(outdir, exist_ok=True)
        seen_path = os.path.join(outdir, 'seen.json')
        seen = set(json.load(open(seen_path))) if os.path.exists(seen_path) else set()
        got = 0
        fail_streak = 0
        for q in QUERIES[kind]:
            sets = search(q)
            sets = sets[:args.sets]
            print(f'[{kind}] q={q!r} → {len(sets)} sets', flush=True)
            if got >= args.max_per_type:
                break
            for s in sets:
                for bm in (s.get('beatmaps') or []):
                    if bm.get('mode') != 'mania' or int(bm.get('cs') or 0) != 4:
                        continue
                    if int(bm.get('count_circles') or 0) < args.min_notes:
                        continue
                    mid = bm['id']
                    if mid in seen:
                        continue
                    path = os.path.join(outdir, f'{mid}.osu')
                    if os.path.exists(path):
                        seen.add(mid)
                        got += 1
                        continue
                    data = get(f'{BASE}/api/osu/{mid}', binary=True)
                    if not data or not data.startswith(b'osu file format'):
                        fail_streak += 1
                        if fail_streak >= 12:
                            print('  (连续失败，暂停 20s)', flush=True)
                            time.sleep(20)
                            fail_streak = 0
                        continue
                    fail_streak = 0
                    open(path, 'wb').write(data)
                    seen.add(mid)
                    got += 1
                    if got >= args.max_per_type:
                        break
                    if got % 25 == 0:
                        print(f'  [{kind}] 已下 {got}', flush=True)
                    time.sleep(0.08)
            json.dump(sorted(seen), open(seen_path, 'w'))
        json.dump(sorted(seen), open(seen_path, 'w'))
        print(f'[{kind}] 完成：{len(seen)} 张谱 → {outdir}', flush=True)


if __name__ == '__main__':
    main()
