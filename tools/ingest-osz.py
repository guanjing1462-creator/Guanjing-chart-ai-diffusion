#!/usr/bin/env python3
"""把 .osz 池解压进工作目录，只处理还没解压的。

    python3 tools/ingest-osz.py --pool /tmp/osz_pool --dest /tmp/osz_x

只提取 .osu / 音频，其余（背景图、音效）全部跳过 —— 一个包 12MB 里
真正需要的不到 8MB，音效 wav 加起来经常比正曲还大。
"""
import argparse
import os
import re
import shutil
import zipfile

KEEP = ('.osu', '.mp3', '.ogg', '.wav', '.m4a')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='/tmp/osz_pool')
    ap.add_argument('--dest', default='/tmp/osz_x')
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    packs = sorted(f for f in os.listdir(args.pool) if f.endswith('.osz'))
    ok = skip = fail = 0
    new_osu = 0

    for p in packs:
        sid = os.path.splitext(p)[0]
        d = os.path.join(args.dest, sid)
        if os.path.isdir(d):
            have_audio = any(x.lower().endswith(KEEP[1:]) for x in os.listdir(d))
            have_osu = any(x.lower().endswith('.osu') for x in os.listdir(d))
            if have_audio and have_osu:
                skip += 1
                continue
        os.makedirs(d, exist_ok=True)
        try:
            with zipfile.ZipFile(os.path.join(args.pool, p)) as z:
                for n in z.namelist():
                    if not n.lower().endswith(KEEP):
                        continue
                    base = os.path.basename(n)
                    if not base:
                        continue
                    with z.open(n) as src, open(os.path.join(d, base), 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    if base.endswith('.osu'):
                        new_osu += 1
            ok += 1
        except Exception as e:
            fail += 1
            print(f'  失败 {sid}: {str(e)[:60]}')

    print(f'解压 {ok} 新增 / {skip} 已存在 / {fail} 失败，.osu 共 +{new_osu}')
    total = sum(1 for x in os.listdir(args.dest) if os.path.isdir(os.path.join(args.dest, x)))
    print(f'{args.dest} 现有 {total} 个 set')


if __name__ == '__main__':
    main()
