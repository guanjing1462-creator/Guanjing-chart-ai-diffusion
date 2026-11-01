#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""package: 把出好的 .osu 和音频打成一个 .osz（osu 谱面包）。

用法：
  python3 tools/package.py --osu a.osu [b.osu ...] --audio "ref/INTERNET YAMERO.ogg" \
      --name "INTERNET YAMERO [AI v11]" -o out/v11/osz/
"""
import argparse
import os
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osu', nargs='+', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('-o', required=True)
    args = ap.parse_args()
    os.makedirs(args.o, exist_ok=True)
    path = os.path.join(args.o, f'{args.name}.osz')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(args.audio, os.path.basename(args.audio))
        for f in args.osu:
            z.write(f, os.path.basename(f))
    print(f'{path}  {os.path.getsize(path)/1e6:.1f}MB  {len(args.osu)} 张谱 + 音频')


if __name__ == '__main__':
    main()
