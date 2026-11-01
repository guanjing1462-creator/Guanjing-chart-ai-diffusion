#!/usr/bin/env python3
"""构建 21 维控制特征库（Mug-Diffusion 条件接口的等价物）。

    python3 tools/build-feature-db.py out/charts -o out/feature-db.jsonl

对 out/charts/ 里每张 4K 谱算：
  - minacalc 8 维（overall + 7 技能，多进程，含 rate 增强所需的口径）
  - ln_ratio（LN 数/总音符数）与 rc/hb/ln 三 bool（<0.1 / 中间 / >0.4，webui 语义）
  - 7 个 pattern bool（该技能分与最高技能分差 ≤1 → 1，ds.py 语义）
  - rank_status 占位 'ranked'（osu.direct /api/osu 不带 set 状态，后续可补）

续传：已写入输出文件的 id 跳过。下载任务边跑边重跑本脚本即可。
"""
import argparse
import ctypes
import glob
import json
import os
from multiprocessing import Pool

SKILLS = ['overall', 'stream', 'jumpstream', 'handstream',
          'stamina', 'jackspeed', 'chordjack', 'technical']
PATTERNS = ['stream', 'jumpstream', 'handstream',
            'jackspeed', 'chordjack', 'technical', 'stamina']

_LIB = None


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.mc_calc.restype = ctypes.c_int
    lib.mc_calc.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint),
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


def parse_osu(path):
    """→ (keys, [(time_ms, mask)], n_ln, n_notes)。LN 尾不计音符。"""
    keys = 4
    rows = {}
    n_ln = 0
    sec = None
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line == '[TimingPoints]':
                sec = 'tp'
                continue
            if line == '[HitObjects]':
                sec = 'ho'
                continue
            if line.startswith('['):
                sec = None
                continue
            if sec != 'ho' or not line or line.startswith('//'):
                continue
            p = line.split(',')
            if len(p) < 5:
                continue
            try:
                t = int(float(p[2]))
                x = int(p[0])
                is_ln = len(p) > 5 and p[5].split(':')[0] not in ('0', '')
            except ValueError:
                continue
            col = x * keys // 512
            if not (0 <= col < keys):
                continue
            if is_ln:
                n_ln += 1
            m = 1 << col
            rows[t] = rows.get(t, 0) | m
    return keys, sorted(rows.items()), n_ln, n_ln + sum(bin(m).count('1') for m in rows.values())


def calc(lib, rows, rate=1.0):
    n = len(rows)
    if n == 0:
        return None
    arr_t = (ctypes.c_double * n)(*[t / 1000.0 for t, _ in rows])   # MinaCalc 要秒
    arr_m = (ctypes.c_uint * n)(*[m for _, m in rows])
    out = (ctypes.c_float * 8)()
    got = lib.mc_calc(arr_t, arr_m, n, rate, 0.93, out)
    if got <= 0:      # 返回值是技能维数（8），不是错误码
        return None
    return dict(zip(SKILLS, [round(float(out[i]), 2) for i in range(min(8, got))]))


def init_worker(lib_path):
    global _LIB
    _LIB = load_lib(lib_path)


def process_one(path):
    try:
        keys, rows, n_ln, n_notes = parse_osu(path)
        if keys != 4 or len(rows) < 20 or n_notes < 50:
            return None
        msd = calc(_LIB, rows)
        if not msd or msd.get('overall', 0) <= 0:
            return None
        ln_ratio = round(n_ln / max(1, n_notes), 3)
        feats = {
            'id': os.path.splitext(os.path.basename(path))[0],
            'n_rows': len(rows),
            'n_notes': n_notes,
            'n_ln': n_ln,
            'ln_ratio': ln_ratio,
            'rc': int(ln_ratio < 0.1),
            'hb': int(0.1 <= ln_ratio <= 0.4),
            'ln': int(ln_ratio > 0.4),
            'rank_status': 'ranked',
            'ett': round(msd['overall'], 2),
            **{f'{k}_ett': round(v, 2) for k, v in msd.items()},
        }
        top = max(msd[k] for k in PATTERNS if k != 'stamina')
        for k in PATTERNS:
            feats[k] = int(top - msd[k] <= 1)
        feats['durSec'] = round((rows[-1][0] - rows[0][0]) / 1000, 1)
        feats['nps'] = round(n_notes / max(0.1, feats['durSec']), 2)
        return feats
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('charts')
    ap.add_argument('-o', '--out', default='out/feature-db.jsonl')
    ap.add_argument('--lib', default=os.environ.get('MINACALC_LIB', './libminacalc.so'))
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    if not os.path.exists(args.lib):
        sys_exit = f'找不到 MinaCalc 库：{args.lib}'
        raise SystemExit(sys_exit)

    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['id'])
                except Exception:
                    pass

    files = sorted(glob.glob(os.path.join(args.charts, '*.osu')))
    todo = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in done_ids]
    print(f'谱面 {len(files)}，已有特征 {len(done_ids)}，待算 {len(todo)}')

    if not todo:
        return

    with Pool(args.workers, initializer=init_worker, initargs=(args.lib,)) as pool:
        made = 0
        with open(args.out, 'a') as f:
            for i, res in enumerate(pool.imap_unordered(process_one, todo, chunksize=16)):
                if res is not None:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
                    made += 1
                if (i + 1) % 500 == 0:
                    print(f'  {i+1}/{len(todo)}  写出 {made}')
        print(f'完成：新增 {made} 条 → {args.out}')


if __name__ == '__main__':
    main()
