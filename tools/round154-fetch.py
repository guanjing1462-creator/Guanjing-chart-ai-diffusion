#!/usr/bin/env python3
"""round154 扩容下载：并发版（复用 tools/fetch-4k-mania.py 的搜索/筛选/解压）。

原脚本串行搜索 + 串行下载，osu.direct 单请求 5–20s，遇到一个慢请求（--max-time 180
再重试）会把整轮卡住十几分钟。这里只改调度，不改口径：

* 搜关键字：8 线程并发，单请求 --max-time 30、不重试（丢了就丢了，候选池足够大）；
* 下载：4 线程并发，单包 ≤80MB，本轮新增字节 ≤ --max-new-bytes（默认 2.4GB）；
* 预筛：搜索结果里没有「Mode=3 且 CS=4」子谱的 set 直接不下载（记为无 4K 谱）；
* 解压：直接调 fetch.extract_osz（只留 .osu + 音频），保持与旧 48 包同目录结构；
* 日志：沿用 out/audio-dl/_fetch_log.json 的 ok/fail/skip 结构，每完成一个就落盘。

用法：python3 tools/round154-fetch.py --new-target 62 --new-bytes 2400000000
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'f4k', os.path.join(ROOT, 'tools', 'fetch-4k-mania.py'))
F4K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F4K)

OUT = F4K.OUT
LOG = F4K.LOG
UA = F4K.UA
BASE = F4K.BASE

# 比原脚本多几个角度，尽量把候选池撑大（原 15 个关键字照抄，后面几个是补充）
EXTRA_KW = ['4k ranked', '4k stamina', '4k jumpstream', 'modcharts 4k',
            '4k ln', '4k jack', 'osu!mania 4k', '4k vsrg', '4k stream',
            '4k tech', '4k speed', '4k vibro']


def search_one(kw):
    url = f'{BASE}/search?q={kw.replace(" ", "%20")}&ps=50'
    cmd = ['curl', '-sL', '-A', UA, '--max-time', '30', url, '-w', '\n%{http_code}']
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=40)
        body = r.stdout
        nl = body.rfind(b'\n')
        code = int(body[nl + 1:])
    except Exception as e:
        return kw, 0, []
    if code != 200 or not body.startswith(b'['):
        return kw, code, []
    try:
        return kw, code, json.loads(body[:nl].decode('utf-8', 'replace'))
    except Exception:
        return kw, code, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-target', type=int, default=62)
    ap.add_argument('--new-bytes', type=int, default=2400 * 1000 * 1000)
    ap.add_argument('--set-max-bytes', type=int, default=80 * 1000 * 1000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--kw-workers', type=int, default=8)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    log = json.load(open(LOG, encoding='utf-8')) if os.path.exists(LOG) else {}
    for k in ('ok', 'fail', 'skip'):
        log.setdefault(k, {})
    done = {int(k) for k in log['ok']}
    old_bytes = sum(v.get('bytes', 0) for v in log['ok'].values())
    print(f'已有 ok {len(log["ok"])} 包 / 累计下载 {old_bytes/1e6:.1f}MB / '
          f'skip {len(log["skip"])} / fail {len(log["fail"])}', flush=True)

    kws = list(dict.fromkeys(F4K.KEYWORDS + EXTRA_KW))
    cand, pool_stats = {}, dict(results=0, not_4k=0, already=0)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.kw_workers) as ex:
        futs = [ex.submit(search_one, kw) for kw in kws]
        for fu in as_completed(futs):
            kw, code, rs = fu.result()
            n4k = 0
            for e in rs:
                pool_stats['results'] += 1
                sid = e.get('SetID')
                if not sid or not F4K.is_4k_mania(e):
                    pool_stats['not_4k'] += 1
                    continue
                n4k += 1
                if sid not in cand:
                    cand[sid] = dict(
                        set_id=sid, title=e.get('Title', ''),
                        artist=e.get('Artist', ''), creator=e.get('Creator', ''),
                        kw=kw,
                        children=[(c['BeatmapID'], c.get('DiffName', ''), c.get('CS'),
                                   c.get('BPM'), c.get('HitLength'))
                                  for c in e['ChildrenBeatmaps']
                                  if c.get('Mode') == 3
                                  and int(round(float(c.get('CS') or 0))) == 4])
            print(f'  [search] {kw!r} http={code} 结果 {len(rs)} 4K {n4k} '
                  f'候选累计 {len(cand)}  {time.time()-t0:.0f}s', flush=True)
    new_ids = [s for s in cand if s not in done]
    pool_stats['already'] = len(cand) - len(new_ids)
    print(f'\n候选 4K mania set {len(cand)}（已下载 {pool_stats["already"]}，'
          f'新增可下 {len(new_ids)}）；搜索用时 {time.time()-t0:.0f}s', flush=True)

    # ---- 并发下载 ----
    new_bytes = 0
    acc = [s for s, v in log['ok'].items() if v.get('accepted')]
    lock = threading.Lock()
    stop = threading.Event()
    t1 = time.time()

    def fetch_one(sid):
        meta = cand[sid]
        tmp = f'/tmp/osz154_{sid}.zip'
        try:
            code, size = F4K.curl(f'{BASE}/d/{sid}', out=tmp, timeout=300)
            if code != 200 or size < 20000:
                return sid, 'fail', f'下载失败 http={code} size={size}', size
            if size > a.set_max_bytes:
                return sid, 'skip', f'超过单包上限 {size/1e6:.1f}MB', size
            with lock:
                over = new_bytes + size > a.new_bytes
            if over:
                return sid, 'skip', f'本轮字节将超限（+{size/1e6:.1f}MB）', size
            osu_n, audios, kept, err = F4K.extract_osz(tmp, sid)
            if err or osu_n == 0:
                return sid, 'fail', err or '解压后没有 .osu', size
            return sid, 'ok', '', size
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    todo = new_ids[: a.new_target * 3]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        it = iter(todo)
        # 先撒满一批
        for _ in range(a.workers * 2):
            sid = next(it, None)
            if sid is None:
                break
            futs[ex.submit(fetch_one, sid)] = sid
        while futs:
            for fu in as_completed(list(futs)):
                sid = futs.pop(fu)
                try:
                    sid, kind, why, size = fu.result()
                except Exception as e:
                    print(f'  [err] {sid} {e}', flush=True)
                    break
                with lock:
                    if kind == 'ok':
                        new_bytes += size
                        dcur = os.path.join(OUT, str(sid))
                        kept = sum(os.path.getsize(os.path.join(dcur, f))
                                   for f in os.listdir(dcur))
                        acc.append(str(sid))
                        log['ok'][str(sid)] = dict(
                            set_id=sid, title=cand[sid]['title'],
                            artist=cand[sid]['artist'], creator=cand[sid]['creator'],
                            bytes=size, kept_bytes=kept,
                            osu_n=len([f for f in os.listdir(os.path.join(OUT, str(sid)))
                                       if f.lower().endswith('.osu')]),
                            audios=[f for f in os.listdir(os.path.join(OUT, str(sid)))
                                    if f.lower().endswith(F4K.AUDIO_EXT)],
                            kw=cand[sid]['kw'], children=cand[sid]['children'],
                            accepted=True, round154=True)
                        print(f'  [ok {len(acc)}] {sid} {size/1e6:5.1f}MB '
                              f'本轮 {new_bytes/1e6:6.1f}MB 累计 ok {len(log["ok"])} / '
                              f'{time.time()-t1:.0f}s  {cand[sid]["title"][:34]}', flush=True)
                    elif kind == 'skip':
                        log['skip'][str(sid)] = dict(reason=why, title=cand[sid]['title'])
                        print(f'  [skip] {sid} {why}  {cand[sid]["title"][:34]}', flush=True)
                    else:
                        log['fail'][str(sid)] = dict(reason=why, title=cand[sid]['title'])
                        print(f'  [fail] {sid} {why}', flush=True)
                    json.dump(log, open(LOG, 'w', encoding='utf-8'),
                              ensure_ascii=False, indent=1)
                    if len(acc) >= 48 + a.new_target or new_bytes >= a.new_bytes:
                        stop.set()
                if stop.is_set():
                    break
                sid2 = next(it, None)
                if sid2 is not None:
                    futs[ex.submit(fetch_one, sid2)] = sid2
                break
            if stop.is_set():
                for fu in list(futs):
                    fu.cancel()
                break

    json.dump(log, open(LOG, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    summary = dict(new_sets=len(acc) - 48, total_sets=len(acc),
                   new_bytes=int(new_bytes), old_bytes=int(old_bytes),
                   elapsed_s=round(time.time()-t0, 1), pool=pool_stats,
                   candidates=len(cand))
    json.dump(summary, open(os.path.join(
        ROOT, 'out', 'diffusion', 'round154-data', 'fetch-summary.json'), 'w',
        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n完成：本轮新增 {summary["new_sets"]} 包 / {new_bytes/1e6:.1f}MB；'
          f'ok 总 {len(log["ok"])}；skip {len(log["skip"])}；fail {len(log["fail"])}；'
          f'用时 {summary["elapsed_s"]:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
