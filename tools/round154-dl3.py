#!/usr/bin/env python3
"""从本地已缓存的搜索响应（/tmp/*.json）建候选池，然后只下载（不搜索）。

osu.direct 从 01:31 起对本 IP 返回 429（搜索和下载都限流），搜索拿不到新结果。
但 01:31 之前我们保存过几份搜索响应，里面已经带 4K mania 子谱信息，足够当下载队列。
"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'f4k', os.path.join(ROOT, 'tools', 'fetch-4k-mania.py'))
F4K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F4K)
LOG = F4K.LOG
BASE, UA = F4K.BASE, F4K.UA

CACHE = ['/tmp/probe.json', '/tmp/p2.json', '/tmp/s_4k.json',
         '/tmp/s_stream_4k.json', '/tmp/s_mania_4k.json', '/tmp/cand154.json']


def load_pool():
    log = json.load(open(LOG, encoding='utf-8'))
    done = {int(k) for k in log['ok']}
    pool = {}
    for p in CACHE:
        if not os.path.exists(p) or os.path.getsize(p) < 1000:
            continue
        try:
            rs = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        if isinstance(rs, dict):
            rs = list(rs.values())
        for e in rs:
            sid = e.get('SetID') or e.get('set_id')
            if not sid or int(sid) in pool or int(sid) in done:
                continue
            if not F4K.is_4k_mania(e):
                continue
            pool[int(sid)] = dict(
                set_id=int(sid), title=e.get('Title', ''), artist=e.get('Artist', ''),
                creator=e.get('Creator', ''), kw='cached-search',
                children=[(c['BeatmapID'], c.get('DiffName', ''), c.get('CS'),
                           c.get('BPM'), c.get('HitLength'))
                          for c in e['ChildrenBeatmaps'] if c.get('Mode') == 3
                          and int(round(float(c.get('CS') or 0))) == 4])
    return pool, log


def dl_one(sid, path):
    n429 = 0
    for attempt in range(4):
        cmd = ['curl', '-sL', '-A', UA, '--retry', '2', '--retry-delay', '3',
               '-C', '-', '--max-time', '180', '-o', path,
               '-w', '%{http_code} %{size_download}', f'{BASE}/d/{sid}']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            parts = (r.stdout or '').strip().split()
            code = int(parts[-2]) if len(parts) >= 2 else 0
        except Exception:
            code = 0
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        if code == 200 and sz > 20000:
            try:
                zipfile.ZipFile(path).testzip()
                return sz, None
            except Exception:
                return 0, 'zip 校验失败（多次中断）'
        if code == 429:
            n429 += 1
            time.sleep(8 + 6 * n429)
            continue
        if code == 503:
            return 0, '503 镜像缺档'
        if code == 404:
            return 0, '404 无此 set'
        time.sleep(2)
    return 0, ('429 限流退避后仍失败' if n429 else '重试仍失败（超时/中断）')


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    pool, log = load_pool()
    print(f'缓存候选（4K mania、未下载）：{len(pool)} 个', flush=True)
    if not pool:
        return 1
    t_end = time.time() + minutes * 60
    got, nb, fails = [0], [0], {}
    lock = threading.Lock()

    def work(sid):
        if time.time() > t_end:
            return sid, None
        path = f'/tmp/dl154_{sid}.zip'
        size, err = dl_one(sid, path)
        if err:
            if os.path.exists(path):
                os.remove(path)
            return sid, ('fail', err, pool[sid]['title'])
        if size > 80 * 1000 * 1000:
            os.remove(path)
            return sid, ('skip', f'超过单包上限 {size/1e6:.1f}MB', pool[sid]['title'])
        osu_n, audios, kept, err2 = F4K.extract_osz(path, sid)
        os.remove(path)
        if err2 or osu_n == 0:
            return sid, ('fail', err2 or '解压无 .osu', pool[sid]['title'])
        return sid, ('ok', size, kept, osu_n, audios)

    q = sorted(pool)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        it = iter(q)
        for _ in range(workers * 2):
            s = next(it, None)
            if s is None:
                break
            futs[ex.submit(work, s)] = s
        while futs:
            for fu in as_completed(list(futs)):
                sid = futs.pop(fu)
                try:
                    sid, res = fu.result()
                except Exception as e:
                    res = ('fail', f'异常 {e}', '')
                if res is None:
                    break
                with lock:
                    if res[0] == 'ok':
                        _, size, kept, osu_n, audios = res
                        nb[0] += size
                        got[0] += 1
                        log['ok'][str(sid)] = dict(
                            set_id=sid, title=pool[sid]['title'],
                            artist=pool[sid]['artist'], creator=pool[sid]['creator'],
                            bytes=size, kept_bytes=kept, osu_n=osu_n, audios=audios,
                            kw=pool[sid]['kw'], children=pool[sid]['children'],
                            accepted=True, round154=True)
                        print(f'  [ok {got[0]}] {sid} {size/1e6:5.1f}MB '
                              f'本轮 {nb[0]/1e6:6.1f}MB {pool[sid]["title"][:32]}',
                              flush=True)
                    elif res[0] == 'skip':
                        log['skip'][str(sid)] = dict(reason=res[1], title=res[2])
                        print(f'  [skip] {sid} {res[1]}', flush=True)
                    else:
                        log['fail'][str(sid)] = dict(reason=res[1], title=res[2])
                        fails[res[1]] = fails.get(res[1], 0) + 1
                        print(f'  [fail] {sid} {res[1]}', flush=True)
                    json.dump(log, open(LOG, 'w', encoding='utf-8'),
                              ensure_ascii=False, indent=1)
                if time.time() > t_end:
                    break
                s2 = next(it, None)
                if s2 is not None:
                    futs[ex.submit(work, s2)] = s2
                break
            if time.time() > t_end:
                for f in list(futs):
                    f.cancel()
                break
    json.dump(log, open(LOG, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n新增 {got[0]} 包 / {nb[0]/1e6:.1f}MB；失败分布 {fails}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
