#!/usr/bin/env python3
"""round154 耐心下载器：osu.direct 当前吞吐极不稳定（单连接 1–70KB/s、常返回 46 字节错误页）。

策略：多连接 + 断点续传 + 反复重试，尽量在给定时间内多拿几个 set。

* 队列：out/audio-dl/_fetch_log.json 已 ok 的 SetID 之外、且 `Mode=3 CS=4` 的候选；
* 每次 curl：`--retry 4 --retry-all-errors --retry-delay 2 -C - --max-time 240`，
  失败重试到 3 次；下载完用 zipfile 校验，损坏则删了重来；
* 解压复用 fetch-4k-mania.py 的 extract_osz（同目录结构）；
* 日志写 out/audio-dl/_fetch_log.json（ok/fail/skip 同旧结构）+ 单独的 dl154-run.log。

用法：python3 tools/round154-dl.py --minutes 18 --workers 6 --target-new 62
"""
import argparse
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
OUT, LOG = F4K.OUT, F4K.LOG
UA = F4K.UA
BASE = F4K.BASE

KW = ['4k mania', 'jumpstream', 'stream 4k', 'mania 4k', '4k dan', '4k stamina',
      'stamina 4k', 'tech 4k', 'osu!mania 4k', '4k stream', '4k tech', '4k vsrg',
      '4k speed', 'chordjack 4k', 'handstream', 'pattern 4k', 'density 4k',
      '4k insane', '4k expert', '4k underjoy', '4k double', '4k lv 20']


def search_all(log):
    cand, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(F4K.search, kw, 1): kw for kw in KW}
        for fu in as_completed(futs):
            kw = futs[fu]
            try:
                rs = fu.result()
            except Exception:
                rs = []
            for e in rs:
                sid = e.get('SetID')
                if not sid or sid in cand or not F4K.is_4k_mania(e):
                    continue
                cand[sid] = dict(
                    set_id=sid, title=e.get('Title', ''), artist=e.get('Artist', ''),
                    creator=e.get('Creator', ''), kw=kw,
                    children=[(c['BeatmapID'], c.get('DiffName', ''), c.get('CS'),
                               c.get('BPM'), c.get('HitLength'))
                              for c in e['ChildrenBeatmaps']
                              if c.get('Mode') == 3
                              and int(round(float(c.get('CS') or 0))) == 4])
            print(f'  [search] {kw!r} {len(rs)} 候选 {len(cand)} {time.time()-t0:.0f}s',
                  flush=True)
    return cand


def download(sid, path):
    """下载到 path，成功返回 (bytes, None)，失败返回 (0, reason)。"""
    n429 = 0
    for attempt in range(6):
        cmd = ['curl', '-sL', '-A', UA, '--retry', '4', '--retry-delay', '2',
               '--retry-all-errors', '-C', '-', '--max-time', '240',
               '-o', path, '-w', '%{http_code} %{size_download}', f'{BASE}/d/{sid}']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=320)
            out = (r.stdout or '').strip().split()
            code = int(out[-2]) if len(out) >= 2 else 0
        except Exception:
            code = 0
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        if code == 200 and sz > 20000:
            try:
                zipfile.ZipFile(path).testzip()
                return sz, None
            except Exception:
                os.remove(path)
                continue
        if code == 429:
            n429 += 1
            time.sleep(12 + 8 * n429)
            continue
        if code == 503:
            return 0, '503 镜像缺档'
        if code == 404:
            return 0, '404 无此 set'
        if sz > 200 and sz < 5000:            # 46 字节错误页之类
            os.remove(path)
        time.sleep(1 + attempt)
    return 0, ('429 限流退避后仍失败' if n429 else '多次重试仍失败（超时/连接中断）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=18)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--target-new', type=int, default=62)
    ap.add_argument('--set-max-bytes', type=int, default=80 * 1000 * 1000)
    ap.add_argument('--cand-file', default='', help='候选池 json（跳过搜索，避开 429）')
    a = ap.parse_args()

    log = json.load(open(LOG, encoding='utf-8'))
    for k in ('ok', 'fail', 'skip'):
        log.setdefault(k, {})
    done = {int(k) for k in log['ok']}
    base_acc = sum(1 for v in log['ok'].values() if v.get('accepted'))
    old_bytes = sum(v.get('bytes', 0) for v in log['ok'].values())
    print(f'起点：ok {len(log["ok"])} 包（接受 {base_acc}）/ 累计 {old_bytes/1e6:.0f}MB',
          flush=True)

    if a.cand_file and os.path.exists(a.cand_file):
        raw = json.load(open(a.cand_file, encoding='utf-8'))
        cand = {int(k): v for k, v in raw.items()}
        print(f'候选池来自 {a.cand_file}：{len(cand)} 个', flush=True)
    else:
        cand = search_all(log)
    queue = [s for s in cand if s not in done]
    print(f'候选 {len(cand)}，可下 {len(queue)}\n', flush=True)

    lock = threading.Lock()
    got = [0]
    nbytes = [0]

    fails = {}
    t_end = time.time() + a.minutes * 60
    stop = threading.Event()

    def work(sid):
        if stop.is_set() or time.time() > t_end:
            return sid, None
        meta = cand[sid]
        path = f'/tmp/dl154_{sid}.zip'
        size, err = download(sid, path)
        if err:
            if os.path.exists(path):
                os.remove(path)
            return sid, ('fail', err, meta['title'])
        if size > a.set_max_bytes:
            os.remove(path)
            return sid, ('skip', f'超过单包上限 {size/1e6:.1f}MB', meta['title'])
        osu_n, audios, kept, err2 = F4K.extract_osz(path, sid)
        os.remove(path)
        if err2 or osu_n == 0:
            return sid, ('fail', err2 or '解压无 .osu', meta['title'])
        return sid, ('ok', size, kept, osu_n, audios)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        it = iter(queue)
        for _ in range(a.workers * 2):
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
                        nbytes[0] += size
                        got[0] += 1
                        log['ok'][str(sid)] = dict(
                            set_id=sid, title=cand[sid]['title'],
                            artist=cand[sid]['artist'], creator=cand[sid]['creator'],
                            bytes=size, kept_bytes=kept, osu_n=osu_n,
                            audios=audios, kw=cand[sid]['kw'],
                            children=cand[sid]['children'], accepted=True,
                            round154=True)
                        print(f'  [ok {got[0]}/{a.target_new}] {sid} '
                              f'{size/1e6:5.1f}MB 本轮 {nbytes[0]/1e6:6.1f}MB  '
                              f'{cand[sid]["title"][:34]}', flush=True)
                    elif res[0] == 'skip':
                        log['skip'][str(sid)] = dict(reason=res[1], title=res[2])
                        print(f'  [skip] {sid} {res[1]}', flush=True)
                    else:
                        log['fail'][str(sid)] = dict(reason=res[1], title=res[2])
                        fails[res[1]] = fails.get(res[1], 0) + 1
                        print(f'  [fail] {sid} {res[1]}', flush=True)
                    json.dump(log, open(LOG, 'w', encoding='utf-8'),
                              ensure_ascii=False, indent=1)
                    if got[0] >= a.target_new:
                        stop.set()
                if stop.is_set() or time.time() > t_end:
                    stop.set()
                    break
                s2 = next(it, None)
                if s2 is not None:
                    futs[ex.submit(work, s2)] = s2
                break
            if stop.is_set():
                for f in list(futs):
                    f.cancel()
                break

    json.dump(log, open(LOG, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n本轮新增 {got[0]} 包 / {nbytes[0]/1e6:.1f}MB；'
          f'ok 总 {len(log["ok"])}；失败原因 {fails}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
