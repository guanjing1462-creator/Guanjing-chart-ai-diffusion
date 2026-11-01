#!/usr/bin/env python3
"""round154 下载（二）：串行搜索（增量存盘，抗 429）+ 2 连接下载（续传），限时运行。"""
import argparse, importlib.util, json, os, subprocess, sys, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('f4k', os.path.join(ROOT,'tools','fetch-4k-mania.py'))
F4K = importlib.util.module_from_spec(spec); spec.loader.exec_module(F4K)
OUT, LOG = F4K.OUT, F4K.LOG
BASE, UA = F4K.BASE, F4K.UA
POOL = '/tmp/cand154.json'
KW = ['4k mania','jumpstream','stream 4k','mania 4k','4k dan','4k stamina',
      'tech 4k','osu!mania 4k','4k stream','4k tech','chordjack 4k','handstream']

def search_pool(done):
    pool = json.load(open(POOL, encoding='utf-8')) if os.path.exists(POOL) else {}
    for kw in KW:
        for t in range(2):
            rs = F4K.search(kw, 1)
            if rs: break
            time.sleep(10 + 10*t)
        n = 0
        for e in rs:
            sid = e.get('SetID')
            if not sid or str(sid) in pool or not F4K.is_4k_mania(e): continue
            pool[str(sid)] = dict(set_id=sid, title=e.get('Title',''), artist=e.get('Artist',''),
                creator=e.get('Creator',''), kw=kw,
                children=[(c['BeatmapID'], c.get('DiffName',''), c.get('CS'), c.get('BPM'), c.get('HitLength'))
                          for c in e['ChildrenBeatmaps'] if c.get('Mode')==3 and int(round(float(c.get('CS') or 0)))==4])
            n += 1
        json.dump(pool, open(POOL,'w'), ensure_ascii=False)
        print(f'  [search] {kw!r} 结果 {len(rs)} +{n} 池 {len(pool)}', flush=True)
    return {int(k): v for k, v in pool.items()}

def dl_one(sid, path):
    n429 = 0
    for attempt in range(5):
        cmd = ['curl','-sL','-A',UA,'--retry','3','--retry-delay','3','--retry-all-errors',
               '-C','-','--max-time','200','-o',path,'-w','%{http_code} %{size_download}',
               f'{BASE}/d/{sid}']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=260)
            parts = (r.stdout or '').strip().split()
            code = int(parts[-2]) if len(parts) >= 2 else 0
        except Exception:
            code = 0
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        if code == 200 and sz > 20000:
            try:
                zipfile.ZipFile(path).testzip(); return sz, None
            except Exception:
                os.remove(path); continue
        if code == 429:
            n429 += 1; time.sleep(10 + 10*n429); continue
        if code == 503: return 0, '503 镜像缺档'
        if code == 404: return 0, '404 无此 set'
        if 0 < sz < 5000:
            os.remove(path)
        time.sleep(2)
    return 0, ('429 限流退避后仍失败' if n429 else '重试仍失败（超时/中断）')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=13)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--set-max-bytes', type=int, default=80*1000*1000)
    a = ap.parse_args()
    log = json.load(open(LOG, encoding='utf-8'))
    for k in ('ok','fail','skip'): log.setdefault(k, {})
    done = {int(k) for k in log['ok']}
    t_end = time.time() + a.minutes*60
    pool = search_pool(done)
    q = [s for s in pool if s not in done]
    # 先下已有半成品的（续传省流量），再按 set 号
    q.sort(key=lambda s: (not os.path.exists(f'/tmp/dl154_{s}.zip'), s))
    print(f'池 {len(pool)} / 可下 {len(q)} / 限时 {a.minutes:.0f} 分钟', flush=True)
    got = [0]; nb = [0]; fails = {}; stop = threading.Event()
    lock = threading.Lock()
    def work(sid):
        if time.time() > t_end: return sid, None
        path = f'/tmp/dl154_{sid}.zip'
        size, err = dl_one(sid, path)
        if err:
            if os.path.exists(path): os.remove(path)
            return sid, ('fail', err, pool[sid]['title'])
        if size > a.set_max_bytes:
            os.remove(path); return sid, ('skip', f'超过单包上限 {size/1e6:.1f}MB', pool[sid]['title'])
        osu_n, audios, kept, err2 = F4K.extract_osz(path, sid)
        os.remove(path)
        if err2 or osu_n == 0: return sid, ('fail', err2 or '解压无 .osu', pool[sid]['title'])
        return sid, ('ok', size, kept, osu_n, audios)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {}; it = iter(q)
        for _ in range(a.workers*2):
            s = next(it, None)
            if s is None: break
            futs[ex.submit(work, s)] = s
        while futs:
            for fu in as_completed(list(futs)):
                sid = futs.pop(fu)
                try: sid, res = fu.result()
                except Exception as e: res = ('fail', f'异常 {e}', '')
                if res is None: break
                with lock:
                    if res[0] == 'ok':
                        _, size, kept, osu_n, audios = res
                        nb[0] += size; got[0] += 1
                        log['ok'][str(sid)] = dict(set_id=sid, title=pool[sid]['title'],
                            artist=pool[sid]['artist'], creator=pool[sid]['creator'],
                            bytes=size, kept_bytes=kept, osu_n=osu_n, audios=audios,
                            kw=pool[sid]['kw'], children=pool[sid]['children'],
                            accepted=True, round154=True)
                        print(f'  [ok {got[0]}] {sid} {size/1e6:5.1f}MB 本轮 {nb[0]/1e6:6.1f}MB '
                              f'{pool[sid]["title"][:32]}', flush=True)
                    elif res[0] == 'skip':
                        log['skip'][str(sid)] = dict(reason=res[1], title=res[2]); print(f'  [skip] {sid} {res[1]}', flush=True)
                    else:
                        log['fail'][str(sid)] = dict(reason=res[1], title=res[2])
                        fails[res[1]] = fails.get(res[1],0)+1
                        print(f'  [fail] {sid} {res[1]}', flush=True)
                    json.dump(log, open(LOG,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
                if time.time() > t_end:
                    stop.set(); break
                s2 = next(it, None)
                if s2 is not None: futs[ex.submit(work, s2)] = s2
                break
            if stop.is_set():
                for f in list(futs): f.cancel()
                break
    json.dump(log, open(LOG,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n本轮新增 {got[0]} 包 / {nb[0]/1e6:.1f}MB；失败 {fails}', flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
