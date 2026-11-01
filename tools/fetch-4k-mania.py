#!/usr/bin/env python3
"""从 osu.direct 抓取 4K mania 谱面包（.osz），解压出 .osu + 音频。

只从 osu.direct 下载；搜索结果的 ChildrenBeatmaps 已带 CS/Mode，
所以能先筛「至少一张 4K mania 谱」再下载，省带宽。

用法：python3 tools/fetch-4k-mania.py [--max-bytes 1500000000] [--target 45]
输出：out/audio-dl/<SetID>/   +  out/audio-dl/_fetch_log.json
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile

UA = 'Mozilla/5.0'
BASE = 'https://osu.direct/api'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out', 'audio-dl')
LOG = os.path.join(OUT, '_fetch_log.json')

KEYWORDS = [
    '4k mania', 'jumpstream', 'stream 4k', 'tech 4k', 'jack 4k',
    'stamina 4k', 'LN 4k', '4k', 'mania 4k', '4k dan', 'handstream',
    'chordjack 4k', 'speed 4k', '4k ranked', 'osu!mania 4k',
]
AUDIO_EXT = ('.mp3', '.ogg', '.wav')
SMALL_KEEP = ('.osu',) + AUDIO_EXT


def curl(url, out=None, timeout=180):
    """返回 (http_code, nbytes)；out=None 时下载到内存返回 bytes。"""
    cmd = ['curl', '-sL', '-A', UA, '--max-time', str(timeout), url]
    if out:
        cmd += ['-o', out, '-w', '%{http_code} %{size_download}']
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            code, size = r.stdout.split()
            return int(code), int(size)
        except Exception:
            return 0, 0
    cmd += ['-w', '\n%{http_code}']
    r = subprocess.run(cmd, capture_output=True)
    body = r.stdout
    nl = body.rfind(b'\n')
    try:
        code = int(body[nl + 1:])
    except Exception:
        return 0, b''
    return code, body[:nl]


def search(kw, tries=2):
    for k in range(tries):
        code, body = curl(f'{BASE}/search?q={kw.replace(" ", "%20")}&ps=50')
        if code == 200 and body.startswith(b'['):
            try:
                return json.loads(body.decode('utf-8', 'replace'))
            except Exception:
                pass
        time.sleep(1.0 + k)
    return []


def is_4k_mania(entry):
    return any(c.get('Mode') == 3 and int(round(float(c.get('CS') or 0))) == 4
               for c in entry.get('ChildrenBeatmaps') or [])


def safe_name(s, n=60):
    s = re.sub(r'[\\/:*?"<>|\n\r\t]', '_', s or '')
    return s.strip()[:n] or 'untitled'


def extract_osz(path, set_id):
    """解压出 .osu 与音频，删掉视频/背景等；返回 (osu 数, 音频名列表, 总字节)。"""
    dst = os.path.join(OUT, str(set_id))
    os.makedirs(dst, exist_ok=True)
    osu_n, audios = 0, []
    try:
        zf = zipfile.ZipFile(path)
    except Exception as e:
        return 0, [], 0, f'zip 损坏 {e}'
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            low = name.lower()
            base = os.path.basename(name)
            if not base:
                continue
            if not low.endswith(SMALL_KEEP):
                continue                                  # 视频/背景图/音效全部跳过
            if info.file_size > 80 * 1024 * 1024:
                continue
            try:
                data = zf.read(info)
            except Exception:
                continue
            with open(os.path.join(dst, base), 'wb') as fh:
                fh.write(data)
            if low.endswith('.osu'):
                osu_n += 1
            else:
                audios.append(base)
    tot = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    return osu_n, audios, tot, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-bytes', type=int, default=1500 * 1000 * 1000)
    ap.add_argument('--target', type=int, default=48)
    ap.add_argument('--set-max-bytes', type=int, default=60 * 1000 * 1000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    log = json.load(open(LOG)) if os.path.exists(LOG) else {'ok': {}, 'fail': {}, 'skip': {}}
    for k in ('ok', 'fail', 'skip'):
        log.setdefault(k, {})

    # 1) 搜索 + 去重 + 4K 过滤
    cand = {}
    for kw in KEYWORDS:
        rs = search(kw)
        n4k = 0
        for e in rs:
            sid = e.get('SetID')
            if not sid or not is_4k_mania(e):
                continue
            n4k += 1
            if sid not in cand:
                cand[sid] = {
                    'set_id': sid,
                    'title': e.get('Title', ''),
                    'artist': e.get('Artist', ''),
                    'creator': e.get('Creator', ''),
                    'kw': kw,
                    'children': [(c['BeatmapID'], c.get('DiffName', ''), c.get('CS'),
                                  c.get('BPM'), c.get('HitLength'))
                                 for c in e['ChildrenBeatmaps']
                                 if c.get('Mode') == 3 and int(round(float(c.get('CS') or 0))) == 4],
                }
        print(f'  [search] {kw!r}: 结果 {len(rs)}，其中 4K mania {n4k}，累计候选 {len(cand)}', flush=True)

    print(f'\n候选 4K mania 谱面包: {len(cand)}', flush=True)

    # 已完成的 set 直接计入
    done_ids = {int(k) for k in log['ok']}
    total_bytes = sum(v.get('bytes', 0) for v in log['ok'].values())
    accept = [k for k, v in log['ok'].items() if v.get('accepted')]
    print(f'已有: 下载成功 {len(log["ok"])}，其中接受 {len(accept)}，累计 {total_bytes/1e6:.1f} MB', flush=True)

    order = sorted(cand, key=lambda s: (s in done_ids, s))
    for sid in order:
        if len(accept) >= a.target:
            print(f'\n达到目标 {a.target} 个接受谱面包，停止', flush=True)
            break
        if total_bytes >= a.max_bytes:
            print(f'\n达到字节上限 {a.max_bytes/1e6:.0f} MB，停止', flush=True)
            break
        if sid in done_ids:
            continue
        meta = cand[sid]
        tmp = f'/tmp/osz_{sid}.zip'
        code, size = curl(f'{BASE}/d/{sid}', out=tmp, timeout=240)
        if code != 200 or size < 20000:
            log['skip'][sid] = {'reason': f'下载失败 http={code} size={size}', **{
                'title': meta['title']}}
            print(f'  [skip] {sid} http={code} size={size}  {meta["title"][:40]}', flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        if size > a.set_max_bytes:
            log['skip'][sid] = {'reason': f'超过单包上限 {size/1e6:.1f}MB',
                                'title': meta['title']}
            print(f'  [skip] {sid} 太大 {size/1e6:.1f}MB', flush=True)
            os.remove(tmp)
            continue
        if total_bytes + size > a.max_bytes:
            log['skip'][sid] = {'reason': f'总字节将超限（+{size/1e6:.1f}MB）',
                                'title': meta['title']}
            os.remove(tmp)
            continue

        osu_n, audios, kept, err = extract_osz(tmp, sid)
        os.remove(tmp)
        if err or osu_n == 0:
            log['fail'][sid] = {'reason': err or '解压后没有 .osu', 'title': meta['title']}
            print(f'  [fail] {sid} {err or "无 .osu"}', flush=True)
            continue

        rec = {
            'set_id': sid, 'title': meta['title'], 'artist': meta['artist'],
            'creator': meta['creator'], 'bytes': size, 'kept_bytes': kept,
            'osu_n': osu_n, 'audios': audios, 'kw': meta['kw'],
            'children': meta['children'],
            'accepted': osu_n > 0,
        }
        log['ok'][str(sid)] = rec
        done_ids.add(sid)
        total_bytes += size
        if rec['accepted']:
            accept.append(sid)
        print(f'  [ok {len(accept)}/{a.target}] {sid} {size/1e6:5.1f}MB '
              f'osu={osu_n} audio={len(audios)} 累计 {total_bytes/1e6:6.1f}MB  '
              f'{meta["title"][:36]}', flush=True)
        json.dump(log, open(LOG, 'w'), ensure_ascii=False, indent=1)

    json.dump(log, open(LOG, 'w'), ensure_ascii=False, indent=1)
    print(f'\n完成：成功 {len(log["ok"])}（接受 {len(accept)}），失败 {len(log["fail"])}，'
          f'跳过 {len(log["skip"])}，总下载 {total_bytes/1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
