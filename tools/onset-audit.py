#!/usr/bin/env python3
"""对全部谱做采音审计：中位偏差 / 空音率 / 强起音覆盖。"""
import glob, json, os
import numpy as np, librosa, importlib.util
spec = importlib.util.spec_from_file_location('ac', 'tools/audit-chart.py')
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

SONG = {
 'Frums_SuperMix': 'ref/multisong/Frums_SuperMix.wav',
 'Heart_Fraction': 'ref/multisong/Heart Fraction.ogg',
 'Plum_FallenEra': 'ref/multisong/Plum-FallenEra.mp3',
 'REDALiCE_x_t_pazolite___Xterfusion': 'ref/multisong/REDALiCE x t+pazolite - Xterfusion.mp3',
 '_______________Flash_Me_Back': 'ref/multisong/かめりあ - Flash Me Back.mp3',
 'seatrus___purifikation': 'ref/multisong/seatrus - purifikation.ogg',
}

def detect(path):
    y, sr = librosa.load(path, sr=22050, mono=True, duration=240)
    hop = 256; fps = sr / hop
    env = np.maximum(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), 0)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    def pp(e, a, b, c, d, dl, w):
        return librosa.util.peak_pick(x=e, pre_max=a, post_max=b, pre_avg=c,
                                      post_avg=d, delta=dl, wait=w)
    pk = set(pp(env, 3, 3, 8, 8, 0.12, 2))
    for lo, hi in [(0, 200), (200, 2000), (2000, 8000)]:
        m = (fr >= lo) & (fr < hi)
        if not m.any(): continue
        be = np.maximum(np.diff(np.log1p(S[m].sum(0)), prepend=0), 0)
        pk |= set(pp(be, 2, 2, 5, 5, 0.55 * (float(np.median(be)) + 1e-9), 1))
    ons = np.array(sorted(pk)) * 1000.0 / fps
    st = np.array([env[min(len(env) - 1, int(t / 1000 * fps))] for t in ons])
    strong = ons[st >= np.percentile(st, 90)]
    return ons, strong

def main():
    res = {}
    for song, audio in SONG.items():
        ons, strong = detect(audio)
        for f in sorted(glob.glob(f'out/all/{song}--*.osu')):
            tier = f.split('--')[-1][:-4]
            n, _ = ac.parse_osu(f)
            ts = np.array([t for t, _, _ in n]); ts = ts[ts <= ons[-1]]
            if not len(ts): continue
            idx = np.clip(np.searchsorted(ons, ts), 1, len(ons) - 1)
            d = np.minimum(np.abs(ts - ons[idx]), np.abs(ts - ons[idx - 1]))
            cov = float(np.mean([np.min(np.abs(ts - s)) <= 40 for s in strong]) * 100)
            res[(song, tier)] = dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                                     empty=float((d > 60).mean() * 100), cover=cov)
            print(f'{song[:20]:20s} {tier:8s} 中位 {np.median(d):5.1f}ms  '
                  f'空音 {(d > 60).mean() * 100:5.2f}%  强覆盖 {cov:5.1f}%', flush=True)
    json.dump({f'{k[0]}|{k[1]}': v for k, v in res.items()}, open('out/multisong/onset-audit.json', 'w'), indent=1)
    import numpy as np2
    med = np2.mean([v['median'] for v in res.values()])
    emp = np2.mean([v['empty'] for v in res.values()])
    cov = np2.mean([v['cover'] for v in res.values()])
    print(f'\n36 张平均：中位偏差 {med:.1f}ms  空音率 {emp:.2f}%  强起音覆盖 {cov:.1f}%')

if __name__ == '__main__':
    main()
