#!/usr/bin/env python3
"""把 out/all 里的谱按歌曲打成大 osz。"""
import glob, os, re, zipfile
SONGS = {
 'Frums_SuperMix': ('ref/multisong/Frums_SuperMix.mp3', 'Frums', 'SuperMix'),
 'Heart_Fraction': ('ref/multisong/Heart Fraction.ogg', 'Heart', 'Heart Fraction'),
 'Plum_FallenEra': ('ref/multisong/Plum-FallenEra.mp3', 'Plum', 'FallenEra'),
 'REDALiCE_x_t_pazolite___Xterfusion': ('ref/multisong/REDALiCE x t+pazolite - Xterfusion.mp3', 'REDALiCE x t+pazolite', 'Xterfusion'),
 '_______________Flash_Me_Back': ('ref/multisong/かめりあ - Flash Me Back.mp3', 'かめりあ', 'Flash Me Back'),
 'seatrus___purifikation': ('ref/multisong/seatrus - purifikation.ogg', 'seatrus', 'purifikation'),
}
os.makedirs('out/osz', exist_ok=True)
tot = 0
for song, (audio, artist, title) in SONGS.items():
    charts = sorted(glob.glob(f'out/all/{song}--*.osu'))
    if not charts:
        continue
    aud_name = os.path.basename(audio)
    zpath = f'out/osz/GJ {title} - all.osz'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for c in charts:
            txt = open(c, encoding='utf-8').read()
            txt = re.sub(r'AudioFilename:.*', f'AudioFilename: {aud_name}', txt, count=1)
            txt = re.sub(r'Title:.*', f'Title: {title}', txt, count=1)
            txt = re.sub(r'Artist:.*', f'Artist: {artist}', txt, count=1)
            z.writestr(os.path.basename(c), txt)
        z.write(audio, aud_name)
    tot += len(charts)
    print(f'{zpath}: {len(charts)} 谱, {os.path.getsize(zpath)//1024} KB', flush=True)
print('合计', tot, '谱')
