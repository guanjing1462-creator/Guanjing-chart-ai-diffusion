#!/usr/bin/env python3
"""把每首歌的六档谱和音频打成一个 .osz（osu! 的谱面包）。"""
import glob, os, re, shutil, zipfile
SONGS = {
 'Frums_SuperMix': 'ref/multisong/Frums_SuperMix.wav',
 'Heart_Fraction': 'ref/multisong/Heart Fraction.ogg',
 'Plum_FallenEra': 'ref/multisong/Plum-FallenEra.mp3',
 'REDALiCE_x_t_pazolite___Xterfusion': 'ref/multisong/REDALiCE x t+pazolite - Xterfusion.mp3',
 '_______________Flash_Me_Back': 'ref/multisong/かめりあ - Flash Me Back.mp3',
 'seatrus___purifikation': 'ref/multisong/seatrus - purifikation.ogg',
}
NICE = {
 'Frums_SuperMix': 'Frums - SuperMix',
 'Heart_Fraction': 'Heart Fraction',
 'Plum_FallenEra': 'Plum - FallenEra',
 'REDALiCE_x_t_pazolite___Xterfusion': 'REDALiCE x t+pazolite - Xterfusion',
 '_______________Flash_Me_Back': 'かめりあ - Flash Me Back',
 'seatrus___purifikation': 'seatrus - purifikation',
}
os.makedirs('out/osz', exist_ok=True)
for song, audio in SONGS.items():
    charts = sorted(glob.glob(f'out/tiers-v17/{song}--*.osu'))
    if not charts: print('跳过', song); continue
    aud_name = os.path.basename(audio)
    title = NICE[song]
    zpath = f'out/osz/{title}.osz'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for c in charts:
            txt = open(c, encoding='utf-8').read()
            # AudioFilename 必须与包内音频名一致
            txt = re.sub(r'AudioFilename:.*', f'AudioFilename: {aud_name}', txt, count=1)
            # 统一 Title（方便 osu! 里归为一组）
            txt = re.sub(r'Title:.*', f'Title: {title.split(" - ")[-1]}', txt, count=1)
            txt = re.sub(r'Artist:.*', f'Artist: {title.split(" - ")[0]}', txt, count=1)
            z.writestr(os.path.basename(c), txt)
        z.write(audio, aud_name)
    print(f'{zpath}  {os.path.getsize(zpath)/1e6:.1f}MB  谱 {len(charts)} 张 + {aud_name}')
