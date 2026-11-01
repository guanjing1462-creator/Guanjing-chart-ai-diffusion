"""Reproducible V16 repair release. No modification of previous releases."""
import hashlib, json, os, pathlib, subprocess, sys, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'out/v37-fillup'; OUT.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT)
env=dict(os.environ,OMP_NUM_THREADS='4')
subprocess.run([sys.executable,'tools/test-pattern-sampler.py'],check=True,env=env)
tiers=[('Rice','rice/technical/mid','rice','speed','0.1','out/packs/corpus-speed.json','0.5',0.010,18,0.35,0.92,1.15),
       ('Stream','rice/stream/high','stream','speed','0','out/packs/corpus-speed.json','0.5',0.010,18,0.35,0.92,1.15),
       ('Tech','rice/technical/high','tech','tech','1','out/packs/corpus-tech.json','0.5',0.025,20,0.55,0.88,1.25),
       ('Jack','rice/technical/high','tech','jack','1','out/packs/corpus-jack.json','0.5',0.100,35,0.30,0.30,1.00),
       ('LN','ln/technical/high','ln','tech','1.2','out/packs/corpus-tech.json','0.5',0.025,25,0.55,0.88,1.20),
       ('Stamina','rice/technical/high','tech','stamina','1','out/packs/corpus-stamina.json','0.5',0.018,12,1.00,0.85,2.00)]
records=[]
for name,style,cls,ptype,boost,corpus,spacing,jtarget,tpct,cprob,fprob,sboost in tiers:
    dest=OUT/f'GJ-V37-{name}.osu'
    cmd=[sys.executable,'tools/paint-gj.py','--analysis','out/final14-s50/analysis-fixed.json','--audio','ref/INTERNET YAMERO.ogg','--lexicon','out/rhythm-lexicon16.json','--style',style,'--style-class',cls,'--ptype',ptype,'--pattern-corpus',corpus,'--fill-spacing',str(spacing),'--jack-target',str(jtarget),'--fill-thr-pct',str(tpct),'--fill-chord-p',str(cprob),'--fill-prob',str(fprob),'--size-boost',str(sboost),'--disjoint-p',('0.30' if name in ('Jack','Stamina') else '0.0'),'--npr-mode',('prop' if name in ('Stamina','Jack') else 'quota'),'--strict','--dps','12.6','--chord-boost',boost,'--model','out/placement3k.pt','--song-id','INTERNET_YAMERO','--target-ett','18','--creator','GJ','--version',f'GJ V37 {name}','--seed','22','-o',str(dest)]
    if name=='LN': cmd+=['--ln-cap','0.18']
    with (OUT/f'{name.lower()}.log').open('w') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
    records.append(dict(tier=name,command=cmd,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    print(f'{name}: generated',flush=True)
subprocess.run([sys.executable,'tools/check-chart.py',*[str(OUT/f'GJ-V37-{r[0]}.osu') for r in tiers]],check=True,env=env)
manifest={'version':'V37','purpose':'V20: pack gap distribution drives note spacing (fill-spacing), so rhythm and hand movement come from the same pack statistics.', 'charts':records,'inputs':{}}
for rel in ['tools/paint-gj.py','tools/test-pattern-sampler.py','tools/release-v16.py','out/pattern-corpus.json','out/placement3k.pt','out/final14-s50/analysis-fixed.json','ref/INTERNET YAMERO.ogg']:
    manifest['inputs'][rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
readme='''# GJ V37 · 修复验证版

在 V16 基础上加「目标保持器」：把**累计**单音 jack 率、手位移与人类同类谱对比，实时改采样权重；jack 超目标时直接禁止同列候选。

- 软权重对「该上下文里 jack 是唯一候选」无效，实测 0.324→0.326；因此加了硬上限。
- 手位移用指数惩罚 exp(-0.35·(move-目标))，Stream 从 1.61 调到 1.41（人类 1.44）。
- 严格采音与断言沿用 V16：每条严格模式的行都必须落在检测起音上。
- 断言只证明程序遵守约束，不证明检测器正确或听感准确。

## 尚未完成

转移表来自原仓库最多 8000 个谱文件的处理结果，字段 n 实际是行样本权重，不是谱面首数。没有新增下载官方分类包；Speed/Stamina 尚未单独建立可靠训练集。人类局部转移不保证拼接后不卡手，仍需多音上下文和 LN 占用检查；不把检查通过等同于音乐性验收。

保留 V13–V15。本版请先试听 Tech 与 Jack 的区别；不以密度或站点 D 作为质量结论。
'''
(OUT/'VERSION.md').write_text(readme)
with zipfile.ZipFile(OUT/'GJ-V37.osz','w',zipfile.ZIP_DEFLATED) as z:
    z.write(ROOT/'ref/INTERNET YAMERO.ogg','INTERNET YAMERO.ogg')
    for row in tiers: z.write(OUT/f'GJ-V37-{row[0]}.osu',f'GJ-V37-{row[0]}.osu')
with zipfile.ZipFile(OUT/'GJ-V37.osz') as z:
    assert z.testzip() is None
    assert len([n for n in z.namelist() if n.endswith('.osu')])==len(tiers)
print(f'V21: {len(tiers)} charts + audio; ZIP integrity verified',flush=True)
