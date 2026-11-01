"""Reproducible V16 repair release. No modification of previous releases."""
import hashlib, json, os, pathlib, subprocess, sys, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'out/v16-verified'; OUT.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT)
env=dict(os.environ,OMP_NUM_THREADS='4')
subprocess.run([sys.executable,'tools/test-pattern-sampler.py'],check=True,env=env)
tiers=[('Rice','rice/technical/mid','rice','stream','0.1'),('Stream','rice/stream/high','stream','stream','0'),('Tech','rice/technical/high','tech','technical','1'),('Jack','rice/technical/high','tech','jack','1'),('LN','ln/technical/high','ln','technical','1.2')]
records=[]
for name,style,cls,ptype,boost in tiers:
    dest=OUT/f'GJ-V16-{name}.osu'
    cmd=[sys.executable,'tools/paint-gj.py','--analysis','out/final14-s50/analysis-fixed.json','--audio','ref/INTERNET YAMERO.ogg','--lexicon','out/rhythm-lexicon16.json','--style',style,'--style-class',cls,'--ptype',ptype,'--strict','--dps','12.6','--chord-boost',boost,'--model','out/placement3k.pt','--song-id','INTERNET_YAMERO','--target-ett','18','--creator','GJ','--version',f'GJ V16 {name}','--seed','22','-o',str(dest)]
    if name=='LN': cmd+=['--ln-cap','0.18']
    with (OUT/f'{name.lower()}.log').open('w') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
    records.append(dict(tier=name,command=cmd,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    print(f'{name}: generated',flush=True)
subprocess.run([sys.executable,'tools/check-chart.py',*[str(OUT/f'GJ-V16-{r[0]}.osu') for r in tiers]],check=True,env=env)
manifest={'version':'V16','purpose':'Sampler repair and strict timing invariant regression release; not listening-approved.', 'charts':records,'inputs':{}}
for rel in ['tools/paint-gj.py','tools/test-pattern-sampler.py','tools/release-v16.py','out/pattern-corpus.json','out/placement3k.pt','out/final14-s50/analysis-fixed.json','ref/INTERNET YAMERO.ogg']:
    manifest['inputs'][rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
readme='''# GJ V16 · 修复验证版

此前 V16 生成失败，本版已修复；不是以前失败产物的重命名。

- 修复采样器缩进错误、PATTERN 解包错误、单音绕过人类转移表、时间间隔单位混用。
- 缺少指定大小的键型时保留原大小回退，不偷偷增加和弦键数。
- 严格采音关闭跨段节奏复制和无条件收尾补音；小节相位使用实际 cell id。
- 生成结束断言：所有严格模式行的格号都属于检测起音集合。它只证明程序遵守约束，不证明检测器正确或听感准确。
- 五档均由同一脚本生成，带 V16 元数据。manifest 记录命令与文件哈希。

## 尚未完成

转移表来自原仓库最多 8000 个谱文件的处理结果，字段 n 实际是行样本权重，不是谱面首数。没有新增下载官方分类包；Speed/Stamina 尚未单独建立可靠训练集。人类局部转移不保证拼接后不卡手，仍需多音上下文和 LN 占用检查；不把检查通过等同于音乐性验收。

保留 V13–V15。本版请先试听 Tech 与 Jack 的区别；不以密度或站点 D 作为质量结论。
'''
(OUT/'VERSION.md').write_text(readme)
with zipfile.ZipFile(OUT/'GJ-V16.osz','w',zipfile.ZIP_DEFLATED) as z:
    z.write(ROOT/'ref/INTERNET YAMERO.ogg','INTERNET YAMERO.ogg')
    for row in tiers: z.write(OUT/f'GJ-V16-{row[0]}.osu',f'GJ-V16-{row[0]}.osu')
with zipfile.ZipFile(OUT/'GJ-V16.osz') as z:
    assert z.testzip() is None
    assert len([n for n in z.namelist() if n.endswith('.osu')])==5
print('V16: 5 charts + audio; ZIP integrity verified',flush=True)
