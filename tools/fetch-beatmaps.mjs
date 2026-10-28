#!/usr/bin/env node
/**
 * fetch-beatmaps.mjs — 从 osu! 镜像站批量抓 4K mania 谱面
 *
 * 用法：
 *   node tools/fetch-beatmaps.mjs --sets 200 --out out/charts
 *   node tools/fetch-beatmaps.mjs --sets 500 --out out/charts --resume
 *
 * 走的是 osu.direct 的 /api/osu/{beatmap_id}，只返回单个 .osu 文本（约 30 KB），
 * 比下载整个 .osz（含音频，动辄 8 MB）快两百多倍。
 *
 * 请克制：
 *   - 默认每个请求间隔 120ms，别把人家镜像站打挂
 *   - 只抓 4K mania，其余跳过
 *   - 这些是社区作品，训练/研究可以，再分发要注意授权和署名
 */
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
function argOf(name, dflt) {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : dflt;
}
const WANT_SETS = Number(argOf('sets', 200));
const OUT_DIR = argOf('out', 'out/charts');
const RESUME = args.includes('--resume');
const DELAY = Number(argOf('delay', 120));
const API = 'https://osu.direct/api';
const UA = 'scene3d-dataset-builder/1.0 (research; contact: site owner)';

fs.mkdirSync(OUT_DIR, { recursive: true });

const indexFile = path.join(OUT_DIR, '_index.json');
let index = {};
if (RESUME && fs.existsSync(indexFile)) {
  index = JSON.parse(fs.readFileSync(indexFile, 'utf8'));
  console.log(`续传：已有 ${Object.keys(index).length} 条记录`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getJSON(url, tries = 3) {
  for (let t = 0; t < tries; t++) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': UA } });
      if (r.status === 429) { await sleep(2000 * (t + 1)); continue; }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.json();
    } catch (e) {
      if (t === tries - 1) throw e;
      await sleep(800 * (t + 1));
    }
  }
}

async function getText(url, tries = 3) {
  for (let t = 0; t < tries; t++) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': UA } });
      if (r.status === 429) { await sleep(2000 * (t + 1)); continue; }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.text();
    } catch (e) {
      if (t === tries - 1) throw e;
      await sleep(800 * (t + 1));
    }
  }
}

/* ---------- 1. 搜 beatmapset，筛 4K mania ---------- */
const targets = [];
let page = 0;
const PER = 50;

while (targets.length < WANT_SETS * 4 && page * PER < WANT_SETS * 4 && page < 200) {
  const url = `${API}/v2/search?mode=3&limit=${PER}&offset=${page * PER}`;
  let sets;
  try {
    sets = await getJSON(url);
  } catch (e) {
    console.log(`  第 ${page} 页失败：${e.message}，停止翻页`);
    break;
  }
  if (!Array.isArray(sets) || !sets.length) break;

  for (const s of sets) {
    for (const b of (s.beatmaps || [])) {
      if (b.mode !== 'mania') continue;
      if (b.cs !== 4) continue;                 // 只要 4K
      if (index[b.id]) continue;                // 续传时跳过
      targets.push({
        id: b.id,
        set: b.beatmapset_id,
        stars: b.difficulty_rating,
        bpm: s.bpm,
        version: b.version,
        title: s.title,
        artist: s.artist
      });
    }
  }
  page++;
  process.stdout.write(`\r搜索第 ${page} 页，累计 4K 难度 ${targets.length} 个   `);
  await sleep(DELAY);
}
console.log(`\n共找到 ${targets.length} 个待下载的 4K 难度`);

/* ---------- 2. 逐个下载 .osu ---------- */
let ok = 0, fail = 0, bytes = 0;
for (let i = 0; i < targets.length; i++) {
  const t = targets[i];
  const file = path.join(OUT_DIR, `${t.id}.osu`);
  if (RESUME && fs.existsSync(file)) {
    // 断点续传时把索引补齐（否则中途 kill 会永久丢元数据）
    if (!index[t.id]) {
      index[t.id] = {
        id: t.id, set: t.set, stars: +(+t.stars).toFixed(2),
        bpm: t.bpm, version: t.version, title: t.title, artist: t.artist
      };
    }
    ok++;
    continue;
  }

  try {
    const text = await getText(`${API}/osu/${t.id}`);
    // 二次校验：确实是 mania 4K 才留
    if (!/^Mode:\s*3/m.test(text)) throw new Error('not mania');
    const cs = text.match(/^CircleSize:\s*([\d.]+)/m);
    if (!cs || Math.round(parseFloat(cs[1])) !== 4) throw new Error('not 4K');

    fs.writeFileSync(file, text);
    index[t.id] = {
      id: t.id, set: t.set, stars: +(+t.stars).toFixed(2),
      bpm: t.bpm, version: t.version, title: t.title, artist: t.artist,
      bytes: text.length
    };
    ok++; bytes += text.length;
  } catch (e) {
    fail++;
  }

  if (i % 20 === 0) {
    // 每 20 条落一次盘：进程被 kill 也不至于丢掉全部元数据
    fs.writeFileSync(indexFile, JSON.stringify(index));
    process.stdout.write(`\r下载 ${i + 1}/${targets.length}  成功 ${ok}  失败 ${fail}  ${(bytes / 1048576).toFixed(1)} MB   `);
  }
  await sleep(DELAY);
}

fs.writeFileSync(indexFile, JSON.stringify(index, null, 1));
console.log(`\n完成：成功 ${ok}，失败 ${fail}，共 ${(bytes / 1048576).toFixed(1)} MB`);
console.log(`索引写入 ${indexFile}`);

/* ---------- 3. 粗估 ---------- */
const sample = Object.values(index).slice(0, 20);
if (sample.length) {
  const avgStar = sample.reduce((a, x) => a + (x.stars || 0), 0) / sample.length;
  console.log(`样本平均星数 ${avgStar.toFixed(2)}★，谱面 ${Object.keys(index).length} 张`);
}
