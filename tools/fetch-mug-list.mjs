#!/usr/bin/env node
/**
 * fetch-mug-list.mjs —— 按 Mug-Diffusion 官方数据集清单补齐 3 万张 4K 谱面
 *
 *   node tools/fetch-mug-list.mjs                # 全量补齐
 *   node tools/fetch-mug-list.mjs --dry          # 只统计缺口
 *
 * 源：osu.direct /api/osu/{beatmap_id}（单谱 .osu 文本，~40-150KB）
 * 清单：out/mugdiffusion-trainset.jsonl（34001 条 / 33523 唯一 id）
 * 续传：以 out/charts/{id}.osu 是否存在为准；进度写 out/fetch-mug-status.json
 *
 * 克制：单流 110ms 间隔 + 429 指数退避，别把镜像站打挂。
 */
import fs from 'node:fs';
import path from 'node:path';

const ROOT = '/workspace/dsh000/06-chart-ai';
const LIST = path.join(ROOT, 'out/mugdiffusion-trainset.jsonl');
const OUT_DIR = path.join(ROOT, 'out/charts');
const STATUS = path.join(ROOT, 'out/fetch-mug-status.json');
const DRY = process.argv.includes('--dry');
const DELAY = Number((process.argv.find(a => a.startsWith('--delay=')) || '--delay=110').split('=')[1]);

const API = 'https://osu.direct/api';
const UA = 'scene3d-dataset-builder/1.0 (research; contact: site owner)';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* 双源：ppy 官方预览端点（与镜像逐字节一致，已验证）优先，
   osu.direct 兜底。fetch 必须带超时——node 默认无超时，
   僵死连接会把 worker 永久卡住（10:18 那次事故的根因）。 */
const SRC = [
  (id) => `https://osu.ppy.sh/osu/${id}`,
  (id) => `${API}/osu/${id}`,
];

async function fetchText(url, tries = 2) {
  for (let t = 0; t < tries; t++) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': UA },
                                   signal: AbortSignal.timeout(20000) });
      if (r.status === 429) { await sleep(2000 * (t + 1)); continue; }
      if (r.status === 404) throw new Error('404');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.text();
    } catch (e) {
      if (e.message === '404') throw e;
      if (t === tries - 1) throw e;
      await sleep(800 * (t + 1));
    }
  }
}

async function getChart(id) {
  let lastErr;
  for (const src of SRC) {
    try {
      return await fetchText(src(id));
    } catch (e) {
      lastErr = e;
      if (e.message === '404') throw e;
    }
  }
  throw lastErr;
}

// ---- 清单 ----
const entries = fs.readFileSync(LIST, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const byId = new Map();
for (const e of entries) if (!byId.has(e.id)) byId.set(e.id, e);

// ---- 缺口 ----
const want = [];
for (const [id, e] of byId) {
  const f = path.join(OUT_DIR, `${id}.osu`);
  if (!fs.existsSync(f)) want.push({ id, e });
}
console.log(`清单 ${byId.size} 个唯一 id，已下载 ${byId.size - want.length}，待抓 ${want.length}`);

if (DRY) {
  fs.writeFileSync(STATUS, JSON.stringify({ total: byId.size, have: byId.size - want.length, want: want.length, done: false }, null, 1));
  process.exit(0);
}

fs.mkdirSync(OUT_DIR, { recursive: true });
/* 4 线程并发：镜像站单请求延迟 ~0.7s，单流只有 1.3/s（ETA 7h），
   并发 4 → ~5/s（ETA <2h），仍远低于打挂镜像的量级。 */
const WORKERS = 8;
let ok = 0, fail = 0, notMania = 0, bytes = 0, cursor = 0;
const failedIds = [];
const t0 = Date.now();
let lastStatus = 0;

function writeStatus(done) {
  const el = (Date.now() - t0) / 1000;
  const rate = (ok + fail) / Math.max(1, el);
  const eta = (want.length - ok - fail) / Math.max(0.1, rate) / 60;
  fs.writeFileSync(STATUS, JSON.stringify({
    total: byId.size, have: byId.size - want.length + ok,
    want: want.length, ok, fail, notMania, bytes,
    rate: +rate.toFixed(2), etaMin: +eta.toFixed(0),
    failedIds: failedIds.slice(-200), done,
    updatedAt: new Date().toISOString(),
  }, null, 1));
}

async function worker(wid) {
  for (;;) {
    const i = cursor++;
    if (i >= want.length) return;
    const { id } = want[i];
    const file = path.join(OUT_DIR, `${id}.osu`);
    if (fs.existsSync(file)) continue;
    try {
      const text = await getChart(id);
      if (!/^Mode:\s*3/m.test(text)) throw new Error('not mania');
      const cs = text.match(/^CircleSize:\s*([\d.]+)/m);
      if (!cs || Math.round(parseFloat(cs[1])) !== 4) throw new Error('not 4K');
      fs.writeFileSync(file, text);
      ok++; bytes += text.length;
    } catch (err) {
      fail++;
      if (err.message === 'not mania' || err.message === 'not 4K') notMania++;
      else failedIds.push(id);
    }
    const now = Date.now();
    if (now - lastStatus > 30000) {
      lastStatus = now;
      const el = (now - t0) / 1000;
      const rate = (ok + fail) / Math.max(1, el);
      const eta = (want.length - ok - fail) / Math.max(0.1, rate) / 60;
      console.log(`进度 ${ok + fail}/${want.length}  ok=${ok} fail=${fail}  ` +
        `${(bytes / 1048576).toFixed(0)}MB  ${rate.toFixed(1)}/s  ETA ${eta.toFixed(0)}min`);
      writeStatus(false);
    }
    await sleep(DELAY);
  }
}

await Promise.all(Array.from({ length: WORKERS }, (_, w) => worker(w)));
writeStatus(true);
console.log(`完成：ok=${ok} fail=${fail}`);
