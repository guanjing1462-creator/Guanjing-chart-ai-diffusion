#!/usr/bin/env node
/**
 * build-dataset.mjs — 把一批谱面转成训练用的 token 序列
 *
 * 用法：
 *   node tools/build-dataset.mjs <谱面目录> [-o out/dataset.jsonl] [--keys 4]
 *
 * 输入：目录下递归找 .osu（也支持 .mcz：会调用系统 unzip 解到临时目录）
 * 输出：JSONL，每行一个样本：
 *   {"id":"xxx","keys":4,"bpm":180,"frames":2100,"tokens":[...],"holds":[...]}
 *
 * 为什么输出 JSONL 而不是 pickle：
 *   就是纯文本，任何语言都能读；Python 侧用 `json.loads` 逐行读，不需要额外依赖。
 *   一个样本一行也方便用 `head` / `wc -l` 做快速检查。
 */
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const CT = require('../src/chart-tokenizer.js');

const args = process.argv.slice(2);
const inputDir = args.find((a) => !a.startsWith('-'));
const outIdx = args.indexOf('-o');
const outPath = outIdx >= 0 ? args[outIdx + 1] : 'out/dataset.jsonl';
const keysIdx = args.indexOf('--keys');
const forceKeys = keysIdx >= 0 ? Number(args[keysIdx + 1]) : 0;

if (!inputDir) {
  console.error('用法: node tools/build-dataset.mjs <谱面目录> [-o out/dataset.jsonl] [--keys 4]');
  process.exit(1);
}

/* ---------- 找谱面文件 ---------- */
function walk(dir, acc = []) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, acc);
    else if (/\.(osu|mcz)$/i.test(ent.name)) acc.push(p);
  }
  return acc;
}

/* ---------- .mcz 里取 .osu ---------- */
function osuFromMcz(file) {
  const tmp = fs.mkdtempSync(path.join(process.env.TMPDIR || '/tmp', 'mcz-'));
  try {
    execFileSync('unzip', ['-o', '-q', file, '-d', tmp], { stdio: 'ignore' });
    const found = walk(tmp).filter((f) => f.endsWith('.osu'));
    if (!found.length) return [];
    // 取最像谱面的那个（有的包里有多个难度）
    return found.map((f) => fs.readFileSync(f, 'utf8'));
  } catch (e) {
    return [];
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

/* ---------- 主流程 ---------- */
const files = walk(inputDir);
console.log(`找到 ${files.length} 个文件`);

const out = [];
let skipped = 0;
const keyHist = {};
const bpmHist = {};

for (const f of files) {
  let texts = [];
  try {
    if (f.toLowerCase().endsWith('.mcz')) texts = osuFromMcz(f);
    else texts = [fs.readFileSync(f, 'utf8')];
  } catch (e) {
    skipped++;
    continue;
  }

  for (const text of texts) {
    let parsed;
    try {
      parsed = require('../src/beatmap-parser.js').parse(text);
    } catch (e) {
      if (process.env.DEBUG) console.error('parse fail', f, e.message);
      skipped++;
      continue;
    }
    if (!parsed || !parsed.notes || !parsed.notes.length) {
      if (process.env.DEBUG) console.error('empty', f);
      skipped++;
      continue;
    }

    const keys = forceKeys || parsed.keys;
    if (keys > 8) { skipped++; continue; }   // 当前编码只支持 ≤8K

    const tok = CT.tokenize(parsed.notes, { keys, bpm: parsed.bpm || 160 });

    out.push({
      id: path.basename(f, path.extname(f)),
      keys,
      bpm: Math.round(parsed.bpm || 160),
      frames: tok.frames,
      stepMs: Math.round(tok.stepMs * 1000) / 1000,
      tokens: Array.from(tok.tokens),
      holds: tok.holds,
      meta: {
        title: parsed.metadata && parsed.metadata.title,
        version: parsed.metadata && parsed.metadata.version,
        notes: parsed.objectCount
      }
    });

    keyHist[keys] = (keyHist[keys] || 0) + 1;
    bpmHist[Math.round(parsed.bpm || 0)] = (bpmHist[Math.round(parsed.bpm || 0)] || 0) + 1;
  }
}

fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
fs.writeFileSync(outPath, out.map((o) => JSON.stringify(o)).join('\n') + '\n');

const totalFrames = out.reduce((a, o) => a + o.frames, 0);
const totalNotes = out.reduce((a, o) => a + (o.meta.notes || 0), 0);

console.log(`\n写出 ${out.length} 个样本 -> ${outPath}`);
console.log(`跳过 ${skipped} 个（解析失败 / >8K / 空谱）`);
console.log(`总帧数 ${totalFrames.toLocaleString()}，总 notes ${totalNotes.toLocaleString()}`);
console.log('键位分布:', keyHist);
console.log('BPM 分布(前5):', Object.entries(bpmHist).sort((a, b) => b[1] - a[1]).slice(0, 5));

/* 训练量够不够的粗估：一个 20M 参数模型大致需要 500 万帧以上才不过拟合 */
const need = 5_000_000;
console.log(`\n数据量评估：当前 ${totalFrames.toLocaleString()} 帧，约需 ${(need / totalFrames).toFixed(1)} 倍才够训一个 20M 参数模型`);
