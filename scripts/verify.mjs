#!/usr/bin/env node
// ============================================================================
// scripts/verify.mjs —— 配置驱动的发布门禁（通用，不绑定任何技术栈）
//
// 读项目根目录 pipeline.config.json 的 "verify" 数组，逐条执行；
// 任何一条失败即停，最后汇总失败清单并以非 0 退出。
// 换技术栈只改配置，不改本文件。
//
// 用法（在项目根目录）：
//   node scripts/verify.mjs              # run all gates
//   node scripts/verify.mjs --list       # 只列出会跑哪些命令，不执行
//   node scripts/verify.mjs --only "rust fmt"          # 只跑指定项（可重复传）
//   node scripts/verify.mjs --skip "frontend build"    # 跳过指定项（可重复传）
//   node scripts/verify.mjs --key precommit        # 跑配置里的 precommit 数组（提交前的快门禁）
//   node scripts/verify.mjs --config path/to/pipeline.config.json
// ============================================================================

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const argv = process.argv.slice(2);
function valuesOf(flag) {
  const out = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === flag && argv[i + 1]) out.push(argv[i + 1]);
  }
  return out;
}
// 未知参数直接报错：拼错的 --xxx 静默忽略会与本意相反（理由同 release.mjs）
const KNOWN_FLAGS = ["--list", "--only", "--skip", "--key", "--config"];
const unknownFlags = argv.filter((a) => a.startsWith("--") && !KNOWN_FLAGS.includes(a));
if (unknownFlags.length > 0) {
  console.error(
    `Unknown argument(s): ${unknownFlags.join(" ")}\n` +
      `Available flags: ${KNOWN_FLAGS.join(" / ")} (--only, --skip, and --key need a value; --config needs a path)`
  );
  process.exit(1);
}

const listOnly = argv.includes("--list");
const only = valuesOf("--only");
const skip = valuesOf("--skip");
const configArgIdx = argv.indexOf("--config");
const configPath = path.resolve(
  ROOT,
  configArgIdx >= 0 && argv[configArgIdx + 1] ? argv[configArgIdx + 1] : "pipeline.config.json"
);

function loadConfig() {
  if (!fs.existsSync(configPath)) {
    console.error(
      `Config file not found: ${configPath}\n` +
        "Copy pipeline.config.example.json to the project root as pipeline.config.json and update its fields."
    );
    process.exit(1);
  }
  try {
    return JSON.parse(fs.readFileSync(configPath, "utf8"));
  } catch (err) {
    console.error(`Invalid JSON config: ${configPath}\n${err.message}`);
    process.exit(1);
  }
}

// 读哪个数组：默认 verify（全量门禁）；--key precommit 读配置里的快门禁
const keyIdx = argv.indexOf("--key");
const configKey = keyIdx >= 0 && argv[keyIdx + 1] ? argv[keyIdx + 1] : "verify";

const config = loadConfig();
const steps = Array.isArray(config[configKey]) ? config[configKey] : [];

if (steps.length === 0) {
  console.error(
    `Config has no ${configKey} array (or it is empty): ${configPath}\n` +
      (configKey === "verify"
        ? "Add gate commands first. If this repository needs no gate, run release.mjs with --no-verify."
        : `--key ${configKey} points to an optional gate; leave it empty only when the caller explicitly skips it.`)
  );
  process.exit(1);
}

/** Windows 下 npm / pnpm / yarn / npx / bun 必须以 .cmd 形式调用，否则 execSync 找不到 */
function normalizeCmd(cmd) {
  if (process.platform !== "win32") return cmd;
  return cmd.replace(/^(npm|pnpm|yarn|npx|bun)(\s)/, "$1.cmd$2");
}

const selected = steps.filter((s) => {
  const name = s.name || s.cmd;
  if (only.length > 0 && !only.includes(name)) return false;
  if (skip.includes(name)) return false;
  return true;
});

if (selected.length === 0) {
  console.error("No steps remain after filtering; check --only / --skip names against the config.");
  process.exit(1);
}

if (listOnly) {
  console.log(`Config: ${path.relative(ROOT, configPath)}`);
  for (const [i, s] of selected.entries()) {
    const cwd = s.cwd ? `  (cwd: ${s.cwd})` : "";
    console.log(`  ${i + 1}. [${s.name || s.cmd}] ${s.cmd}${cwd}`);
  }
  process.exit(0);
}

const failed = [];

for (const step of selected) {
  const name = step.name || step.cmd;
  const cwd = step.cwd ? path.resolve(ROOT, step.cwd) : ROOT;
  console.log(`\n=== [${name}] ===`);
  try {
    execSync(normalizeCmd(step.cmd), { cwd, stdio: "inherit" });
    console.log(`--- ${name} passed ---`);
  } catch {
    failed.push(name);
  }
}

if (failed.length > 0) {
  console.error(`\n[verify] ${failed.length} step(s) failed:\n  - ${failed.join("\n  - ")}`);
  process.exit(1);
}

console.log(`\n[verify] All selected steps passed: ${selected.map((s) => s.name || s.cmd).join(" / ")}.`);
