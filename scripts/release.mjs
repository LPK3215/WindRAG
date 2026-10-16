#!/usr/bin/env node
// ============================================================================
// scripts/release.mjs —— 配置驱动的发版（通用，不绑定任何技术栈）
//
// 读项目根目录 pipeline.config.json：
//   versionFiles  版本回写位置（type: json / toml-package / cargo-lock / regex）
//   tagPrefix     tag 前缀，默认 "v"
//   remote        git 远程名，默认 "origin"（= 主仓 GitHub）
//   pushRemotes   （可选）分支 + tag 要推到哪些远程，第一个为primary
//   verify        （由 scripts/verify.mjs 读取）发布门禁命令
// 换技术栈只改配置，不改本文件。
//
// 流程：
//   1. 检查工作区无未提交改动（--force 可跳过）
//   2. 跑发布门禁（--no-verify / --dry-run 跳过）
//   3. 检查 tag 是否已存在
//   4. 把新版本号回写到 versionFiles 列出的每一处
//   5. 提交版本回写 → 打 tag → 推送分支 + tag（主仓必推；镜像推失败不阻断发版）
//   6. 主仓 CI 收到 tag 后构建并发布 Release
//
// 用法（在项目根目录）：
//   node scripts/release.mjs 0.2.4                # normal release
//   node scripts/release.mjs 0.2.4 --dry-run      # preview only
//   node scripts/release.mjs 0.2.4 --no-verify    # skip the gate
//   node scripts/release.mjs 0.2.4 --force        # allow a dirty worktree
//   node scripts/release.mjs 0.2.4 --config path/to/pipeline.config.json
// ============================================================================

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const args = process.argv.slice(2);
const positional = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--config") {
    i++;
    continue;
  }
  if (args[i].startsWith("--")) continue;
  positional.push(args[i]);
}
// 未知参数直接报错：拼错的 --xxx 会被静默忽略（例如想跳门禁却打成 `--skip-verify`），
// 结果与本来意图相反却不报错 —— 这类静默失败比直接失败更难查。
const KNOWN_FLAGS = ["--force", "--dry-run", "--no-verify", "--config"];
const unknownFlags = args.filter((a) => a.startsWith("--") && !KNOWN_FLAGS.includes(a));
if (unknownFlags.length > 0) {
  console.error(
    `Unknown argument(s): ${unknownFlags.join(" ")}\n` +
      `Available flags: ${KNOWN_FLAGS.join(" / ")} (--config needs a path)`
  );
  process.exit(1);
}

const versionArg = positional[0];
const force = args.includes("--force");
const dryRun = args.includes("--dry-run");
const skipVerify = args.includes("--no-verify") || dryRun;
const configIdx = args.indexOf("--config");
const configPath = path.resolve(
  ROOT,
  configIdx >= 0 && args[configIdx + 1] ? args[configIdx + 1] : "pipeline.config.json"
);

if (!versionArg) {
  console.error(
    "Usage: node scripts/release.mjs <version, for example 0.2.4> [--no-verify] [--force] [--dry-run] [--config <path>]"
  );
  process.exit(1);
}

const version = versionArg.replace(/^v/, "");
// 接受标准 semver（含预发布与构建元数据）：0.2.4 / v0.2.4 / 1.0.0-rc.1 / 2.0.0-beta.3+build.5
const SEMVER = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
if (!SEMVER.test(version)) {
  console.error(
    `Invalid version: ${versionArg}\nExpected 0.2.4 or v0.2.4 (pre-release versions such as 1.0.0-rc.1 are also accepted).`
  );
  process.exit(1);
}

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

const config = loadConfig();
const tagPrefix = typeof config.tagPrefix === "string" ? config.tagPrefix : "v";
const remote = typeof config.remote === "string" && config.remote ? config.remote : "origin";
const tag = `${tagPrefix}${version}`;
const versionFiles = Array.isArray(config.versionFiles) ? config.versionFiles : [];

if (versionFiles.length === 0) {
  console.error(`Config has no versionFiles (or it is empty): ${configPath}\nNo version file can be updated.`);
  process.exit(1);
}

function rel(abs) {
  return path.relative(ROOT, abs) || abs;
}

function sh(cmd, opts = {}) {
  console.log(`\n$ ${cmd}`);
  execSync(cmd, { cwd: ROOT, stdio: "inherit", ...opts });
}

/**
 * 推一次；命中「代理 / 网络瞬时抖动」的典型报错时**原样重试一次**。
 *
 * 为什么必须重试：在挂了本地代理（http.proxy / HTTPS_PROXY）的机器上，
 * 首次 push 出现 `TLS connect error: ...unexpected eof while reading`、
 * `RPC failed; curl 55 Recv failure`、`send-pack: unexpected disconnect`
 * 是常态，**原样再推一次就成功**。不重试等于把成功判成失败，
 * 而这类报错和「凭证错 / 仓库不存在」长得完全不一样，必须分开：
 * 非瞬时类报错原样抛出，不重试也不掩盖。
 */
const TRANSIENT_NET =
  /TLS connect error|unexpected eof|RPC failed|Recv failure|Connection was reset|early EOF|remote end hung up|Empty reply from server|Connection timed out/i;

function pushWithRetry(label, cmd) {
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      execSync(cmd, { cwd: ROOT, stdio: ["ignore", "inherit", "pipe"] });
      return;
    } catch (err) {
      const msg = `${err.stderr || ""}${err.stdout || ""}${err.message || ""}`;
      if (attempt === 1 && TRANSIENT_NET.test(msg)) {
        const first = msg.trim().split(/\r?\n/).filter(Boolean).pop() || "network error";
        console.log(`  ! ${label}: transient proxy or network error (${first}); retrying once...`);
        continue;
      }
      if (msg.trim()) console.error(msg.trim());
      throw err;
    }
  }
}

function remoteExists(name) {
  try {
    execSync(`git remote get-url "${name}"`, { cwd: ROOT, stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

/** remote URL → 网页地址（仅用于打印链接，失败就当没有） */
function remoteWebUrl(name) {
  try {
    const url = execSync(`git remote get-url "${name}"`, { cwd: ROOT, encoding: "utf8" }).trim();
    return url
      .replace(/^git@([^:]+):/, "https://$1/")
      .replace(/^ssh:\/\/git@([^/]+)\//, "https://$1/")
      .replace(/\.git$/, "");
  } catch {
    return "";
  }
}

/** 顶层 version 字段（package.json / tauri.conf.json） */
function bumpJson(abs, raw, nextVersion) {
  const obj = JSON.parse(raw);
  if (typeof obj.version !== "string") throw new Error(`${rel(abs)} is missing a top-level version field`);
  if (obj.version === nextVersion) return null;
  const old = obj.version;
  obj.version = nextVersion;
  return { file: abs, old, next: `${JSON.stringify(obj, null, 2)}\n` };
}

/** TOML [package] 段的 version 行（Cargo.toml） */
function bumpTomlPackage(abs, raw, nextVersion) {
  const marker = raw.indexOf("[package]");
  if (marker < 0) throw new Error(`${rel(abs)} has no [package] section`);
  const rest = raw.slice(marker);
  const m = rest.match(/^version = "([^"]*)"/m);
  if (!m) throw new Error(`${rel(abs)} has no version line in [package]`);
  if (m[1] === nextVersion) return null;
  const next = rest.replace(/^version = "[^"]*"/m, `version = "${nextVersion}"`);
  return { file: abs, old: m[1], next: raw.slice(0, marker) + next };
}

/** Cargo.lock 中 name = <pkgName> 的那个 [[package]] 块的 version */
function bumpCargoLock(abs, raw, nextVersion, pkgName) {
  if (!pkgName) {
    throw new Error(`${rel(abs)} uses type cargo-lock and requires a name field in the config`);
  }
  const lines = raw.split("\n");
  const out = [];
  let inBlock = false;
  let isTarget = false;
  let found = false;
  let oldVersion = null;
  for (const line of lines) {
    if (line.trim() === "[[package]]") {
      inBlock = true;
      isTarget = false;
      out.push(line);
      continue;
    }
    if (inBlock && !isTarget) {
      const m = line.match(/^name = "([^"]*)"/);
      if (m) {
        found = true;
        isTarget = m[1] === pkgName;
        out.push(line);
        continue;
      }
    }
    if (inBlock && isTarget) {
      const m = line.match(/^version = "([^"]*)"/);
      if (m) {
        oldVersion = m[1];
        out.push(oldVersion === nextVersion ? line : `version = "${nextVersion}"`);
        continue;
      }
    }
    out.push(line);
  }
  if (!found) throw new Error(`Cannot find the [[package]] block for name = "${pkgName}" in ${rel(abs)}`);
  if (oldVersion === null) throw new Error(`The block for name = "${pkgName}" has no version line in ${rel(abs)}`);
  if (oldVersion === nextVersion) return null;
  return { file: abs, old: oldVersion, next: out.join("\n") };
}

/** 自定义正则：pattern 匹配含旧版本的那段文本，replace 里的 $VERSION 代表新版本 */
function bumpRegex(abs, raw, nextVersion, entry) {
  if (!entry.pattern || typeof entry.replace !== "string") {
    throw new Error(`${rel(abs)} regex entries require both pattern and replace`);
  }
  const flags = entry.flags || "m";
  if (!new RegExp(entry.pattern, flags).test(raw)) {
    throw new Error(`Regex matched no content: ${entry.pattern} (${rel(abs)})`);
  }
  const next = raw.replace(
    new RegExp(entry.pattern, flags),
    entry.replace.replaceAll("$VERSION", nextVersion)
  );
  if (next === raw) return null;
  return { file: abs, old: "(regex)", next };
}

function bump(entry) {
  const abs = path.resolve(ROOT, entry.path);
  if (!fs.existsSync(abs)) throw new Error(`Version file not found: ${entry.path}`);
  const raw = fs.readFileSync(abs, "utf8");
  switch (entry.type) {
    case "json":
      return bumpJson(abs, raw, version);
    case "toml-package":
      return bumpTomlPackage(abs, raw, version);
    case "cargo-lock":
      return bumpCargoLock(abs, raw, version, entry.name);
    case "regex":
      return bumpRegex(abs, raw, version, entry);
    default:
      throw new Error(
        `Unknown version file type: ${entry.type} (${entry.path}). Use json, toml-package, cargo-lock, or regex.`
      );
  }
}

function ensureGitIdentity() {
  try {
    execSync("git config user.name", { cwd: ROOT, stdio: "pipe" });
    execSync("git config user.email", { cwd: ROOT, stdio: "pipe" });
  } catch {
    console.error(
      "\nGit user.name / user.email are not configured, so the version update cannot be committed.\n" +
        'Run: git config user.name "Your Name" && git config user.email "you@example.com"'
    );
    process.exit(1);
  }
}

// 1) 工作区状态
const status = execSync("git status --porcelain", { cwd: ROOT }).toString().trim();
if (status && !force) {
  console.error(
    "The worktree has uncommitted changes; release from a clean history.\n" +
      "Commit or discard the changes, or use --force to continue.\n\n" +
      status
  );
  process.exit(1);
}

// 2) 门禁
// 直接调用同目录的 verify.mjs —— 不依赖 package.json 里的 scripts，
// 这样没有 package.json 的项目（纯 Python / 纯 Rust）也能用同一套。
if (!skipVerify) {
  sh(`"${process.execPath}" "${path.join(__dirname, "verify.mjs")}"`);
}

// 3) tag 已存在检查
try {
  execSync(`git rev-parse "${tag}"`, { cwd: ROOT, stdio: "ignore" });
  console.error(
    `Tag ${tag} already exists. Delete it (git tag -d ${tag} && git push ${remote} :${tag}) before releasing.`
  );
  process.exit(1);
} catch {
  /* tag 不存在，继续 */
}

// 4) 版本号回写
const bumps = versionFiles.map(bump).filter(Boolean);

if (bumps.length === 0) {
  console.log(`\n[release] All version files already contain ${version}; no update needed.`);
} else {
  console.log(`\n[release] Updating version files to ${version}:`);
  for (const b of bumps) {
    console.log(`  - ${rel(b.file)}: ${b.old} -> ${version}`);
  }
}

if (dryRun) {
  console.log("\n[dry-run] Preview only: no files, commits, or pushes were changed.");
  process.exit(0);
}

if (bumps.length > 0) {
  ensureGitIdentity();
  for (const b of bumps) fs.writeFileSync(b.file, b.next);
  const quoted = bumps.map((b) => `"${b.file}"`).join(" ");
  sh(`git add -- ${quoted}`);
  // 用 --no-verify 提交：本脚本第 2 步已经跑过全量门禁（verify），
  // 而项目的 pre-commit 钩子跑的是 precommit（人工提交用的快检子集）——
  // 再跑一遍既冗余，又会在钩子环境不满足时（如刚 clone 没装依赖）把发版
  // 中断在「版本号已回写写盘、但没提交」的半成品状态。钩子是给人工提交用的，
  // 自动化流程自带更强的门禁。
  sh(`git commit --no-verify -m "chore: bump version to ${tag}"`);
}

// 5) 打 tag + 推送
sh(`git tag "${tag}"`);
const branch = execSync("git rev-parse --abbrev-ref HEAD", { cwd: ROOT }).toString().trim();

// 推送目标：默认只有主仓（config.remote = origin = GitHub）；
// 配了 pushRemotes.list 就按列出的顺序依次推，主仓必须失败即停，镜像失败只告警。
const pushCfg =
  config.pushRemotes && typeof config.pushRemotes === "object" ? config.pushRemotes : null;
const primaryName =
  pushCfg && typeof pushCfg.primary === "string" && pushCfg.primary ? pushCfg.primary : remote;
const targets = [];
if (pushCfg && Array.isArray(pushCfg.list)) {
  for (const r of pushCfg.list) {
    if (typeof r !== "string" || !r.trim()) continue;
    const name = r.trim();
    if (!targets.some((t) => t.name === name)) targets.push({ name, primary: name === primaryName });
  }
}
// 没配 pushRemotes.list 时列表里只有主仓 —— 保持原有行为，只推 config.remote
if (!targets.some((t) => t.primary)) targets.unshift({ name: primaryName, primary: true });

console.log(
  `\n[release] Pushing ${branch} + ${tag} -> ${targets.map((t) => t.name + (t.primary ? " (primary)" : "")).join(" / ")}`
);

let primaryWeb = "";
const mirrorFailed = [];
const mirrorPushed = [];

for (const t of targets) {
  if (!remoteExists(t.name)) {
    if (t.primary) {
      console.error(
        `\nPrimary remote ${t.name} does not exist; release aborted.\n` +
          `Add it first: git remote add ${t.name} <repository-url> (GitHub is the primary remote.)`
      );
      process.exit(1);
    }
    console.log(`  - Remote ${t.name} is not configured; skipping. Add it later and push with git push ${t.name} ${tag}.`);
    continue;
  }

  const web = remoteWebUrl(t.name);
  try {
    pushWithRetry(`${t.name} branch`, `git push "${t.name}" "${branch}"`);
    pushWithRetry(`${t.name} tag`, `git push "${t.name}" "${tag}"`);
    console.log(`  [ok] ${t.name}${web ? "  " + web : ""}`);
    if (t.primary) primaryWeb = web;
    else mirrorPushed.push(t.name);
  } catch (err) {
    if (t.primary) {
      console.error(
        `\nPrimary remote ${t.name} push failed; release aborted (the tag is local).\n` +
          `After fixing network or permission issues, run: git push ${t.name} ${branch} && git push ${t.name} ${tag}\n` +
          `${err.message}`
      );
      process.exit(1);
    }
    mirrorFailed.push(t.name);
    console.error(
      `\n  ! Mirror ${t.name} push failed: ${err.message}\n` +
        `    Primary push succeeded. Retry later with: git push ${t.name} ${branch} && git push ${t.name} ${tag}`
    );
  }
}

const mirrors = targets.filter((t) => !t.primary);
console.log(`\n[release] ${tag} pushed. CI will build and publish the Release after receiving the tag${primaryWeb ? ":" : "."}`);
if (primaryWeb) console.log(`  ${primaryWeb}/releases`);
if (mirrors.length > 0) {
  if (mirrorPushed.length > 0) console.log(`  Mirror branch + tag synced: ${mirrorPushed.join(" / ")}`);
  if (mirrorFailed.length > 0) console.log(`  Mirror pushes failed: ${mirrorFailed.join(" / ")} (retry with git push <remote> ${branch} && git push <remote> ${tag})`);
  console.log(
    `  Artifacts require a separate step: node scripts/mirror-sync.mjs${mirrorFailed.length ? " (push the missing mirror tag first)" : ""}`
  );
}
