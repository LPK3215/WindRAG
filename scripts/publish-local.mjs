#!/usr/bin/env node
// ============================================================================
// scripts/publish-local.mjs —— 本地出包 + 发布 Release（不依赖 GitHub Actions）
//
// 为什么需要它：CI 可能因为**与你的代码完全无关**的原因跑不起来 ——
//   ① 账号被计费锁定（`the job was not started because your account is locked
//      due to a billing issue`）→ 所有仓库全部无法启动，公开仓也一样；
//   ② 私有仓免费额度用尽（Free 账号 2000 分钟/月）；
//   ③ 组织/仓库策略禁用了 Actions。
// 这时 tag 推上去了，CI 却不出包。本脚本走**同一条发版语义**，只是把"出包"
// 从云端搬回本地：门禁 → 出包 → 把产物挂到该 tag 的正式 Release 上。
//
// 读项目根目录 pipeline.config.json 的 "localPublish" 段：
//   steps      出包步骤（逐条执行，任一失败即停）。留空 = 纯文本发布
//   artifacts  产物 glob（相对项目根），命中的文件挂到 Release。留空 = 纯文本发布
//
// 「发布」不必等于「出安装包」：文档站、纯库、只发版本说明的项目，两个数组都留空即可 ——
// 建一个带 release notes 的正式 Release，不挂任何文件。
//
// 换技术栈只改配置，不改本文件。
//
// 用法（在项目根目录；建 Release 需已安装并登录 gh）：
//   node scripts/publish-local.mjs              # gate, build, and create a Release
//   node scripts/publish-local.mjs --dry-run    # preview only
//   node scripts/publish-local.mjs --no-release # build locally without GitHub
//   node scripts/publish-local.mjs --no-verify  # skip the gate
//   node scripts/publish-local.mjs --tag v0.2.1 # select a tag explicitly
//   node scripts/publish-local.mjs --force      # allow a dirty worktree
// ============================================================================

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// ---------------------------------------------------------------- 参数
const args = process.argv.slice(2);

const KNOWN_FLAGS = [
  "--dry-run",
  "--no-release",
  "--no-verify",
  "--force",
  "--clobber",
  "--config",
  "--tag",
  "--notes",
];
const unknownFlags = args.filter((a) => a.startsWith("--") && !KNOWN_FLAGS.includes(a));
if (unknownFlags.length > 0) {
  console.error(
    `Unknown argument(s): ${unknownFlags.join(" ")}\n` +
      `Available flags: ${KNOWN_FLAGS.join(" / ")} (--config, --tag, and --notes need a value)`
  );
  process.exit(1);
}

function valueOf(flag) {
  const i = args.indexOf(flag);
  if (i < 0) return undefined;
  const v = args[i + 1];
  return v && !v.startsWith("--") ? v : undefined;
}

const dryRun = args.includes("--dry-run");
const noRelease = args.includes("--no-release");
const skipVerify = args.includes("--no-verify");
const force = args.includes("--force");
const clobber = args.includes("--clobber");
const explicitTag = valueOf("--tag");
const notesFileArg = valueOf("--notes");

const configPath = path.resolve(
  ROOT,
  valueOf("--config") || "pipeline.config.json"
);

// ---------------------------------------------------------------- 配置
function loadConfig() {
  if (!fs.existsSync(configPath)) {
    console.error(
      `Config file not found: ${configPath}\n` +
        "Copy pipeline.config.example.json to the project root as pipeline.config.json."
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
const versionFiles = Array.isArray(config.versionFiles) ? config.versionFiles : [];
const lp = config.localPublish && typeof config.localPublish === "object" ? config.localPublish : {};
const steps = Array.isArray(lp.steps) ? lp.steps : [];
const artifactGlobs = Array.isArray(lp.artifacts) ? lp.artifacts : [];

function rel(abs) {
  return path.relative(ROOT, abs) || abs;
}

function sh(cmd, opts = {}) {
  console.log(`\n$ ${cmd}`);
  if (dryRun) return "";
  return execSync(cmd, { cwd: ROOT, stdio: "inherit", ...opts });
}

/**
 * Windows 上 npm / pnpm / yarn / npx / bun 是 .cmd 批处理，execSync 直接调会
 * 找不到可执行文件 —— 统一改写为 .cmd 版本（与 verify.mjs 的规则一致）。
 */
function shStep(cmd, cwd) {
  let c = cmd;
  if (process.platform === "win32") {
    c = c.replace(/^(npm|npx|pnpm|yarn|bun)(\s|$)/, "$1.cmd$2");
  }
  console.log(`\n$ ${c}${cwd ? `   (cwd: ${cwd})` : ""}`);
  if (dryRun) return;
  execSync(c, { cwd: cwd ? path.resolve(ROOT, cwd) : ROOT, stdio: "inherit" });
}

// ---------------------------------------------------------------- 版本 / tag
/** 从 json 或 toml-package 版本文件里读出当前版本号 */
function readVersionFrom(entry) {
  const abs = path.resolve(ROOT, entry.path);
  if (!fs.existsSync(abs)) return null;
  const raw = fs.readFileSync(abs, "utf8");
  if (entry.type === "json") {
    try {
      const obj = JSON.parse(raw);
      return typeof obj.version === "string" ? obj.version : null;
    } catch {
      return null;
    }
  }
  if (entry.type === "toml-package") {
    // Cargo.toml 的 [package] 与 pyproject.toml 的 [project] 都认
    const m =
      raw.match(/\[package\][\s\S]*?\n\s*version\s*=\s*"([^"]+)"/) ||
      raw.match(/\[project\][\s\S]*?\n\s*version\s*=\s*"([^"]+)"/);
    return m ? m[1] : null;
  }
  return null;
}

function resolveTag() {
  if (explicitTag) return explicitTag.startsWith(tagPrefix) ? explicitTag : `${tagPrefix}${explicitTag}`;

  // versionFrom 可写成字符串（按扩展名自动判类型）或 { path, type } 对象
  let from = null;
  if (typeof lp.versionFrom === "string") {
    from = { path: lp.versionFrom, type: /\.toml$/i.test(lp.versionFrom) ? "toml-package" : "json" };
  } else if (lp.versionFrom && typeof lp.versionFrom === "object" && lp.versionFrom.path) {
    from = lp.versionFrom;
  }
  const candidates = from ? [from, ...versionFiles] : versionFiles;
  for (const entry of candidates) {
    const v = readVersionFrom(entry);
    if (v) return `${tagPrefix}${v}`;
  }
  console.error(
    "Cannot resolve a tag: no parseable version file (json or toml-package) is configured and --tag was not provided.\n" +
      `Add "localPublish": { "versionFrom": "package.json" } to pipeline.config.json, or pass --tag <tag>.`
  );
  process.exit(1);
}

// ---------------------------------------------------------------- 产物 glob
/** `*` 单层、`**` 任意层、其余按字面。够用且不引依赖。 */
function expandGlob(pattern) {
  const segs = pattern.split(/[\\/]/).filter((s) => s && s !== ".");
  let dirs = [ROOT];

  for (let i = 0; i < segs.length; i++) {
    const seg = segs[i];
    const isLast = i === segs.length - 1;
    const next = [];

    for (const dir of dirs) {
      if (seg === "**") {
        // 任意层（含零层）：把 dir 自身与所有子孙目录都作为候选
        next.push(...walkDirs(dir));
        continue;
      }
      if (!fs.existsSync(dir)) continue;

      if (seg.includes("*")) {
        const re = new RegExp(
          "^" + seg.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^\\\\/]*") + "$"
        );
        for (const name of fs.readdirSync(dir)) {
          if (!re.test(name)) continue;
          const p = path.join(dir, name);
          if (isLast ? fs.statSync(p).isFile() : fs.statSync(p).isDirectory()) next.push(p);
        }
      } else {
        const p = path.join(dir, seg);
        if (!fs.existsSync(p)) continue;
        if (isLast ? fs.statSync(p).isFile() : fs.statSync(p).isDirectory()) next.push(p);
      }
    }
    dirs = next;
  }
  return [...new Set(dirs)];
}

function walkDirs(root, out = []) {
  out.push(root);
  let entries = [];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    if (e.name === "node_modules" || e.name === ".git" || e.name === ".venv") continue;
    walkDirs(path.join(root, e.name), out);
  }
  return out;
}

// ---------------------------------------------------------------- release notes
/** 抽 CHANGELOG 里对应版本那一段（与 publish.yml 的 awk 语义一致，跨平台不依赖 awk） */
function extractChangelogSection(version) {
  const p = path.join(ROOT, "CHANGELOG.md");
  if (!fs.existsSync(p)) return null;
  const lines = fs.readFileSync(p, "utf8").split(/\r?\n/);
  const esc = version.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const head = new RegExp(`^##\\s*\\[?${esc}\\]?`);

  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    if (head.test(lines[i])) {
      start = i + 1;
      break;
    }
  }
  if (start < 0) return null;

  let end = lines.length;
  for (let i = start; i < lines.length; i++) {
    if (/^##\s/.test(lines[i])) {
      end = i;
      break;
    }
  }
  const body = lines.slice(start, end).join("\n").trim();
  return body || null;
}

function ghReady() {
  try {
    execSync("gh --version", { stdio: "pipe" });
  } catch {
    console.error(
      "\nCreating a Release requires GitHub CLI (gh). Install it from https://cli.github.com and run `gh auth login`.\n" +
        "Use --no-release to build locally without creating a Release."
    );
    process.exit(1);
  }
  // 只读本地凭据、**不发网络请求** —— `gh auth status` 要联网校验 token，
  // 在需要代理的网络里会失败，于是把「已登录」误判成「未登录」，白白挡住发版。
  if (process.env.GH_TOKEN || process.env.GITHUB_TOKEN) return;
  try {
    execSync("gh auth token", { stdio: "pipe" });
  } catch {
    console.error(
      "\ngh has no usable credentials. Run `gh auth login` or set GH_TOKEN / GITHUB_TOKEN.\n" +
        "Use --no-release to build locally without creating a Release.\n" +
        "If GitHub requires a proxy, configure HTTPS_PROXY / HTTP_PROXY before uploading."
    );
    process.exit(1);
  }
}

// ---------------------------------------------------------------- 主流程
const tag = resolveTag();
const version = tag.replace(new RegExp(`^${tagPrefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`), "");

console.log(`\n[publish-local] Repository: ${ROOT}`);
console.log(`[publish-local] Tag: ${tag}${dryRun ? "   (dry-run)" : ""}`);
console.log(`[publish-local] Build steps: ${steps.length}; artifact globs: ${artifactGlobs.length}`);

// steps 与 artifacts 都允许为空 —— 那意味着「纯文本发布」：只建带 release notes 的
// Release、不挂任何文件。文档站、纯库、只发版本说明的项目都属于这一类，
// "发布"不必等于"出安装包"。
if (steps.length === 0 && artifactGlobs.length === 0) {
  console.log(
    "\n[publish-local] No build steps or artifacts configured; creating a text-only Release with notes."
  );
}

// 1) 工作区
const status = execSync("git status --porcelain", { cwd: ROOT }).toString().trim();
if (status && !force) {
  console.error(
    "The worktree has uncommitted changes; build from a clean history so artifacts match the tag.\n" +
      "Commit or discard the changes, or use --force to continue.\n\n" +
      status
  );
  process.exit(1);
}

// 2) tag 必须已存在（本脚本不负责打 tag —— 那是 release.mjs 的事）
let hasLocalTag = false;
try {
  execSync(`git rev-parse "${tag}"`, { cwd: ROOT, stdio: "ignore" });
  hasLocalTag = true;
} catch {
  /* 不存在 */
}
if (!hasLocalTag) {
  console.error(
    `Local tag ${tag} does not exist. This script builds and creates Releases; release.mjs creates tags:\n` +
      `  node scripts/release.mjs ${version}\n` +
      "(It updates configured version files, commits, tags, and pushes the branch and tag.)"
  );
  process.exit(1);
}

// 3) tag 必须已在远程（Release 是挂在远程 tag 上的）
let hasRemoteTag = false;
try {
  const out = execSync(`git ls-remote --tags ${remote} "${tag}"`, { cwd: ROOT }).toString().trim();
  hasRemoteTag = out.length > 0;
} catch {
  /* 取不到就当没有 */
}
if (!hasRemoteTag && !noRelease) {
  console.error(
    `Remote ${remote} does not have tag ${tag}; a Release must reference a remote tag.\n` +
      `Push it first: git push ${remote} ${tag}`
  );
  process.exit(1);
}

// 4) 门禁
if (!skipVerify) {
  console.log("\n[publish-local] Running release gates...");
  if (dryRun) {
    console.log(`$ "${process.execPath}" "${path.join(__dirname, "verify.mjs")}"`);
  } else {
    execSync(`"${process.execPath}" "${path.join(__dirname, "verify.mjs")}"`, {
      cwd: ROOT,
      stdio: "inherit",
    });
  }
} else {
  console.log("\n[publish-local] Release gates skipped (--no-verify).");
}

// 5) 出包
console.log("\n[publish-local] Local build:");
for (const s of steps) {
  const name = s.name || s.cmd;
  console.log(`  - [${name}] ${s.cmd}${s.cwd ? `  (cwd: ${s.cwd})` : ""}`);
}
for (const s of steps) shStep(s.cmd, s.cwd);

// 6) 收集产物
const files = artifactGlobs.flatMap((g) => expandGlob(g)).filter((f, i, a) => a.indexOf(f) === i);
if (artifactGlobs.length === 0) {
  console.log("\n[publish-local] No artifacts configured; text-only Release.");
} else {
  console.log(`\n[publish-local] Artifacts:`);
  if (files.length === 0) {
    console.log(dryRun ? "  (dry-run: build not executed, so artifacts are not available)" : "  (no matches)");
    if (!dryRun) {
      console.warn(
        `  Warning: this may be expected when a platform cannot produce an artifact locally.\n` +
          `    Check the glob and continue if it is correct. Current glob: ${artifactGlobs.join(", ")}`
      );
    }
  } else {
    for (const f of files) {
      const size = fs.statSync(f).size;
      console.log(`  - ${rel(f)}  (${(size / 1024).toFixed(1)} KiB)`);
    }
  }
}

// 7) Release
if (noRelease) {
  console.log("\n[publish-local] --no-release selected; artifacts remain local and GitHub was not contacted.");
  process.exit(0);
}

ghReady();

// Release 是否已存在
let releaseExists = false;
try {
  execSync(`gh release view "${tag}"`, { cwd: ROOT, stdio: "ignore" });
  releaseExists = true;
} catch {
  /* 不存在 */
}

// release notes：--notes 指定文件 > CHANGELOG 对应段 > 交给 gh 自动生成
let notesFile = notesFileArg || null;
if (!notesFile) {
  const body = extractChangelogSection(version);
  if (body) {
    const tmp = path.join(ROOT, `.publish-local-notes-${version}.md`);
    if (!dryRun) fs.writeFileSync(tmp, body + "\n");
    notesFile = rel(tmp);
    console.log(`\n[publish-local] Release notes loaded from the ${version} section of CHANGELOG.md`);
  } else {
    console.log(`\n[publish-local] CHANGELOG.md has no ${version} section; gh will generate release notes`);
  }
}

if (dryRun) {
  console.log(
    `\n[dry-run] Would ${releaseExists ? "update the existing Release" : "create a Release"} ${tag} with ${files.length} artifact(s). No writes were performed.`
  );
  process.exit(0);
}

try {
  if (!releaseExists) {
    const quoted = files.map((f) => `"${f}"`).join(" ");
    const notesArgs = notesFile ? `--notes-file "${notesFile}"` : "--generate-notes";
    sh(`gh release create "${tag}" ${quoted} --title "${tag}" ${notesArgs} --latest`);
    console.log(
      `\n[publish-local] Release ${tag} created` +
        (files.length > 0 ? ` (${files.length} artifact(s))` : " (text-only, no artifacts)")
    );
  } else if (clobber) {
    const quoted = files.map((f) => `"${f}"`).join(" ");
    sh(`gh release upload "${tag}" ${quoted} --clobber`);
    if (notesFile) sh(`gh release edit "${tag}" --notes-file "${notesFile}"`);
    console.log(`\n[publish-local] Release ${tag} artifacts replaced (--clobber).`);
  } else {
    console.error(
      `\nRelease ${tag} already exists. Add --clobber to replace its artifacts.\n` +
        `Inspect it with: gh release view ${tag}`
    );
    process.exit(1);
  }
} finally {
  // 清掉临时 notes 文件，别留在工作区里
  const tmp = path.join(ROOT, `.publish-local-notes-${version}.md`);
  if (!notesFileArg && fs.existsSync(tmp)) fs.unlinkSync(tmp);
}

console.log(`\n[publish-local] Complete. View with: gh release view ${tag} --web`);
