#!/usr/bin/env node
// ============================================================================
// scripts/mirror-sync.mjs —— 把 GitHub Release 镜像同步到 Gitee / CNB
//
// 混合策略（方案 3）：GitHub 仍作为 CI 出包主力，Gitee/CNB 只做**分发镜像**。
// 用户在 GitHub 国内访问慢时，可从 Gitee/CNB 下载同一套产物。
//
// 流程：
//   1. 从 GitHub Release 读取 tag、title、body、assets（用 gh CLI）
//   2. push tag 到各镜像远程（git push <remote> <tag>）
//   3. 在 Gitee / CNB 创建同名 Release（调 API）
//   4. 从 GitHub 下载 assets → 上传到 Gitee / CNB
//
// 读项目根目录 pipeline.config.json 的 "mirrorSync" 段。
//
// 用法（在项目根目录）：
//   node scripts/mirror-sync.mjs                # sync the latest Release
//   node scripts/mirror-sync.mjs --tag v1.2.3   # select a tag
//   node scripts/mirror-sync.mjs --dry-run      # preview only
//   node scripts/mirror-sync.mjs --skip-assets  # sync tag and text only
//   node scripts/mirror-sync.mjs --force        # replace an existing mirror Release
//   node scripts/mirror-sync.mjs --only gitee   # select mirror platforms
// ============================================================================

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import https from "node:https";
import http from "node:http";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// ---------------------------------------------------------------- 参数
const args = process.argv.slice(2);
const KNOWN_FLAGS = ["--dry-run", "--skip-assets", "--force", "--tag", "--only", "--config"];
const unknownFlags = args.filter((a) => a.startsWith("--") && !KNOWN_FLAGS.includes(a));
if (unknownFlags.length > 0) {
  console.error(
    `Unknown argument(s): ${unknownFlags.join(" ")}\n` +
      `Available flags: ${KNOWN_FLAGS.join(" / ")} (--tag, --only, and --config need a value)`
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
const skipAssets = args.includes("--skip-assets");
const force = args.includes("--force");
const explicitTag = valueOf("--tag");
const onlyFilter = valueOf("--only"); // "gitee,cnb" 或单个
const configPathArg = valueOf("--config");

const configPath = path.resolve(ROOT, configPathArg || "pipeline.config.json");

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
const ms = config.mirrorSync && typeof config.mirrorSync === "object" ? config.mirrorSync : null;

if (!ms) {
  console.error(
    'pipeline.config.json has no "mirrorSync" section.\n' +
      "See pipeline.config.example.json for the mirrorSync configuration."
  );
  process.exit(1);
}

const gitee = ms.gitee || null;
const cnb = ms.cnb || null;

if (!gitee && !cnb) {
  console.error("mirrorSync must configure at least one of gitee or cnb.");
  process.exit(1);
}

// 过滤：--only gitee,cnb
const enabled = { gitee: !!gitee, cnb: !!cnb };
if (onlyFilter) {
  const wanted = onlyFilter.split(",").map((s) => s.trim().toLowerCase());
  for (const key of Object.keys(enabled)) {
    if (!wanted.includes(key)) enabled[key] = false;
  }
}

// ---------------------------------------------------------------- 工具函数
function sh(cmd, opts = {}) {
  console.log(`\n$ ${cmd}`);
  if (dryRun) return "";
  return execSync(cmd, { cwd: ROOT, stdio: "pipe", ...opts });
}

function shInherit(cmd, opts = {}) {
  console.log(`\n$ ${cmd}`);
  if (dryRun) return;
  execSync(cmd, { cwd: ROOT, stdio: "inherit", ...opts });
}

/**
 * HTTP/HTTPS JSON 请求（原生 fetch 可用：Node >= 22）
 */
async function fetchJSON(url, opts = {}) {
  const resp = await fetch(url, {
    ...opts,
    headers: { "Content-Type": "application/json", ...opts.headers },
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status} ${url}\n${text.slice(0, 500)}`);
  }
  return resp.json();
}

/**
 * multipart/form-data 上传文件（Gitee attach_files 用）
 */
async function uploadMultipart(url, fields, files) {
  const boundary = `----MirrorSync${Date.now()}`;
  const parts = [];

  for (const [key, val] of Object.entries(fields)) {
    parts.push(
      Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="${key}"\r\n\r\n${val}\r\n`
      )
    );
  }

  for (const file of files) {
    const data = fs.readFileSync(file.path);
    parts.push(
      Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="files[]"; filename="${file.name}"\r\n` +
          `Content-Type: application/octet-stream\r\n\r\n`
      )
    );
    parts.push(data);
    parts.push(Buffer.from("\r\n"));
  }

  parts.push(Buffer.from(`--${boundary}--\r\n`));
  const body = Buffer.concat(parts);

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": `multipart/form-data; boundary=${boundary}`,
      "Content-Length": String(body.length),
    },
    body,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status} ${url}\n${text.slice(0, 500)}`);
  }
  return resp.json();
}

/**
 * PUT 上传二进制流（CNB 附件上传用）
 */
async function uploadPUT(url, filePath) {
  const data = fs.readFileSync(filePath);
  const resp = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: data,
  });
  if (!resp.ok && resp.status !== 201) {
    const text = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status} ${url}\n${text.slice(0, 300)}`);
  }
  return resp;
}

/**
 * 下载文件到本地临时目录
 */
async function downloadFile(url, dest) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Download failed (${resp.status}): ${url}`);
  const buf = Buffer.from(await resp.arrayBuffer());
  fs.writeFileSync(dest, buf);
  return buf.length;
}

// ---------------------------------------------------------------- GitHub 数据源
function ghReady() {
  try {
    execSync("gh --version", { stdio: "pipe" });
  } catch {
    console.error(
      "\nThis script requires GitHub CLI (gh). Install it from https://cli.github.com and run `gh auth login`."
    );
    process.exit(1);
  }
}

function getLatestTag() {
  try {
    const out = execSync(`git tag --sort=-creatordate`, { cwd: ROOT }).toString().trim();
    const first = out.split("\n")[0];
    if (first) return first;
  } catch { /* ignore */ }
  return null;
}

function getGitHubRelease(tag) {
  try {
    const json = execSync(
      `gh release view "${tag}" --json tagName,name,body,assets`,
      { cwd: ROOT, encoding: "utf8" }
    );
    return JSON.parse(json);
  } catch (err) {
    console.error(`\nFailed to read GitHub Release ${tag}: ${err.message}`);
    console.error("Confirm that the tag is published on GitHub: gh release view " + tag);
    process.exit(1);
  }
}

// ---------------------------------------------------------------- Gitee 同步
async function syncToGitee(release, tmpDir) {
  const token = process.env[gitee.tokenEnv];
  if (!token) {
    console.error(`\n[Gitee] Environment variable ${gitee.tokenEnv} is not set; skipping.`);
    return false;
  }

  const owner = gitee.owner;
  const repoName = gitee.repo;
  const apiBase = "https://gitee.com/api/v5";
  const tagName = release.tagName;
  const name = release.name || tagName;
  const body = (release.body || "").slice(0, 50000); // Gitee body 限制
  const targetBranch = gitee.targetBranch || "master";

  console.log(`\n[Gitee] Syncing ${tagName} -> ${owner}/${repoName}`);

  // 1) push tag
  if (gitee.remote) {
    console.log(`[Gitee] git push ${gitee.remote} ${tagName}`);
    if (!dryRun) {
      try {
        shInherit(`git push "${gitee.remote}" "${tagName}"`);
      } catch {
        console.log(`[Gitee] Tag may already exist; continuing...`);
      }
    }
  }

  if (dryRun) {
    console.log(`[Gitee] (dry-run) Would create Release: ${name}`);
    return true;
  }

  // 2) 检查是否已存在
  try {
    const existing = await fetchJSON(
      `${apiBase}/repos/${owner}/${repoName}/releases/tags/${tagName}?access_token=${token}`
    );
    if (existing && existing.id) {
      if (force) {
        console.log(`[Gitee] Release already exists (id=${existing.id}); --force will delete it first.`);
        await fetchJSON(
          `${apiBase}/repos/${owner}/${repoName}/releases/${existing.id}?access_token=${token}`,
          { method: "DELETE" }
        );
      } else {
        console.log(`[Gitee] Release ${tagName} already exists (id=${existing.id}); skipping. Use --force to replace it.`);
        return true;
      }
    }
  } catch {
    // 404 = 不存在，正常继续
  }

  // 3) 创建 Release
  const createBody = new URLSearchParams({
    access_token: token,
    tag_name: tagName,
    name,
    body,
    target_branch: targetBranch,
  });
  const created = await fetchJSON(`${apiBase}/repos/${owner}/${repoName}/releases`, {
    method: "POST",
    body: createBody.toString(),
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  console.log(`[Gitee] [ok] Release created (id=${created.id})`);

  // 4) 上传 assets
  if (skipAssets || !release.assets || release.assets.length === 0) {
    console.log(`[Gitee] ${skipAssets ? "Asset upload skipped (--skip-assets)" : "No assets"}`);
    return true;
  }

  const localAssets = ms.localAssets || false;
  for (const asset of release.assets) {
    let localPath;
    if (localAssets) {
      // 从本地构建产物目录取（不重复下载）
      localPath = path.join(ROOT, "dist", asset.name);
      if (!fs.existsSync(localPath)) {
        console.log(`[Gitee] Local asset ${asset.name} not found; downloading from GitHub...`);
        localPath = null;
      }
    }
    if (!localPath) {
      localPath = path.join(tmpDir, asset.name);
      console.log(`[Gitee] Downloading ${asset.name}...`);
      const size = await downloadFile(asset.browser_download_url || asset.url, localPath);
      console.log(`[Gitee]   ${(size / 1024).toFixed(0)} KiB`);
    }

    console.log(`[Gitee] Uploading ${asset.name} -> release ${created.id}`);
    await uploadMultipart(
      `${apiBase}/repos/${owner}/${repoName}/releases/${created.id}/attach_files`,
      { access_token: token },
      [{ path: localPath, name: asset.name }]
    );
    console.log(`[Gitee]   [ok] ${asset.name}`);
  }

  console.log(`[Gitee] Sync complete: https://gitee.com/${owner}/${repoName}/releases/tag/${tagName}`);
  return true;
}

// ---------------------------------------------------------------- CNB 同步
async function syncToCNB(release, tmpDir) {
  const token = process.env[cnb.tokenEnv];
  if (!token) {
    console.error(`\n[CNB] Environment variable ${cnb.tokenEnv} is not set; skipping.`);
    return false;
  }

  const repoSlug = cnb.repo; // 如 "org/repo" 或 "user/repo"
  const apiBase = "https://api.cnb.cool";
  const authHeaders = { Authorization: `Bearer ${token}` };
  const tagName = release.tagName;
  const name = release.name || tagName;
  const body = release.body || "";

  console.log(`\n[CNB] Syncing ${tagName} -> ${repoSlug}`);

  // 1) push tag
  if (cnb.remote) {
    console.log(`[CNB] git push ${cnb.remote} ${tagName}`);
    if (!dryRun) {
      try {
        shInherit(`git push "${cnb.remote}" "${tagName}"`);
      } catch {
        console.log(`[CNB] Tag may already exist; continuing...`);
      }
    }
  }

  if (dryRun) {
    console.log(`[CNB] (dry-run) Would create Release: ${name}`);
    return true;
  }

  // 2) 检查是否已存在
  try {
    const existing = await fetchJSON(
      `${apiBase}/${repoSlug}/-/releases/tags/${tagName}`,
      { headers: authHeaders }
    );
    if (existing && existing.id) {
      if (force) {
        console.log(`[CNB] Release already exists (id=${existing.id}); --force will delete it first.`);
        await fetchJSON(`${apiBase}/${repoSlug}/-/releases/${existing.id}`, {
          method: "DELETE",
          headers: authHeaders,
        });
      } else {
        console.log(`[CNB] Release ${tagName} already exists (id=${existing.id}); skipping. Use --force to replace it.`);
        return true;
      }
    }
  } catch {
    // 404 = 不存在
  }

  // 3) 创建 Release
  const created = await fetchJSON(`${apiBase}/${repoSlug}/-/releases`, {
    method: "POST",
    headers: authHeaders,
    body: JSON.stringify({
      tag_name: tagName,
      name,
      body,
      target_commitish: cnb.targetBranch || "main",
      prerelease: false,
    }),
  });
  console.log(`[CNB] [ok] Release created (id=${created.id})`);

  // 4) 上传 assets
  if (skipAssets || !release.assets || release.assets.length === 0) {
    console.log(`[CNB] ${skipAssets ? "Asset upload skipped (--skip-assets)" : "No assets"}`);
    return true;
  }

  const localAssets = ms.localAssets || false;
  for (const asset of release.assets) {
    let localPath;
    if (localAssets) {
      localPath = path.join(ROOT, "dist", asset.name);
      if (!fs.existsSync(localPath)) {
        console.log(`[CNB] Local asset ${asset.name} not found; downloading from GitHub...`);
        localPath = null;
      }
    }
    if (!localPath) {
      localPath = path.join(tmpDir, asset.name);
      console.log(`[CNB] Downloading ${asset.name}...`);
      const size = await downloadFile(asset.browser_download_url || asset.url, localPath);
      console.log(`[CNB]   ${(size / 1024).toFixed(0)} KiB`);
    }

    // CNB 的 asset 上传是两步：先请求 upload-url，再 PUT 文件
    console.log(`[CNB] Uploading ${asset.name} -> release ${created.id}`);
    const uploadInfo = await fetchJSON(
      `${apiBase}/${repoSlug}/-/releases/${created.id}/asset-upload-url`,
      {
        method: "POST",
        headers: { ...authHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ name: asset.name }),
      }
    );
    // uploadInfo 包含 upload_url 和 upload_token + asset_path
    await uploadPUT(uploadInfo.upload_url || uploadInfo.url, localPath);
    // 确认上传完成
    if (uploadInfo.upload_token) {
      await fetchJSON(
        `${apiBase}/${repoSlug}/-/releases/${created.id}/asset-upload-confirmation/${uploadInfo.upload_token}/${encodeURIComponent(asset.name)}`,
        { method: "POST", headers: authHeaders }
      );
    }
    console.log(`[CNB]   [ok] ${asset.name}`);
  }

  console.log(`[CNB] Sync complete: https://cnb.cool/${repoSlug}/-/releases/tag/${tagName}`);
  return true;
}

// ---------------------------------------------------------------- 主流程
ghReady();

// 推导 tag
const tag = explicitTag
  ? (explicitTag.startsWith(tagPrefix) ? explicitTag : `${tagPrefix}${explicitTag}`)
  : getLatestTag();

if (!tag) {
  console.error("Cannot resolve a tag: no local git tag exists; pass --tag explicitly.");
  process.exit(1);
}

console.log(`\n[mirror-sync] Repository: ${ROOT}`);
console.log(`[mirror-sync] Tag: ${tag}${dryRun ? "   (dry-run)" : ""}`);
console.log(`[mirror-sync] Mirror targets: ${Object.entries(enabled).filter(([, v]) => v).map(([k]) => k).join(", ")}`);

// 读 GitHub Release
const release = getGitHubRelease(tag);
console.log(`[mirror-sync] GitHub Release: ${release.name || tag}, ${(release.assets || []).length} asset(s)`);

// 临时目录用于下载 assets
const tmpDir = path.join(ROOT, `.mirror-sync-tmp-${Date.now()}`);
if (!dryRun && !skipAssets && (release.assets || []).length > 0) {
  fs.mkdirSync(tmpDir, { recursive: true });
}

let ok = true;
try {
  if (enabled.gitee) {
    const result = await syncToGitee(release, tmpDir);
    if (!result) ok = false;
  }
  if (enabled.cnb) {
    const result = await syncToCNB(release, tmpDir);
    if (!result) ok = false;
  }
} catch (err) {
  console.error(`\n[mirror-sync] [error] ${err.message}`);
  ok = false;
} finally {
  // 清理临时目录
  if (!dryRun && fs.existsSync(tmpDir)) {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

if (ok) {
  console.log(`\n[mirror-sync] [ok] All mirrors synced.`);
} else {
  console.error(`\n[mirror-sync] Some mirrors failed; see the log above.`);
  process.exit(1);
}
