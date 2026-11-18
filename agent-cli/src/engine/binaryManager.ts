import { createWriteStream } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { brandEnv } from '../util/env.js';
import { chmod, copyFile, mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, join, normalize, relative, sep } from 'node:path';
import AdmZip from 'adm-zip';
import * as tar from 'tar';
import { binDir, cacheDir } from '../config.js';
import { USER_AGENT } from '../version.js';
import { atomicWrite, pathExists, safeJoin, sha256File } from '../util/fs.js';
import { fetchJson, type HttpError } from '../util/http.js';
import { execFile, which } from '../util/exec.js';
import { engineEnv } from './llamaRunner.js';
import { detectEngineHost, engineCandidates, explainEngineFailure, parseDeviceList, parseVariant, type EngineCandidate, type EngineHost, type LlamaVariant } from './engineHost.js';
import { ProgressBar } from '../ui/progress.js';
import type { Logger } from '../logger.js';
import type { GpuSnapshot, GpuVendor } from '../hardware/types.js';
import { directoryBytes, diskWriteAllowance, requireDiskSpace, storagePolicy } from '../util/storage.js';

const MAX_ARCHIVE_BYTES = 2 * 1024 ** 3;
const MAX_ENGINE_CACHE_BYTES = 10 * 1024 ** 3;

export type { LlamaVariant } from './engineHost.js';

export type LlamaInstall = {
  path: string;
  version: string;
  variant: LlamaVariant;
  /** `portable` builds come from this project's engine releases (older Linux). */
  build?: 'official' | 'portable';
  cuda?: number;
  sha256: string | null;
  binarySha256?: string | null;
  verified: boolean;
  source: 'env' | 'path' | 'manifest' | 'download';
  dir?: string;
};

export type ReleaseAsset = {
  name: string;
  browser_download_url: string;
  size?: number;
  digest?: string | null;
};

export type ReleaseInfo = {
  tag_name: string;
  name?: string;
  published_at?: string;
  assets: ReleaseAsset[];
};

export type InstallOptions = {
  platform?: NodeJS.Platform;
  arch?: string;
  env?: NodeJS.ProcessEnv;
  home?: string;
  gpu?: GpuSnapshot | null;
  logger?: Logger;
  fetchImpl?: typeof fetch;
  showProgress?: boolean;
  stream?: NodeJS.WriteStream;
  requireChecksum?: boolean;
  release?: ReleaseInfo;
  version?: string;
  /** The build to install; by default the official build for the GPU vendor. */
  candidate?: EngineCandidate;
  /** false: record the install but do not make it the engine yet (it is checked first). */
  activate?: boolean;
};

export type EnsureResult =
  | { found: true; install: LlamaInstall; warning?: string }
  | { found: false; reason: string; install?: undefined };

type Manifest = {
  installs: LlamaInstall[];
  latest?: string;
  /** Builds that were downloaded but could not run here, by kind ("cuda12", "portable-cpu"). */
  failed?: Record<string, { reason: string; at: number }>;
};

const RELEASE_CACHE_MS = 6 * 60 * 60 * 1000;
const RELEASE_URL = 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest';
const RELEASE_LIST_URL = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20';
const RELEASE_URL_FALLBACK = 'https://api.github.com/repos/ggerganov/llama.cpp/releases/latest';
/**
 * Portable Linux builds made by .github/workflows/engine.yml, published as `engine-<llama.cpp tag>`
 * releases. This CLI version asks for ENGINE_TAG first, and for the newest engine release after it.
 */
export const ENGINE_TAG = 'engine-b12503';
const ENGINE_REPO_API = 'https://api.github.com/repos/somemusic69-sudo/Sublera-cli/releases';
/** A failed engine build is not downloaded again for this long, except by `doctor --fix`. */
const FAILED_RETRY_MS = 7 * 24 * 60 * 60 * 1000;

export function variantFor(platform: NodeJS.Platform, arch: string, vendor: GpuVendor | null | undefined): LlamaVariant {
  if (platform === 'darwin') return arch === 'arm64' ? 'metal' : 'cpu';
  if (vendor === 'nvidia') return 'cuda';
  if (vendor === 'amd' || vendor === 'intel') return 'vulkan';
  return 'cpu';
}

function asCandidate(variant: LlamaVariant | EngineCandidate): EngineCandidate {
  return typeof variant === 'string' ? { variant, source: 'official', label: `official ${variant} build` } : variant;
}

/** Names an install directory and release cache entry, such as "cuda12" or "portable-cpu". */
function candidateKey(candidate: EngineCandidate): string {
  return `${candidate.source === 'portable' ? 'portable-' : ''}${candidate.variant}${candidate.cuda ?? ''}`;
}

function patternsFor(platform: NodeJS.Platform, arch: string, candidate: EngineCandidate): RegExp[] {
  const cpu = arch === 'arm64' ? 'arm64' : 'x64';
  const archive = (body: string) => new RegExp(`${body}\\.(zip|tar\\.gz)$`, 'i');
  if (candidate.source === 'portable') {
    if (platform !== 'linux') return [];
    const kind = candidate.variant === 'cuda' ? `cuda-${candidate.cuda ?? '\\d+'}(\\.[\\d.]+)?` : candidate.variant;
    return [archive(`^sublera-llama-[\\w.-]+-linux-${kind}-${cpu}`)];
  }
  if (platform === 'darwin') {
    return arch === 'arm64'
      ? [/macos-arm64\.(zip|tar\.gz)$/i, /bin-macos-arm64\.(zip|tar\.gz)$/i, /macos.*arm64.*\.(zip|tar\.gz)$/i]
      : [/macos-x64\.(zip|tar\.gz)$/i, /bin-macos-x64\.(zip|tar\.gz)$/i];
  }
  const os = platform === 'win32' ? 'win' : '(ubuntu|linux)';
  switch (candidate.variant) {
    case 'cuda': {
      const versioned = archive(`-${os}-cuda-(cu)?${candidate.cuda ?? '\\d+'}(\\.[\\d.]+)?-${cpu}`);
      // Old releases had one CUDA build without a version in its name.
      return candidate.cuda && candidate.cuda !== 12 ? [versioned] : [versioned, archive(`-${os}-cuda-${cpu}`)];
    }
    case 'vulkan': return [archive(`-${os}-vulkan-${cpu}`)];
    case 'rocm': return [archive(`-${os}-(rocm|hip)(-[\\w.]+)?-${cpu}`)];
    case 'metal': return [];
    default: return platform === 'win32' ? [archive(`-win-cpu-${cpu}`)] : [archive(`-${os}(-[\\d.]+)?-${cpu}`)];
  }
}

export function pickReleaseAsset(release: ReleaseInfo, platform: NodeJS.Platform, arch: string, variant: LlamaVariant | EngineCandidate): { main: ReleaseAsset | null; companions: ReleaseAsset[] } {
  const candidate = asCandidate(variant);
  const assets = release.assets.filter((a) => !/\.sha256(\.txt)?$/i.test(a.name) && !/^cudart/i.test(a.name));
  let main: ReleaseAsset | null = null;
  for (const pattern of patternsFor(platform, arch, candidate)) {
    const found = assets.find((a) => pattern.test(a.name));
    if (found) { main = found; break; }
  }
  const cpu = arch === 'arm64' ? /(arm64|aarch64)/i : /(x64|x86_64)/i;
  const cudaVersion = main?.name.match(/cuda-([\d.]+)-(x64|arm64)/i)?.[1] || null;
  // Portable builds carry their CUDA runtime in the main archive.
  const companions = main && candidate.variant === 'cuda' && candidate.source === 'official'
    ? release.assets.filter((a) => {
      if (!/^cudart/i.test(a.name) || /\.sha256/i.test(a.name)) return false;
      if (!cpu.test(a.name)) return false;
      if (platform === 'win32' && !/win/i.test(a.name)) return false;
      if (platform === 'linux' && !/(ubuntu|linux)/i.test(a.name)) return false;
      if (platform === 'darwin') return false;
      if (cudaVersion && !a.name.includes(`cuda-${cudaVersion}-`)) return false;
      return true;
    })
    : [];
  return { main, companions };
}

function releaseCachePath(env: NodeJS.ProcessEnv, key = 'default'): string {
  return join(cacheDir(env), `llama-release-${key}.json`);
}

async function readReleaseCache(env: NodeJS.ProcessEnv, key: string): Promise<ReleaseInfo | null> {
  try {
    const raw = await readFile(releaseCachePath(env, key), 'utf8');
    const data = JSON.parse(raw) as { fetchedAt: number; release: ReleaseInfo };
    if (!data || Date.now() - data.fetchedAt > RELEASE_CACHE_MS || !data.release?.assets?.length) return null;
    return data.release;
  } catch {
    return null;
  }
}

const LIST_MEMO_MS = 5 * 60 * 1000;
const listMemo = new WeakMap<typeof fetch, Map<string, { at: number; list: Promise<ReleaseInfo[]> }>>();

/** One GitHub API answer as a list of releases. Successful answers are shared for a few minutes. */
function releaseList(url: string, fetchImpl: typeof fetch): Promise<ReleaseInfo[]> {
  let memo = listMemo.get(fetchImpl);
  if (!memo) listMemo.set(fetchImpl, memo = new Map());
  const hit = memo.get(url);
  if (hit && Date.now() - hit.at < LIST_MEMO_MS) return hit.list;
  const list = (async () => {
    const response = await fetchImpl(url, { headers: { 'user-agent': USER_AGENT, accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(20_000) });
    if (response.status === 404) return [];
    if (!response.ok) {
      const limited = (response.status === 403 || response.status === 429) && /rate limit/i.test(await response.text().catch(() => ''));
      throw new Error(limited ? 'the GitHub API rate limit was reached; try again later' : `the GitHub API answered HTTP ${response.status}`);
    }
    const data = await response.json() as ReleaseInfo | ReleaseInfo[];
    return Array.isArray(data) ? data : [data];
  })();
  memo.set(url, { at: Date.now(), list });
  list.catch(() => { if (memo.get(url)?.list === list) memo.delete(url); });
  return list;
}

async function fetchRelease(options: InstallOptions, platform: NodeJS.Platform, arch: string, candidate: EngineCandidate): Promise<ReleaseInfo> {
  if (options.release) return options.release;
  const env = options.env ?? process.env;
  const cacheKey = `${platform}-${arch}-${candidateKey(candidate)}`;
  const cached = await readReleaseCache(env, cacheKey);
  if (cached && pickReleaseAsset(cached, platform, arch, candidate).main) return cached;
  const fetchImpl = options.fetchImpl ?? fetch;
  const portable = candidate.source === 'portable';
  // llama.cpp publishes its builds as pre-releases, which "latest" does not return.
  const urls = portable
    ? [`${ENGINE_REPO_API}/tags/${ENGINE_TAG}`, `${ENGINE_REPO_API}?per_page=100`]
    : [RELEASE_LIST_URL, RELEASE_URL, RELEASE_URL_FALLBACK];
  let release: ReleaseInfo | null = null;
  let problem: string | null = null;
  for (const url of urls) {
    try {
      const list = await releaseList(url, fetchImpl);
      release = list.find((item) => item?.tag_name && Array.isArray(item.assets) && item.assets.length > 0
        && (!portable || item.tag_name.startsWith('engine-'))
        && pickReleaseAsset(item, platform, arch, candidate).main) || null;
      if (release) break;
    } catch (error) {
      problem ??= error instanceof Error ? error.message : String(error);
    }
  }
  if (!release) {
    if (problem) throw new Error(`Could not read the ${portable ? 'Sublera engine' : 'llama.cpp'} releases: ${problem}`);
    throw new Error(portable
      ? `No ${candidate.label} is published yet for ${platform}/${arch}`
      : `No llama.cpp release with a ${candidate.label} was found for ${platform}/${arch}`);
  }
  await mkdir(cacheDir(env), { recursive: true });
  await writeFile(releaseCachePath(env, cacheKey), JSON.stringify({ fetchedAt: Date.now(), release }), 'utf8');
  return release;
}

async function checksumFor(asset: ReleaseAsset, release: ReleaseInfo, options: InstallOptions, isMain: boolean): Promise<{ expected: string | null; published: boolean }> {
  const env = options.env ?? process.env;
  const override = brandEnv(env, 'SUBLET_LLAMA_SHA256');
  if (isMain && override && /^[a-f0-9]{64}$/i.test(override.trim())) return { expected: override.trim().toLowerCase(), published: true };
  if (asset.digest && /^sha256:[a-f0-9]{64}$/i.test(asset.digest)) return { expected: asset.digest.slice(7).toLowerCase(), published: true };
  const sidecar = release.assets.find((a) => new RegExp(`^${escapeRegExp(asset.name)}\\.sha256(?:\\.txt)?$`, 'i').test(a.name));
  if (sidecar) {
    try {
      const fetchImpl = options.fetchImpl ?? fetch;
      const response = await fetchImpl(sidecar.browser_download_url, { headers: { 'user-agent': USER_AGENT }, signal: AbortSignal.timeout(20_000) });
      const text = await response.text();
      const match = text.match(/\b([a-f0-9]{64})\b/i);
      if (match?.[1]) return { expected: match[1].toLowerCase(), published: true };
    } catch {
      /* no published digest */
    }
  }
  return { expected: null, published: false };
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function downloadFile(url: string, dest: string, options: InstallOptions & { label: string }): Promise<string> {
  await mkdir(dirname(dest), { recursive: true });
  const fetchImpl = options.fetchImpl ?? fetch;
  const bar = options.showProgress === false ? null : new ProgressBar({
    label: options.label,
    total: 0,
    enabled: options.showProgress ?? Boolean((options.stream ?? process.stderr).isTTY),
    stream: options.stream ?? process.stderr,
  });
  let lastError: unknown;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetchImpl(url, { redirect: 'follow', headers: { 'user-agent': USER_AGENT }, signal: AbortSignal.timeout(30 * 60_000) });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const total = Number(response.headers.get('content-length') || 0) || null;
      const reserve = storagePolicy(options.env).reserveBytes;
      const budget = Math.min(MAX_ARCHIVE_BYTES, await diskWriteAllowance(dirname(dest), reserve));
      if (budget <= 0 || (total != null && (!Number.isSafeInteger(total) || total < 0 || total > budget))) {
        await response.body.cancel().catch(() => undefined);
        throw new Error('Engine archive exceeds storage budget (maximum 2 GiB), or insufficient free disk space');
      }
      if (total) bar?.setTotal(total);
      const writer = createWriteStream(dest, { flags: 'w' });
      let received = 0;
      const reader = response.body.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value?.length) {
            // Free space can shrink while streaming (other downloads/processes).
            // Compare remaining allowance with this chunk plus writes still buffered,
            // not all bytes received: flushed bytes already reduced free disk space.
            const allowance = await diskWriteAllowance(dirname(dest), reserve);
            if (received + value.length > budget || value.length + writer.writableLength > allowance) {
              throw new Error('Engine archive exceeds storage budget');
            }
            if (!writer.write(Buffer.from(value))) {
              await new Promise<void>((resolve, reject) => {
                const onDrain = () => { cleanup(); resolve(); };
                const onError = (error: Error) => { cleanup(); reject(error); };
                const cleanup = () => {
                  writer.off('drain', onDrain);
                  writer.off('error', onError);
                };
                writer.once('drain', onDrain);
                writer.once('error', onError);
              });
            }
            received += value.length;
            bar?.set(received, total ?? undefined);
          }
        }
        await new Promise<void>((resolve, reject) => writer.end((error?: Error | null) => error ? reject(error) : resolve()));
      } catch (error) {
        writer.destroy();
        throw error;
      } finally {
        await reader.cancel().catch(() => undefined);
        reader.releaseLock();
      }
      bar?.done();
      return dest;
    } catch (error) {
      lastError = error;
      bar?.fail();
      await rm(dest, { force: true });
      if (attempt < 3) await new Promise((r) => setTimeout(r, 750 * attempt));
    }
  }
  throw new Error(`Could not download ${basename(url)}: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}

export async function extractArchive(archive: string, destination: string): Promise<void> {
  await mkdir(destination, { recursive: true });
  const lower = archive.toLowerCase();
  if (lower.endsWith('.zip')) {
    const zip = new AdmZip(archive);
    const entries = zip.getEntries();
    if (entries.length > 10_000) throw new Error('Archive has too many entries');
    let uncompressed = 0;
    for (const entry of entries) {
      uncompressed += Number(entry.header.size || 0);
      if (uncompressed > 4 * 1024 * 1024 * 1024) throw new Error('Archive expands beyond the 4GB safety limit');
      if (entry.isDirectory) {
        await mkdir(safeJoin(destination, entry.entryName), { recursive: true });
        continue;
      }
      const target = safeJoin(destination, entry.entryName);
      await mkdir(join(target, '..'), { recursive: true });
      const data = entry.getData();
      await writeFile(target, data);
      if (/[\\/](llama-server|server)(\.exe)?$/i.test(target)) {
        await chmod(target, 0o755).catch(() => undefined);
      }
    }
    return;
  }
  if (lower.endsWith('.tar.gz') || lower.endsWith('.tgz')) {
    // Linux and macOS builds ship library link chains (libllama.so -> libllama.so.0 -> libllama.so.0.9.0).
    // tar refuses to create a link through another link, so links are skipped here and copied as files below.
    const links = new Map<string, { linkpath: string; hard: boolean }>();
    let entries = 0;
    let expanded = 0;
    await tar.x({
      file: archive,
      cwd: destination,
      strict: true,
      filter: (path, entry) => {
        safeJoin(destination, path);
        entries++;
        expanded += Number(entry.size || 0);
        if (entries > 10_000 || !Number.isSafeInteger(expanded) || expanded > 4 * 1024 ** 3) throw new Error('Archive exceeds entry or 4 GiB expansion limit');
        const type = 'type' in entry ? entry.type : undefined;
        if (type === 'SymbolicLink' || type === 'Link') {
          links.set(path.replace(/\/+$/, ''), { linkpath: String((entry as { linkpath?: string }).linkpath ?? ''), hard: type === 'Link' });
          return false;
        }
        return true;
      },
    });
    await materializeLinks(destination, links);
    return;
  }
  throw new Error(`Unsupported archive type: ${basename(archive)}`);
}

/** Copies the file each archive link points to, following link chains, and never outside the destination. */
async function materializeLinks(destination: string, links: Map<string, { linkpath: string; hard: boolean }>): Promise<void> {
  let expanded = await directoryBytes(destination);
  const resolveLink = (path: string, depth = 0): string => {
    const link = links.get(path);
    if (!link) return path;
    if (depth > 16) throw new Error(`Archive link loop at ${path}`);
    const target = link.hard
      ? normalize(link.linkpath)
      : normalize(join(dirname(path), link.linkpath));
    safeJoin(destination, target);
    return resolveLink(target.split(sep).join('/'), depth + 1);
  };
  for (const path of links.keys()) {
    const source = safeJoin(destination, resolveLink(path));
    const info = await stat(source).catch(() => null);
    if (!info?.isFile()) continue;
    expanded += info.size;
    if (expanded > 4 * 1024 ** 3) throw new Error('Archive links exceed the 4 GiB expansion limit');
    const target = safeJoin(destination, path);
    await mkdir(dirname(target), { recursive: true });
    await copyFile(source, target);
    await chmod(target, info.mode & 0o777).catch(() => undefined);
  }
}

async function locateServerBinary(dir: string): Promise<string | null> {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isFile() && /^(llama-server|server)(\.exe)?$/i.test(entry.name)) return full;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const found = await locateServerBinary(join(dir, entry.name));
    if (found) return found;
  }
  return null;
}

function manifestPath(env: NodeJS.ProcessEnv): string {
  return join(binDir(env), 'manifest.json');
}

async function readManifest(env: NodeJS.ProcessEnv): Promise<Manifest> {
  try {
    const data = JSON.parse(await readFile(manifestPath(env), 'utf8')) as Manifest;
    const failed = data.failed && typeof data.failed === 'object' && !Array.isArray(data.failed) ? data.failed : undefined;
    return { installs: Array.isArray(data.installs) ? data.installs : [], latest: data.latest, ...(failed ? { failed } : {}) };
  } catch {
    return { installs: [] };
  }
}

async function writeManifest(env: NodeJS.ProcessEnv, manifest: Manifest): Promise<void> {
  await mkdir(binDir(env), { recursive: true });
  await atomicWrite(manifestPath(env), JSON.stringify(manifest, null, 2));
}

export async function findExistingLlama(options: InstallOptions = {}): Promise<LlamaInstall | null> {
  const env = options.env ?? process.env;
  const explicit = env.LLAMA_SERVER;
  if (explicit && await pathExists(explicit)) {
    return { path: explicit, version: 'env', variant: 'cpu', sha256: null, verified: false, source: 'env' };
  }
  const manifest = await readManifest(env);
  if (manifest.latest) {
    const install = manifest.installs.find((i) => i.path === manifest.latest);
    if (install && await pathExists(install.path)) {
      if (!install.binarySha256 || await sha256File(install.path) !== install.binarySha256) return null;
      return { ...install, source: 'manifest' };
    }
  }
  const fromPath = await which('llama-server');
  if (fromPath) return { path: fromPath, version: 'path', variant: 'cpu', sha256: null, verified: false, source: 'path' };
  return null;
}

function oneLine(text: string): string {
  return text.replace(/[\x00-\x1f\x7f]+/g, ' ').replace(/\s+/g, ' ').trim();
}

const GPU_NAMES: Record<LlamaVariant, string> = { cuda: 'CUDA', vulkan: 'Vulkan', rocm: 'ROCm', metal: 'Metal', cpu: 'CPU' };

/**
 * Check the loader and shared libraries before reporting an engine as usable. For a GPU build,
 * also check that the engine finds a GPU: llama.cpp skips a GPU backend it cannot load and
 * silently runs on the CPU.
 */
export async function probeLlamaServer(binary: string, env: NodeJS.ProcessEnv = process.env, variant?: LlamaVariant, host?: EngineHost | null): Promise<string | null> {
  const run = (args: string[]) => execFile(binary, args, {
    timeoutMs: 30_000, maxOutputBytes: 65_536,
    env: engineEnv(env).env, inheritEnv: false,
  });
  const result = await run(['--version']);
  if (result.code !== 0 || result.timedOut || result.error) {
    const raw = result.error || result.stderr || result.stdout || `exit code ${result.code}`;
    const hint = explainEngineFailure(raw, host) ?? 'Check system libraries and install a compatible llama-server; set LLAMA_SERVER to its path.';
    return `llama-server could not start: ${oneLine(raw).slice(0, 600)}. ${hint}`;
  }
  if (!variant || variant === 'cpu') return null;
  const devices = await run(['--list-devices']);
  // Builds without --list-devices cannot be checked here; the model load still shows a problem.
  if (devices.code !== 0 || devices.timedOut || devices.error || !devices.stdout.includes('Available devices:')) return null;
  if (parseDeviceList(devices.stdout).length) return null;
  const log = oneLine(devices.stderr.split(/\r?\n/).filter((line) => /fail|error|cannot|not found/i.test(line)).join(' ')).slice(0, 400);
  const hint = explainEngineFailure(devices.stderr, host);
  return `llama-server started, but its ${GPU_NAMES[variant]} backend found no GPU${log ? ` (${log})` : ''}.${hint ? ` ${hint}` : ''}`;
}

function describeInstall(install: LlamaInstall): string {
  if (install.source === 'env') return 'LLAMA_SERVER';
  if (install.source === 'path') return 'llama-server on PATH';
  return `installed ${install.build === 'portable' ? 'portable' : 'official'} ${GPU_NAMES[install.variant] ?? install.variant}${install.cuda ? ` ${install.cuda}` : ''} build`;
}

function installKey(install: LlamaInstall): string {
  return `${install.build === 'portable' ? 'portable-' : ''}${install.variant}${install.cuda ?? ''}`;
}

export type EnsureOptions = InstallOptions & {
  fix?: boolean;
  simulation?: boolean;
  host?: EngineHost;
  /** Also try builds that failed here in the last days (`doctor --fix`). */
  retryFailed?: boolean;
};

export async function ensureLlamaServer(options: EnsureOptions = {}): Promise<EnsureResult> {
  if (options.simulation) return { found: false, reason: 'Simulation mode: llama-server is not needed.' };
  const env = options.env ?? process.env;
  const existing = await findExistingLlama(options);
  let fallback: LlamaInstall | null = null;
  let existingProblem: string | null = null;
  const failures: string[] = [];
  let host: EngineHost | null = options.host ?? null;
  const getHost = async () => host ??= await detectEngineHost({ platform: options.platform, arch: options.arch, gpu: options.gpu, env });
  const forcedText = brandEnv(env, 'SUBLET_LLAMA_VARIANT');
  const forced = parseVariant(forcedText);
  const badVariant = forcedText && !forced ? `SUBLERA_LLAMA_VARIANT=${forcedText} is not one of cuda, vulkan, rocm, metal, cpu; it was ignored.` : null;
  // On a computer with a GPU, a CPU engine is a last resort, unless the host asked for it.
  const gpu = forced === 'cpu' ? null : options.gpu?.primary;
  const gpuHost = Boolean(gpu && ['nvidia', 'amd', 'intel'].includes(gpu.vendor));
  const found = (install: LlamaInstall, warning?: string): EnsureResult => {
    const text = [badVariant, warning].filter(Boolean).join(' ');
    return { found: true, install, ...(text ? { warning: text } : {}) };
  };
  if (existing) {
    // An installed build of another kind than SUBLERA_LLAMA_VARIANT asks for is replaced.
    const wrongKind = Boolean(forced && existing.source === 'manifest' && existing.variant !== forced && !options.candidate);
    if (wrongKind && !options.fix) {
      return { found: false, reason: `The installed engine is a ${describeInstall(existing)}, but SUBLERA_LLAMA_VARIANT=${forced}. Run: sublera doctor --fix` };
    }
    if (wrongKind) {
      failures.push(`${describeInstall(existing)}: not used, SUBLERA_LLAMA_VARIANT=${forced}`);
    } else if (!options.requireChecksum || existing.verified) {
      const problem = await probeLlamaServer(existing.path, env, existing.variant, existing.source === 'manifest' ? await getHost() : null);
      if (!problem) {
        // A CPU build from an earlier install is kept, but a GPU build is tried first when one can run.
        if (!(options.fix && gpuHost && existing.source === 'manifest' && existing.variant === 'cpu' && !options.candidate)) {
          return found(existing);
        }
        fallback = existing;
      } else if (!options.fix || existing.source === 'env') {
        return { found: false, reason: existing.source === 'manifest' ? `${problem} Run: sublera doctor --fix to try other engine builds.` : problem };
      } else {
        existingProblem = problem;
        failures.push(`${describeInstall(existing)}: ${problem}`);
      }
    } else if (!options.fix || existing.source === 'env' || existing.source === 'path') {
      return { found: false, reason: 'The existing llama-server has no verified release checksum; --require-checksum refuses it. Remove custom LLAMA_SERVER/PATH overrides and run sublera doctor --fix --require-checksum.' };
    }
  }
  if (!options.fix) return { found: false, reason: 'llama-server is not installed. Run: sublera doctor --fix' };
  if (options.candidate) {
    // One requested build, as before candidates existed.
    try {
      const install = await installLlamaServer({ ...options, activate: false });
      const problem = await probeLlamaServer(install.path, env, install.variant, await getHost());
      if (problem) return { found: false, reason: problem };
      return found(await activateInstall(env, install));
    } catch (error) {
      return { found: false, reason: error instanceof Error ? error.message : String(error) };
    }
  }
  const resolved = await getHost();
  const plan = engineCandidates(resolved, forced);
  const failed = (await readManifest(env)).failed ?? {};
  const skipped = plan.skipped.map((s) => `${s.label}: skipped, ${s.reason}`);
  const cpuWarning = () => `llama-server runs on the CPU, not on the ${gpu?.name || 'GPU'}: no GPU build of llama.cpp can run here. ${[...failures, ...skipped].join('; ')}`;
  for (const candidate of fallback ? plan.candidates.filter((c) => c.variant !== 'cpu') : plan.candidates) {
    const key = candidateKey(candidate);
    const earlier = failed[key];
    if (earlier && !options.retryFailed && Date.now() - earlier.at < FAILED_RETRY_MS) {
      failures.push(`${candidate.label}: failed here on ${new Date(earlier.at).toISOString().slice(0, 10)} (${earlier.reason}); sublera doctor --fix tries it again`);
      continue;
    }
    let install: LlamaInstall;
    try {
      install = await installLlamaServer({ ...options, candidate, activate: false });
    } catch (error) {
      failures.push(`${candidate.label}: ${oneLine(error instanceof Error ? error.message : String(error))}`);
      continue;
    }
    // The install that already failed above is not checked or reported twice.
    const again = existingProblem != null && install.path === existing?.path;
    const problem = again ? existingProblem : await probeLlamaServer(install.path, env, candidate.variant, resolved);
    if (problem) {
      if (!again) failures.push(`${candidate.label}: ${problem}`);
      options.logger?.warn(`The ${candidate.label} of llama.cpp cannot run here; trying the next build.`, { reason: problem });
      // Remember the failure and free the space, so a new llama.cpp release does not download it again.
      await discardInstall(env, install, key, problem);
      continue;
    }
    const active = await activateInstall(env, install, key);
    if (existingProblem != null && existing?.source === 'manifest' && existing.path !== active.path) await discardInstall(env, existing, installKey(existing), null);
    return found(active, candidate.variant === 'cpu' && gpuHost ? cpuWarning() : undefined);
  }
  if (fallback) return found(fallback, cpuWarning());
  return {
    found: false,
    reason: `No llama.cpp build can run on this computer. ${[...failures, ...skipped].join('; ')}. Fix one of these problems and run sublera doctor --fix, or build llama-server for this system and set LLAMA_SERVER to its path.`,
  };
}

const pendingInstalls = new Map<string, Promise<unknown>>();

/** Runs one engine-directory transaction at a time per SUBLET_HOME in this process. */
function queued<T>(env: NodeJS.ProcessEnv, task: () => Promise<T>): Promise<T> {
  const home = binDir(env);
  const key = process.platform === 'win32' ? home.toLowerCase() : home;
  const previous = pendingInstalls.get(key);
  const pending = (previous ?? Promise.resolve()).catch(() => undefined).then(task);
  pendingInstalls.set(key, pending);
  return pending.finally(() => {
    if (pendingInstalls.get(key) === pending) pendingInstalls.delete(key);
  });
}

// The caller owns the cross-process instance lock (the running agent already holds it).
// Queue the entire read/modify/write transaction, not just the manifest write.
export async function installLlamaServer(options: InstallOptions = {}): Promise<LlamaInstall> {
  const env = { ...(options.env ?? process.env) };
  return queued(env, () => installLlamaServerUnlocked({ ...options, env }));
}

/** Makes a checked install the engine that later runs use. */
async function activateInstall(env: NodeJS.ProcessEnv, install: LlamaInstall, key?: string): Promise<LlamaInstall> {
  return queued(env, async () => {
    const manifest = await readManifest(env);
    if (!manifest.installs.some((i) => i.path === install.path)) manifest.installs.push(install);
    manifest.latest = install.path;
    if (key && manifest.failed) delete manifest.failed[key];
    await writeManifest(env, manifest);
    return install;
  });
}

/** Removes a downloaded build that cannot run here and, with a reason, remembers the failure. */
async function discardInstall(env: NodeJS.ProcessEnv, install: LlamaInstall, key: string, reason: string | null): Promise<void> {
  await queued(env, async () => {
    const manifest = await readManifest(env);
    if (reason) manifest.failed = { ...manifest.failed, [key]: { reason: reason.slice(0, 300), at: Date.now() } };
    if (manifest.latest !== install.path && install.dir && install.source !== 'env' && install.source !== 'path') {
      manifest.installs = manifest.installs.filter((i) => i.path !== install.path);
      await writeManifest(env, manifest);
      await rm(install.dir, { recursive: true, force: true }).catch(() => undefined);
    } else {
      await writeManifest(env, manifest);
    }
  });
}

async function installLlamaServerUnlocked(options: InstallOptions): Promise<LlamaInstall> {
  const env = options.env ?? process.env;
  const platform = options.platform ?? process.platform;
  const arch = options.arch ?? process.arch;
  const vendor = options.gpu?.primary?.vendor;
  const candidate = options.candidate ?? asCandidate(variantFor(platform, arch, vendor));
  const variant = candidate.variant;
  const activate = options.activate !== false;
  const policy = storagePolicy(env);
  const release = await fetchRelease(options, platform, arch, candidate);
  const picked = pickReleaseAsset(release, platform, arch, candidate);
  if (!picked.main) {
    throw new Error(`No llama.cpp release asset found for ${platform}/${arch} (${candidate.label}) in ${release.tag_name}`);
  }
  const version = (options.version || release.tag_name || 'latest').replace(/[^\w.-]+/g, '-');
  const installDir = join(binDir(env), `llama-${version}-${candidateKey(candidate)}`);
  const existingBinary = await locateServerBinary(installDir).catch(() => null);
  if (existingBinary) {
    const manifest = await readManifest(env);
    const known = manifest.installs.find((i) => i.path === existingBinary);
    if (known?.binarySha256 && (!options.requireChecksum || known.verified) && await sha256File(existingBinary) === known.binarySha256) {
      if (activate) {
        manifest.latest = existingBinary;
        await writeManifest(env, manifest);
      }
      return { ...known, source: 'manifest' };
    }
  }

  const id = randomUUID();
  const work = join(cacheDir(env), `install-${id}`);
  const staging = `${installDir}.${id}.tmp`;
  const backup = `${installDir}.${id}.bak`;
  let backedUp = false;
  let promoted = false;
  const archiveDir = join(work, 'archives');
  const extractDir = join(work, 'extract');
  await mkdir(binDir(env), { recursive: true });
  if (await directoryBytes(binDir(env)) >= MAX_ENGINE_CACHE_BYTES) throw new Error('Engine cache reached 10 GiB; remove unused engine versions while the agent is stopped');
  // Archives, extracted libraries and staging coexist during installation.
  await requireDiskSpace(binDir(env), 8 * 1024 ** 3, policy.reserveBytes);
  await mkdir(archiveDir, { recursive: true });
  await mkdir(extractDir, { recursive: true });
  try {
    const assets = [picked.main, ...picked.companions];
    if (assets.length > 4) throw new Error('Too many companion engine archives');
    let mainSha: string | null = null;
    let anyVerified = true;
    const companionDirs: string[] = [];
    for (const [index, asset] of assets.entries()) {
      const archive = join(archiveDir, asset.name);
      await downloadFile(asset.browser_download_url, archive, {
        ...options,
        label: `Downloading ${asset.name}`,
      });
      const actual = await sha256File(archive);
      const check = await checksumFor(asset, release, options, index === 0);
      if (check.expected && check.expected !== actual) {
        throw new Error(`Checksum mismatch for ${asset.name}. Expected ${check.expected}, got ${actual}.`);
      }
      if (!check.published) {
        if (options.requireChecksum) {
          throw new Error(`No published SHA-256 for ${asset.name}; refusing because --require-checksum is set.`);
        }
        anyVerified = false;
        options.logger?.warn(`No published SHA-256 for ${asset.name}; recorded the local hash.`);
      }
      if (index === 0) mainSha = actual;
      await requireDiskSpace(extractDir, 4 * 1024 ** 3, policy.reserveBytes);
      const target = index === 0 ? extractDir : join(work, `companion-${index}`);
      companionDirs.push(target);
      await extractArchive(archive, target);
      let extracted = 0;
      for (const dir of companionDirs) extracted += await directoryBytes(dir);
      if (extracted > 4 * 1024 ** 3) throw new Error('Combined engine extraction exceeds 4 GiB');
    }
    const binary = await locateServerBinary(extractDir);
    if (!binary) throw new Error('No llama-server binary found inside the release archive');
    // llama.cpp loads its GPU backend from the directory of llama-server, and that backend finds the
    // CUDA runtime through $ORIGIN. The Linux runtime archive has its own folder, so put its files there.
    for (const dir of companionDirs.slice(1)) await copyFilesFlat(dir, dirname(binary));
    const extractedBytes = await directoryBytes(extractDir);
    if (await directoryBytes(binDir(env)) + extractedBytes > MAX_ENGINE_CACHE_BYTES) throw new Error('Engine cache would exceed 10 GiB; remove unused versions while the agent is stopped');
    await requireDiskSpace(binDir(env), extractedBytes, policy.reserveBytes);
    await rm(staging, { recursive: true, force: true });
    await mkdir(staging, { recursive: true });
    const entries = await readdir(extractDir, { withFileTypes: true });
    for (const entry of entries) {
      const source = join(extractDir, entry.name);
      const target = join(staging, entry.name);
      if (entry.isDirectory()) {
        await copyDir(source, target);
      } else {
        await copyFile(source, target);
        if (/^(llama-server|server)(\.exe)?$/i.test(entry.name)) await chmod(target, 0o755).catch(() => undefined);
      }
    }
    // Validate and hash the copied candidate before disturbing the previous install.
    const stagedBinary = await locateServerBinary(staging);
    if (!stagedBinary) throw new Error('llama-server was extracted but could not be found in staging');
    await chmod(stagedBinary, 0o755).catch(() => undefined);
    const binarySha256 = await sha256File(stagedBinary);
    const installedBinary = join(installDir, relative(staging, stagedBinary));
    if (await pathExists(installDir)) {
      await rename(installDir, backup);
      backedUp = true;
    }
    await rename(staging, installDir);
    promoted = true;
    const install: LlamaInstall = {
      path: installedBinary,
      version,
      variant,
      build: candidate.source,
      ...(candidate.cuda ? { cuda: candidate.cuda } : {}),
      sha256: mainSha,
      binarySha256,
      verified: anyVerified,
      source: 'download',
      dir: installDir,
    };
    const manifest = await readManifest(env);
    manifest.installs = manifest.installs.filter((i) => i.path !== installedBinary);
    manifest.installs.push(install);
    if (activate) manifest.latest = installedBinary;
    await writeManifest(env, manifest);
    // A failed cleanup must not roll back an already committed manifest.
    if (backedUp) await rm(backup, { recursive: true, force: true }).catch(() => undefined);
    return install;
  } catch (error) {
    // Keep the backup if rollback itself fails, so it remains recoverable.
    if (promoted) await rm(installDir, { recursive: true, force: true });
    if (backedUp) await rename(backup, installDir);
    throw error;
  } finally {
    await rm(work, { recursive: true, force: true }).catch(() => undefined);
    await rm(staging, { recursive: true, force: true }).catch(() => undefined);
  }
}

/** Copies every file below `source` into `target` itself, without the folders. */
async function copyFilesFlat(source: string, target: string): Promise<void> {
  for (const entry of await readdir(source, { withFileTypes: true })) {
    const from = join(source, entry.name);
    if (entry.isDirectory()) {
      await copyFilesFlat(from, target);
    } else if (entry.isFile()) {
      await copyFile(from, join(target, entry.name));
      await chmod(join(target, entry.name), (await stat(from)).mode & 0o777).catch(() => undefined);
    }
  }
}

async function copyDir(source: string, target: string): Promise<void> {
  await mkdir(target, { recursive: true });
  for (const entry of await readdir(source, { withFileTypes: true })) {
    const from = join(source, entry.name);
    const to = join(target, entry.name);
    if (entry.isDirectory()) await copyDir(from, to);
    else await copyFile(from, to);
  }
}

export async function binaryStatus(options: InstallOptions = {}): Promise<{ install: LlamaInstall | null; manifestPath: string }> {
  return { install: await findExistingLlama(options), manifestPath: manifestPath(options.env ?? process.env) };
}

export async function verifyInstalledBinary(install: LlamaInstall): Promise<boolean> {
  if (!install.binarySha256) return false;
  const actual = await sha256File(install.path);
  return actual === install.binarySha256;
}

export function isHttpError(error: unknown): error is HttpError {
  return Boolean(error && typeof error === 'object' && 'status' in error);
}

export async function readReleaseForTest(): Promise<ReleaseInfo> {
  const response = await fetchJson<ReleaseInfo>(RELEASE_URL, { headers: { accept: 'application/vnd.github+json' } });
  return response;
}

void rename;
void stat;
void ensureDirUnused;
function ensureDirUnused() { /* keeps import shape stable for older Node typings */ }
