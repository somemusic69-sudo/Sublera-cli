import { readdir } from 'node:fs/promises';
import { delimiter, join } from 'node:path';
import { execFile as realExecFile } from '../util/exec.js';
import { pathExists } from '../util/fs.js';
import type { GpuSnapshot, GpuVendor } from '../hardware/types.js';

export type LlamaVariant = 'cuda' | 'vulkan' | 'rocm' | 'metal' | 'cpu';

/** What the engine installer needs to know about this computer to pick a llama.cpp build. */
export type EngineHost = {
  platform: NodeJS.Platform;
  arch: string;
  vendor: GpuVendor | null;
  /** glibc version on Linux, such as "2.31"; null on musl (Alpine) or when unknown. */
  glibc: string | null;
  /** NVIDIA driver major version, such as 535; null when unknown. */
  nvidiaDriver: number | null;
  /** Lowest compute capability of the NVIDIA GPUs, such as 8.6; null when unknown. */
  computeCapability: number | null;
  /** Linux: a Vulkan loader and a Vulkan driver (ICD) were found. null when not checked. */
  vulkan: { loader: boolean; driver: boolean } | null;
  /** Linux: the ROCm kernel driver (/dev/kfd) and HIP runtime were found. null when not checked. */
  rocm: { driver: boolean; runtime: boolean } | null;
};

/** One llama.cpp build to try. `portable` builds come from this project's engine releases. */
export type EngineCandidate = {
  variant: LlamaVariant;
  source: 'official' | 'portable';
  /** CUDA major version of a CUDA build. */
  cuda?: number;
  label: string;
};

export type EnginePlan = {
  candidates: EngineCandidate[];
  skipped: Array<{ label: string; reason: string }>;
};

/**
 * Official llama.cpp Linux builds need at least this glibc. Measured on b12503: the CPU and Vulkan
 * builds are made on Ubuntu 22.04 (GLIBC_2.34), the CUDA and ROCm builds on Ubuntu 24.04 (GLIBC_2.38).
 */
export const OFFICIAL_LINUX_GLIBC: Record<LlamaVariant, string> = { cuda: '2.38', rocm: '2.38', vulkan: '2.34', cpu: '2.34', metal: '0' };
/** The official ROCm build does not carry the ROCm runtime; it links this HIP library from the host. */
const ROCM_HIP_LIBRARY = 'libamdhip64.so.7';
/** The portable builds from .github/workflows/engine.yml are made on Rocky Linux 8. */
export const PORTABLE_LINUX_GLIBC = '2.28';
/**
 * What each CUDA build can run on. `sass` is native GPU code: it runs on GPUs of the same major
 * compute capability and the same or a higher minor one, except "a" code, which runs only on that
 * exact GPU. `ptx` is compiled by the driver for the same or a newer GPU, which needs `ptxDriver`.
 * `driver` is the oldest NVIDIA driver that can load the build at all.
 */
type CudaBuild = { cuda: number; driver: number; sass: string[]; ptx: string[]; ptxDriver: number };
const CUDA_BUILDS: Record<string, CudaBuild> = {
  // llama.cpp b12503, CUDA 12.8 on Linux: its compressed GPU code needs a CUDA 12.4 (R550) driver.
  'official-linux-12': { cuda: 12, driver: 550, sass: ['8.6', '8.9', '12.0a'], ptx: ['5.0', '6.1', '7.0', '7.5', '8.0', '9.0'], ptxDriver: 570 },
  // llama.cpp b12503, CUDA 12.4 on Windows.
  'official-win32-12': { cuda: 12, driver: 528, sass: ['8.6', '8.9'], ptx: ['5.0', '6.1', '7.0', '7.5', '8.0', '9.0'], ptxDriver: 551 },
  // llama.cpp b12503, CUDA 13.5. CUDA 13 removed Maxwell, Pascal and Volta.
  'official-13': { cuda: 13, driver: 580, sass: ['8.6', '8.9', '10.3', '12.0a', '12.1a'], ptx: ['7.5', '8.0', '9.0'], ptxDriver: 620 },
  // .github/workflows/engine.yml, CUDA 12.4: native code from Maxwell to Hopper, PTX for newer GPUs.
  'portable-linux-12': { cuda: 12, driver: 525, sass: ['5.0', '6.0', '6.1', '7.0', '7.5', '8.0', '8.6', '8.9', '9.0'], ptx: ['9.0'], ptxDriver: 550 },
};

const VARIANTS: readonly LlamaVariant[] = ['cuda', 'vulkan', 'rocm', 'metal', 'cpu'];

export function parseVariant(value: string | undefined | null): LlamaVariant | null {
  const lower = String(value ?? '').trim().toLowerCase();
  return (VARIANTS as readonly string[]).includes(lower) ? lower as LlamaVariant : null;
}

export function compareDotted(a: string, b: string): number {
  const pa = a.split('.').map((n) => Number.parseInt(n, 10) || 0);
  const pb = b.split('.').map((n) => Number.parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const diff = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (diff) return Math.sign(diff);
  }
  return 0;
}

function glibcProblem(host: EngineHost, needs: string): string | null {
  if (host.platform !== 'linux') return null;
  if (!host.glibc) return 'this Linux has no glibc (for example Alpine/musl) or its version is unknown';
  return compareDotted(host.glibc, needs) < 0 ? `needs glibc ${needs} or newer; this system has glibc ${host.glibc}` : null;
}

function cudaBuild(source: EngineCandidate['source'], platform: NodeJS.Platform, cuda: number): CudaBuild | undefined {
  if (cuda >= 13) return source === 'official' ? CUDA_BUILDS['official-13'] : undefined;
  return CUDA_BUILDS[`${source}-${platform === 'win32' ? 'win32' : 'linux'}-${cuda}`];
}

function nativeCode(sass: string, cc: number): boolean {
  const exact = sass.endsWith('a');
  const value = Number.parseFloat(sass);
  return exact ? Math.abs(value - cc) < 0.01 : Math.floor(value) === Math.floor(cc) && value <= cc + 0.001;
}

/** Why a CUDA build cannot run on this driver and GPU, or null when it can (or when it is not known). */
export function cudaProblem(host: Pick<EngineHost, 'platform' | 'nvidiaDriver' | 'computeCapability'>, source: EngineCandidate['source'], cuda: number): string | null {
  const build = cudaBuild(source, host.platform, cuda);
  if (!build) return null;
  const driver = host.nvidiaDriver;
  if (driver != null && driver < build.driver) return `needs NVIDIA driver ${build.driver} or newer; this system has driver ${driver}`;
  const cc = host.computeCapability;
  if (cc == null || build.sass.some((sass) => nativeCode(sass, cc))) return null;
  const gpu = `compute capability ${cc.toFixed(1)}`;
  if (!build.ptx.some((ptx) => Number.parseFloat(ptx) <= cc + 0.001)) return `has no GPU code for ${gpu}`;
  if (driver != null && driver < build.ptxDriver) return `needs NVIDIA driver ${build.ptxDriver} or newer for ${gpu} (no native GPU code for it); this system has driver ${driver}`;
  return null;
}

/**
 * The builds to try, best first, and the builds left out with the reason. Each candidate is
 * installed and started in turn until one works, so a missing library or driver on one build
 * does not stop the engine.
 */
export function engineCandidates(host: EngineHost, forced?: LlamaVariant | null): EnginePlan {
  const plan: EnginePlan = { candidates: [], skipped: [] };
  const add = (candidate: EngineCandidate, reason: string | null) => {
    if (reason) plan.skipped.push({ label: candidate.label, reason });
    else if (forced && candidate.variant !== forced) plan.skipped.push({ label: candidate.label, reason: `SUBLERA_LLAMA_VARIANT=${forced}` });
    else plan.candidates.push(candidate);
  };
  const official = (variant: LlamaVariant, label: string, cuda?: number): EngineCandidate => ({ variant, source: 'official', label, ...(cuda ? { cuda } : {}) });
  const portable = (variant: LlamaVariant, label: string, cuda?: number): EngineCandidate => ({ variant, source: 'portable', label, ...(cuda ? { cuda } : {}) });
  const { platform, arch, vendor } = host;

  if (platform === 'darwin') {
    add(arch === 'arm64' ? official('metal', 'official Metal build') : official('cpu', 'official CPU build'), null);
    return plan;
  }
  if ((platform !== 'linux' && platform !== 'win32') || (arch !== 'x64' && arch !== 'arm64')) {
    plan.skipped.push({ label: 'all llama.cpp builds', reason: `no build is published for ${platform}/${arch}; set LLAMA_SERVER to your own llama-server` });
    return plan;
  }
  const linux = platform === 'linux';
  const officialProblem = (variant: LlamaVariant) => glibcProblem(host, OFFICIAL_LINUX_GLIBC[variant]);
  const portableProblem = !linux || arch !== 'x64' ? null : glibcProblem(host, PORTABLE_LINUX_GLIBC);
  const withPortable = linux && arch === 'x64';

  if (vendor === 'nvidia') {
    // CUDA 12 first: it supports the most GPUs and drivers. CUDA 13 builds are the only arm64 CUDA builds.
    if (arch === 'x64') add(official('cuda', 'official CUDA 12 build', 12), officialProblem('cuda') ?? cudaProblem(host, 'official', 12));
    add(official('cuda', 'official CUDA 13 build', 13), officialProblem('cuda') ?? cudaProblem(host, 'official', 13));
    if (withPortable) add(portable('cuda', 'portable CUDA 12 build', 12), portableProblem ?? cudaProblem(host, 'portable', 12));
  }
  if (vendor === 'nvidia' || vendor === 'amd' || vendor === 'intel') {
    const vulkanProblem = linux && host.vulkan && !(host.vulkan.loader && host.vulkan.driver)
      ? `no Vulkan ${host.vulkan.loader ? 'driver (ICD)' : 'loader (libvulkan.so.1)'} found${vendor === 'nvidia' ? '; in a container, add "graphics" to NVIDIA_DRIVER_CAPABILITIES' : ''}`
      : null;
    add(official('vulkan', 'official Vulkan build'), officialProblem('vulkan') ?? vulkanProblem);
  }
  if (vendor === 'amd' && linux && arch === 'x64') {
    const rocmProblem = host.rocm && !host.rocm.driver ? 'no ROCm kernel driver (/dev/kfd) found'
      : host.rocm && !host.rocm.runtime ? `no ROCm runtime (${ROCM_HIP_LIBRARY}) found` : null;
    add(official('rocm', 'official ROCm build'), officialProblem('rocm') ?? rocmProblem);
  }
  add(official('cpu', 'official CPU build'), officialProblem('cpu'));
  if (withPortable) add(portable('cpu', 'portable CPU build'), portableProblem);
  return plan;
}

export type DetectOptions = {
  platform?: NodeJS.Platform;
  arch?: string;
  gpu?: GpuSnapshot | null;
  env?: NodeJS.ProcessEnv;
  exec?: typeof realExecFile;
  glibc?: string | null;
};

export async function detectEngineHost(options: DetectOptions = {}): Promise<EngineHost> {
  const platform = options.platform ?? process.platform;
  const env = options.env ?? process.env;
  const exec = options.exec ?? realExecFile;
  const vendor = options.gpu?.primary?.vendor ?? null;
  const host: EngineHost = {
    platform,
    arch: options.arch ?? process.arch,
    vendor,
    glibc: platform === 'linux' ? (options.glibc !== undefined ? options.glibc : await glibcVersion(exec)) : null,
    nvidiaDriver: null,
    computeCapability: null,
    vulkan: null,
    rocm: null,
  };
  if (vendor === 'nvidia') Object.assign(host, await nvidiaInfo(exec));
  if (platform === 'linux') {
    const libraries = new LibraryFinder(env, exec);
    host.vulkan = { loader: await libraries.has('libvulkan.so.1'), driver: await vulkanDriverOnLinux(env) };
    if (vendor === 'amd') host.rocm = { driver: await pathExists('/dev/kfd'), runtime: await libraries.has(ROCM_HIP_LIBRARY, ['/opt/rocm/lib', '/opt/rocm/lib64']) };
  }
  return host;
}

/** The running glibc; null on musl. */
export async function glibcVersion(exec: typeof realExecFile = realExecFile): Promise<string | null> {
  const result = await exec('getconf', ['GNU_LIBC_VERSION'], { timeoutMs: 5_000 });
  const match = result.code === 0 ? result.stdout.match(/glibc\s+(\d+\.\d+)/) : null;
  if (match) return match[1]!;
  try {
    // Node's report also has it. Network details are skipped where Node supports that (they can be slow).
    const report = process.report as unknown as { excludeNetwork?: boolean; getReport(): { header?: { glibcVersionRuntime?: string } } };
    const previous = report.excludeNetwork;
    report.excludeNetwork = true;
    try {
      return report.getReport().header?.glibcVersionRuntime || null;
    } finally {
      report.excludeNetwork = previous;
    }
  } catch {
    return null;
  }
}

export function parseNvidiaDriverInfo(text: string): { nvidiaDriver: number | null; computeCapability: number | null } {
  let driver: number | null = null;
  let compute: number | null = null;
  for (const line of text.split(/\r?\n/)) {
    const [driverText, computeText] = line.split(',').map((c) => c.trim());
    const major = Number.parseInt(driverText || '', 10);
    if (Number.isFinite(major) && major > 0) driver = driver == null ? major : Math.min(driver, major);
    const cc = Number(computeText);
    if (computeText && Number.isFinite(cc) && cc > 0) compute = compute == null ? cc : Math.min(compute, cc);
  }
  return { nvidiaDriver: driver, computeCapability: compute };
}

async function nvidiaInfo(exec: typeof realExecFile): Promise<{ nvidiaDriver: number | null; computeCapability: number | null }> {
  // compute_cap is missing from old drivers, and an unknown field fails the whole query.
  for (const fields of ['driver_version,compute_cap', 'driver_version']) {
    const result = await exec('nvidia-smi', [`--query-gpu=${fields}`, '--format=csv,noheader'], { timeoutMs: 10_000 });
    if (result.code === 0 && result.stdout.trim()) {
      const info = parseNvidiaDriverInfo(result.stdout);
      if (info.nvidiaDriver != null) return info;
    }
  }
  return { nvidiaDriver: null, computeCapability: null };
}

const LIB_DIRS = [
  '/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu', '/usr/lib/aarch64-linux-gnu', '/lib/aarch64-linux-gnu',
  '/usr/lib64', '/lib64', '/usr/lib', '/lib', '/usr/local/lib', '/usr/local/lib64',
];
const ICD_DIRS = ['/etc/xdg/vulkan/icd.d', '/usr/local/etc/vulkan/icd.d', '/etc/vulkan/icd.d', '/usr/local/share/vulkan/icd.d', '/usr/share/vulkan/icd.d'];

/** Finds a shared library the way the loader would, without loading it. */
class LibraryFinder {
  private cache: string | null = null;

  constructor(private readonly env: NodeJS.ProcessEnv, private readonly exec: typeof realExecFile) {}

  async has(name: string, extraDirs: string[] = []): Promise<boolean> {
    const dirs = [...(this.env.LD_LIBRARY_PATH || '').split(delimiter).filter(Boolean), ...extraDirs, ...LIB_DIRS];
    for (const dir of dirs) {
      if (await pathExists(join(dir, name))) return true;
    }
    return new RegExp(`(^|\\s)${name.replace(/[.+]/g, '\\$&')}\\s`, 'm').test(await this.ldconfig());
  }

  private async ldconfig(): Promise<string> {
    if (this.cache != null) return this.cache;
    this.cache = '';
    for (const ldconfig of ['ldconfig', '/sbin/ldconfig', '/usr/sbin/ldconfig']) {
      const result = await this.exec(ldconfig, ['-p'], { timeoutMs: 5_000 });
      if (result.code === 0) { this.cache = result.stdout; break; }
    }
    return this.cache;
  }
}

async function vulkanDriverOnLinux(env: NodeJS.ProcessEnv): Promise<boolean> {
  if (env.VK_ICD_FILENAMES || env.VK_DRIVER_FILES || env.VK_ADD_DRIVER_FILES) return true;
  const xdg = (name: string) => (env[name] || '').split(delimiter).filter(Boolean).map((dir) => join(dir, 'vulkan', 'icd.d'));
  const home = env.HOME ? [join(env.HOME, '.config', 'vulkan', 'icd.d'), join(env.HOME, '.local', 'share', 'vulkan', 'icd.d')] : [];
  for (const dir of [...home, ...ICD_DIRS, ...xdg('XDG_CONFIG_DIRS'), ...xdg('XDG_DATA_DIRS')]) {
    const files = await readdir(dir).catch(() => [] as string[]);
    if (files.some((name) => name.endsWith('.json'))) return true;
  }
  return false;
}

/** Plain-language help for the loader errors that stop a llama.cpp build. */
export function explainEngineFailure(detail: string, host?: Pick<EngineHost, 'glibc'> | null): string | null {
  const hints: string[] = [];
  const versions = [...detail.matchAll(/(GLIBC|GLIBCXX|CXXABI)_([\d.]+)'? not found/g)];
  if (versions.length) {
    const highest = new Map<string, string>();
    for (const [, name, version] of versions) {
      const seen = highest.get(name!);
      if (!seen || compareDotted(version!, seen) > 0) highest.set(name!, version!);
    }
    const needs = [...highest].map(([name, version]) => `${name}_${version}`).join(', ');
    hints.push(`This system's C/C++ libraries are older than this build needs (${needs}${host?.glibc ? `; this system has glibc ${host.glibc}` : ''}).`);
  }
  const libraries = new Set([...detail.matchAll(/(lib[\w.+-]+?\.so(?:\.\d+)*): cannot open shared object file/g)].map((m) => m[1]!));
  for (const lib of libraries) {
    if (/^libcuda\.so/.test(lib)) {
      hints.push('The NVIDIA driver library libcuda.so.1 is not visible. In a container, start it with GPU access (for example docker run --gpus all) and NVIDIA_DRIVER_CAPABILITIES with "compute,utility".');
    } else if (/^libvulkan\.so/.test(lib)) {
      hints.push('The Vulkan loader (libvulkan.so.1) is not installed. Vulkan also needs a GPU driver with Vulkan support; in an NVIDIA container add "graphics" to NVIDIA_DRIVER_CAPABILITIES.');
    } else if (/^libgomp\.so/.test(lib)) {
      hints.push('Install the OpenMP runtime (package libgomp1 or libgomp).');
    } else if (/^(libamdhip64|libhipblas|librocblas|libhsa-runtime64|librocm)/.test(lib)) {
      hints.push(`The ROCm runtime (${lib}) is not installed.`);
    } else {
      hints.push(`The system library ${lib} is missing.`);
    }
  }
  if (/exec format error|cannot execute binary file/i.test(detail)) hints.push('This build is for a different processor type.');
  if (/driver version is insufficient|unsupported display driver \/ cuda driver combination/i.test(detail)) hints.push('The NVIDIA driver is too old for this CUDA build; update the driver.');
  if (/no CUDA-capable device is detected/i.test(detail)) hints.push('No NVIDIA GPU is visible to this process (check CUDA_VISIBLE_DEVICES and the container GPU access).');
  if (/forward compatibility was attempted on non supported HW/i.test(detail)) hints.push('A CUDA forward-compatibility libcuda (for example from /usr/local/cuda/compat) is used on a GPU that does not support it; remove it from LD_LIBRARY_PATH.');
  return hints.length ? hints.join(' ') : null;
}

/** GPU lines of `llama-server --list-devices`, such as "CUDA0: NVIDIA RTX A6000 (48669 MiB, 48000 MiB free)". */
export function parseDeviceList(text: string): string[] {
  const start = text.indexOf('Available devices:');
  if (start < 0) return [];
  return text.slice(start).split(/\r?\n/).slice(1)
    .map((line) => line.trim())
    .filter((line) => /^[\w.-]+: .+\(\d+ MiB, \d+ MiB free\)$/.test(line));
}
