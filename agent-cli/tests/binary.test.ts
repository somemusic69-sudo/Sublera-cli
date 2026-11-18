import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { acquireInstanceLock } from '../src/util/instanceLock.js';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import AdmZip from 'adm-zip';
import { installLlamaServer, pickReleaseAsset, variantFor, findExistingLlama, verifyInstalledBinary, ensureLlamaServer } from '../src/engine/binaryManager.js';

test('concurrent installs serialize per home, reuse validated installs, and retain all manifest entries', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-concurrent-bin-'));
  let unlock: (() => Promise<void>) | undefined;
  t.after(async () => { await unlock?.(); await rm(home, { recursive: true, force: true }); });
  const bytes = zipWith('bin/llama-server', 'concurrent engine');
  const { server, base, requests } = await serveFiles({ '/main.zip': bytes });
  t.after(() => { server.close(); });
  const env = { ...process.env, SUBLET_HOME: home };
  const options = {
    env, platform: 'linux' as const, arch: 'x64', showProgress: false,
    release: { tag_name: 'b-concurrent', assets: [{ name: 'llama-bin-ubuntu-x64.zip', browser_download_url: `${base}/main.zip`, digest: `sha256:${sha256(bytes)}` }] },
  };
  // Installation must also work for the agent, which already owns this lock.
  unlock = await acquireInstanceLock(env);
  const installs = await Promise.all([
    installLlamaServer(options), installLlamaServer(options),
    installLlamaServer({ ...options, version: 'other' }),
  ]);
  assert.equal(requests(), 2);
  assert.equal(installs[0].path, installs[1].path);
  assert.equal(installs[1].source, 'manifest');
  for (const install of installs) assert.equal(await verifyInstalledBinary(install), true);
  const manifest = JSON.parse(await readFile(join(home, 'bin', 'manifest.json'), 'utf8'));
  assert.equal(manifest.installs.length, 2);
  assert.equal(manifest.latest, installs[2].path);
  assert.deepEqual((await readdir(join(home, 'bin'))).filter((name) => /\.(tmp|bak)$/.test(name)), []);
  assert.deepEqual((await readdir(join(home, 'cache'))).filter((name) => name.startsWith('install-')), []);
});

test('failed replacement preserves the previous install and does not poison the install queue', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-replace-bin-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const bytes = zipWith('llama-server', 'previous engine');
  const invalid = zipWith('README', 'not an engine');
  const { server, base } = await serveFiles({ '/main.zip': bytes, '/invalid.zip': invalid });
  t.after(() => { server.close(); });
  const env = { ...process.env, SUBLET_HOME: home };
  const options = {
    env, platform: 'linux' as const, arch: 'x64', showProgress: false,
    release: { tag_name: 'b-replace', assets: [{ name: 'llama-bin-ubuntu-x64.zip', browser_download_url: `${base}/main.zip` }] },
  };
  const first = await installLlamaServer(options);
  const manifestPath = join(home, 'bin', 'manifest.json');
  const before = await readFile(manifestPath, 'utf8');
  await assert.rejects(installLlamaServer({ ...options, requireChecksum: true, release: {
    tag_name: 'b-replace', assets: [{ name: 'llama-bin-ubuntu-x64.zip', browser_download_url: `${base}/invalid.zip`, digest: `sha256:${sha256(invalid)}` }],
  } }), /No llama-server binary/);
  assert.equal(await readFile(first.path, 'utf8'), 'previous engine');
  assert.equal(await readFile(manifestPath, 'utf8'), before);
  // Fail manifest commit after promotion to exercise rollback of the old directory.
  await rm(manifestPath);
  await mkdir(manifestPath);
  await assert.rejects(installLlamaServer(options));
  assert.equal(await readFile(first.path, 'utf8'), 'previous engine');
  await rm(manifestPath, { recursive: true });
  assert.equal(await verifyInstalledBinary(await installLlamaServer(options)), true);
});

test('doctor --fix respects an existing instance lock before any engine mutations', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-doctor-lock-'));
  const release = await acquireInstanceLock({ SUBLET_HOME: home });
  t.after(async () => { await release(); await rm(home, { recursive: true, force: true }); });
  const commands = new URL('../src/commands.ts', import.meta.url).href;
  const logger = new URL('../src/logger.ts', import.meta.url).href;
  const code = `
    import { cmdDoctor } from ${JSON.stringify(commands)};
    import { Logger } from ${JSON.stringify(logger)};
    globalThis.fetch = async () => new Response('{}', { status: 200 });
    process.exitCode = await cmdDoctor({ fix: true, json: true, requireChecksum: false }, new Logger({ level: 'silent' }));
  `;
  const result = await promisify(execFile)(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', code], {
    cwd: fileURLToPath(new URL('../', import.meta.url)),
    env: { ...process.env, SUBLET_HOME: home, PATH: '', LLAMA_SERVER: '', SUBLET_KEY: '' },
    timeout: 30_000,
  }).then(() => { throw new Error('doctor should report failure'); }, (error) => error);
  assert.equal(result.code, 1);
  const report = JSON.parse(result.stdout);
  assert.match(report.checks.find((item: { name: string }) => item.name === 'llama-server').detail, /SUBLET_HOME is locked/);
  assert.deepEqual(await readdir(home), ['agent.lock']);
});

test('doctor --fix while the agent runs reports the installed engine without changing it', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-doctor-busy-'));
  const release = await acquireInstanceLock({ SUBLET_HOME: home });
  t.after(async () => { await release(); await rm(home, { recursive: true, force: true }); });
  const commands = new URL('../src/commands.ts', import.meta.url).href;
  const logger = new URL('../src/logger.ts', import.meta.url).href;
  const code = `
    import { cmdDoctor } from ${JSON.stringify(commands)};
    import { Logger } from ${JSON.stringify(logger)};
    globalThis.fetch = async () => new Response('{}', { status: 200 });
    process.exitCode = await cmdDoctor({ fix: true, json: true, requireChecksum: false }, new Logger({ level: 'silent' }));
  `;
  const result = await promisify(execFile)(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', code], {
    cwd: fileURLToPath(new URL('../', import.meta.url)),
    env: { ...process.env, SUBLET_HOME: home, LLAMA_SERVER: process.execPath, SUBLET_KEY: '' },
    timeout: 30_000,
  }).catch((error) => error);
  const report = JSON.parse(result.stdout);
  const engine = report.checks.find((item: { name: string }) => item.name === 'llama-server');
  assert.equal(engine.status, 'warn');
  assert.match(engine.detail, /not changed: .*SUBLET_HOME is locked/);
  assert.equal(report.llama.path, process.execPath);
  assert.deepEqual(await readdir(home), ['agent.lock']);
});

test('strict checksum mode refuses an existing custom engine without downloading', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'sublet-strict-engine-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const env = { ...process.env, SUBLET_HOME: dir, LLAMA_SERVER: process.execPath };
  assert.equal((await ensureLlamaServer({ env })).found, true);
  const strict = await ensureLlamaServer({ env, fix: true, requireChecksum: true, fetchImpl: async () => { throw new Error('must not download'); } });
  assert.equal(strict.found, false);
  if (!strict.found) assert.match(strict.reason, /require-checksum/);
});

test('an engine that cannot execute is not reported ready', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'sublet-broken-engine-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const binary = join(dir, 'broken-engine');
  await writeFile(binary, 'not an executable');
  const result = await ensureLlamaServer({ env: { ...process.env, SUBLET_HOME: dir, LLAMA_SERVER: binary } });
  assert.equal(result.found, false);
  if (!result.found) assert.match(result.reason, /llama-server could not start/);
});

test('doctor fails instead of reporting an unusable engine as ready', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-doctor-broken-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const binary = join(home, 'broken-engine');
  await writeFile(binary, 'not an executable');
  const commands = new URL('../src/commands.ts', import.meta.url).href;
  const logger = new URL('../src/logger.ts', import.meta.url).href;
  const code = `
    import { cmdDoctor } from ${JSON.stringify(commands)};
    import { Logger } from ${JSON.stringify(logger)};
    globalThis.fetch = async () => new Response('{}', { status: 200 });
    process.exitCode = await cmdDoctor({ fix: false, json: true, requireChecksum: false }, new Logger({ level: 'silent' }));
  `;
  const result = await promisify(execFile)(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', code], {
    cwd: fileURLToPath(new URL('../', import.meta.url)),
    env: { ...process.env, SUBLET_HOME: home, LLAMA_SERVER: binary, SUBLET_KEY: '' },
    timeout: 30_000,
  }).then(() => { throw new Error('doctor should report failure'); }, (error) => error);
  assert.equal(result.code, 1);
  const report = JSON.parse(result.stdout);
  const engine = report.checks.find((item: { name: string }) => item.name === 'llama-server');
  assert.equal(engine.status, 'fail');
  assert.match(engine.detail, /llama-server could not start/);
});

function release(name: string, url: string, size: number) {
  return {
    tag_name: 'b9999',
    assets: [{ name, browser_download_url: url, size }],
  };
}

test('release assets follow the GPU and operating system', () => {
  const assets = [
    { name: 'cudart-llama-bin-win-cuda-12.4-x64.zip', browser_download_url: 'http://x/cudart' },
    { name: 'llama-b1-bin-win-cuda-x64.zip', browser_download_url: 'http://x/cuda' },
    { name: 'llama-b1-bin-win-vulkan-x64.zip', browser_download_url: 'http://x/vulkan' },
    { name: 'llama-b1-bin-win-cpu-x64.zip', browser_download_url: 'http://x/cpu' },
    { name: 'llama-b1-bin-ubuntu-x64.zip', browser_download_url: 'http://x/linux' },
    { name: 'llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz', browser_download_url: 'http://x/linux-cuda' },
    { name: 'llama-b1-bin-macos-arm64.tar.gz', browser_download_url: 'http://x/mac' },
  ];
  const rel = { tag_name: 'b1', assets };
  assert.equal(pickReleaseAsset(rel, 'win32', 'x64', 'cuda').main?.name, 'llama-b1-bin-win-cuda-x64.zip');
  assert.equal(pickReleaseAsset(rel, 'win32', 'x64', 'vulkan').main?.name, 'llama-b1-bin-win-vulkan-x64.zip');
  assert.equal(pickReleaseAsset(rel, 'linux', 'x64', 'cuda').main?.name, 'llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz');
  assert.equal(pickReleaseAsset(rel, 'linux', 'x64', 'cpu').main?.name, 'llama-b1-bin-ubuntu-x64.zip');
  assert.equal(pickReleaseAsset(rel, 'darwin', 'arm64', 'metal').main?.name, 'llama-b1-bin-macos-arm64.tar.gz');
  assert.equal(pickReleaseAsset(rel, 'win32', 'x64', 'cuda').companions[0]?.name, 'cudart-llama-bin-win-cuda-12.4-x64.zip');
  assert.equal(variantFor('win32', 'x64', 'nvidia'), 'cuda');
  assert.equal(variantFor('linux', 'x64', 'amd'), 'vulkan');
  assert.equal(variantFor('darwin', 'arm64', 'apple'), 'metal');
  assert.equal(variantFor('darwin', 'x64', 'apple'), 'cpu');
});

test('an archive is downloaded, checksum-verified, extracted, and recorded', async () => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-bin-'));
  const zip = new AdmZip();
  zip.addFile('bin/llama-server.exe', Buffer.from('fake llama binary'));
  const archive = join(home, 'fixture.zip');
  zip.writeZip(archive);
  const bytes = await readFile(archive);
  const sha = createHash('sha256').update(bytes).digest('hex');
  const server = createServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'application/zip', 'content-length': bytes.length });
    res.end(bytes);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  const env = { ...process.env, SUBLET_HOME: home, SUBLET_LLAMA_SHA256: sha };
  try {
    const install = await installLlamaServer({
      platform: 'win32',
      arch: 'x64',
      env,
      gpu: {
        primary: { vendor: 'nvidia' } as never,
        gpus: [],
        source: 'test',
        platform: 'win32',
        checkedAt: Date.now(),
        warnings: [],
      },
      release: release('llama-b9999-bin-win-cuda-x64.zip', `http://127.0.0.1:${port}/llama.zip`, bytes.length),
      showProgress: false,
    });
    assert.equal(install.variant, 'cuda');
    assert.equal(install.verified, true);
    assert.equal(install.sha256, sha);
    assert.ok(install.path.endsWith('llama-server.exe'));
    await stat(install.path);
    const manifest = JSON.parse(await readFile(join(home, 'bin', 'manifest.json'), 'utf8'));
    assert.equal(manifest.latest, install.path);
    assert.ok(install.binarySha256);
    const found = await findExistingLlama({ env });
    assert.equal(found?.path, install.path);
  } finally {
    server.close();
    await rm(home, { recursive: true, force: true });
  }
});

test('a bad binary checksum is rejected before extraction', async () => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-bin-'));
  const zip = new AdmZip();
  zip.addFile('llama-server.exe', Buffer.from('fake'));
  const archive = join(home, 'fixture.zip');
  zip.writeZip(archive);
  const bytes = await readFile(archive);
  const server = createServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'application/zip', 'content-length': bytes.length });
    res.end(bytes);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  const env = { ...process.env, SUBLET_HOME: home, SUBLET_LLAMA_SHA256: '0'.repeat(64) };
  try {
    await assert.rejects(installLlamaServer({
      platform: 'win32',
      arch: 'x64',
      env,
      gpu: { primary: { vendor: 'nvidia' } as never, gpus: [], source: 'test', platform: 'win32', checkedAt: Date.now(), warnings: [] },
      release: release('llama-b9999-bin-win-cuda-x64.zip', `http://127.0.0.1:${port}/llama.zip`, bytes.length),
      showProgress: false,
    }), /Checksum mismatch/);
    await assert.rejects(stat(join(home, 'bin', 'manifest.json')), /ENOENT/);
  } finally {
    server.close();
    await rm(home, { recursive: true, force: true });
  }
});

async function serveFiles(files: Record<string, Buffer>) {
  let requests = 0;
  const server = createServer((req, res) => {
    requests++;
    const body = files[req.url || ''];
    if (!body) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.writeHead(200, { 'content-type': 'application/zip', 'content-length': body.length });
    res.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  return { server, base: `http://127.0.0.1:${port}`, requests: () => requests };
}

function zipWith(name: string, content: string): Buffer {
  const zip = new AdmZip();
  zip.addFile(name, Buffer.from(content));
  return zip.toBuffer();
}

const sha256 = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

test('a modified or unhashed installed binary is reinstalled instead of reused', async () => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-bin-'));
  const bytes = zipWith('bin/llama-server', 'genuine llama binary');
  const { server, base, requests } = await serveFiles({ '/llama.zip': bytes });
  const env = { ...process.env, SUBLET_HOME: home };
  const options = {
    platform: 'linux' as const,
    arch: 'x64',
    env,
    release: { tag_name: 'b9999', assets: [{ name: 'llama-b9999-bin-ubuntu-x64.zip', browser_download_url: `${base}/llama.zip`, digest: `sha256:${sha256(bytes)}` }] },
    showProgress: false,
  };
  try {
    const first = await installLlamaServer(options);
    await writeFile(first.path, 'tampered');
    assert.notEqual((await findExistingLlama({ env }))?.path, first.path);
    const second = await installLlamaServer(options);
    assert.equal(requests(), 2);
    assert.equal(second.path, first.path);
    assert.equal(await readFile(second.path, 'utf8'), 'genuine llama binary');
    assert.equal(await verifyInstalledBinary(second), true);

    const manifestFile = join(home, 'bin', 'manifest.json');
    const manifest = JSON.parse(await readFile(manifestFile, 'utf8'));
    for (const install of manifest.installs) delete install.binarySha256;
    await writeFile(manifestFile, JSON.stringify(manifest));
    assert.notEqual((await findExistingLlama({ env }))?.path, first.path);
    await installLlamaServer(options);
    assert.equal(requests(), 3);
  } finally {
    server.close();
    await rm(home, { recursive: true, force: true });
  }
});

test('SUBLET_LLAMA_SHA256 pins only the main archive, not the CUDA runtime companion', async () => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-bin-'));
  const main = zipWith('llama-server.exe', 'cuda llama');
  const cudart = zipWith('cudart64_12.dll', 'cuda runtime');
  const { server, base } = await serveFiles({ '/main.zip': main, '/cudart.zip': cudart });
  const env = { ...process.env, SUBLET_HOME: home, SUBLET_LLAMA_SHA256: sha256(main) };
  try {
    const install = await installLlamaServer({
      platform: 'win32',
      arch: 'x64',
      env,
      gpu: { primary: { vendor: 'nvidia' } as never, gpus: [], source: 'test', platform: 'win32', checkedAt: Date.now(), warnings: [] },
      release: {
        tag_name: 'b9999',
        assets: [
          { name: 'llama-b9999-bin-win-cuda-12.4-x64.zip', browser_download_url: `${base}/main.zip` },
          { name: 'cudart-llama-bin-win-cuda-12.4-x64.zip', browser_download_url: `${base}/cudart.zip`, digest: `sha256:${sha256(cudart)}` },
        ],
      },
      showProgress: false,
    });
    assert.equal(install.verified, true);
    assert.equal(install.sha256, sha256(main));
    await stat(join(install.dir || '', 'cudart64_12.dll'));
  } finally {
    server.close();
    await rm(home, { recursive: true, force: true });
  }
});

test('the Linux CUDA runtime archive lands next to llama-server, where the CUDA backend looks for it', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-cudart-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const main = new AdmZip();
  main.addFile('llama-b1/llama-server', Buffer.from('cuda llama'));
  main.addFile('llama-b1/libggml-cuda.so', Buffer.from('backend'));
  const cudart = new AdmZip();
  cudart.addFile('cudart-llama-b1-bin-ubuntu-cuda-12.8-x64/libcudart.so.12', Buffer.from('runtime'));
  const { server, base } = await serveFiles({ '/main.zip': main.toBuffer(), '/cudart.zip': cudart.toBuffer() });
  t.after(() => { server.close(); });
  const install = await installLlamaServer({
    platform: 'linux', arch: 'x64', env: { ...process.env, SUBLET_HOME: home }, showProgress: false,
    candidate: { variant: 'cuda', source: 'official', cuda: 12, label: 'official CUDA 12 build' },
    release: { tag_name: 'b1', assets: [
      { name: 'llama-b1-bin-ubuntu-cuda-12.8-x64.zip', browser_download_url: `${base}/main.zip` },
      { name: 'cudart-llama-b1-bin-ubuntu-cuda-12.8-x64.zip', browser_download_url: `${base}/cudart.zip` },
    ] },
  });
  assert.equal(install.cuda, 12);
  assert.equal(install.build, 'official');
  assert.equal(await readFile(join(install.path, '..', 'libcudart.so.12'), 'utf8'), 'runtime');
});

/** A fake llama-server: `--version` exits with `versionExit`, `--list-devices` prints `devices`. */
function fakeEngine(devices: string[], versionExit = 0, stderr = ''): string {
  return [
    '#!/bin/sh',
    `if [ "$1" = "--version" ]; then printf '%s' '${stderr.replace(/'/g, "'\\''")}' >&2; exit ${versionExit}; fi`,
    'if [ "$1" = "--list-devices" ]; then echo "Available devices:";',
    ...(devices.length ? devices.map((line) => `  echo "  ${line}";`) : ['  echo "  (none)";']),
    'fi',
    'exit 0',
  ].join('\n');
}

const nvidiaGpu = {
  primary: { vendor: 'nvidia', name: 'NVIDIA RTX A6000' } as never,
  gpus: [], source: 'test', platform: 'linux' as const, checkedAt: 0, warnings: [],
};
const oldLinux = {
  platform: 'linux' as const, arch: 'x64', vendor: 'nvidia' as const, glibc: '2.31',
  nvidiaDriver: 535, computeCapability: 8.6, vulkan: { loader: false, driver: false }, rocm: null,
};
const onLinux = { platform: 'linux' as const, arch: 'x64', host: oldLinux, gpu: nvidiaGpu, showProgress: false };

test('a GPU build that cannot start or finds no GPU is skipped for the next build', { skip: process.platform === 'win32' }, async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-fallback-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const glibc = "llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by libllama.so.1)";
  const broken = zipWith('llama-server', fakeEngine([], 1, glibc));
  const blind = zipWith('llama-server', fakeEngine([]));
  const cpu = zipWith('llama-server', fakeEngine([]));
  const { server, base } = await serveFiles({ '/broken.zip': broken, '/blind.zip': blind, '/cpu.zip': cpu });
  t.after(() => { server.close(); });
  const env = { ...process.env, SUBLET_HOME: home, LLAMA_SERVER: '', SUBLERA_LLAMA_VARIANT: '', SUBLET_LLAMA_VARIANT: '' };
  const assets = (cuda: string) => ({ tag_name: 'b1', assets: [
    { name: 'sublera-llama-b1-linux-cuda-12.8-x64.zip', browser_download_url: `${base}/${cuda}.zip` },
    { name: 'sublera-llama-b1-linux-cpu-x64.zip', browser_download_url: `${base}/cpu.zip` },
  ] });

  const first = await ensureLlamaServer({ ...onLinux, env, fix: true, release: assets('broken') });
  assert.equal(first.found, true);
  if (!first.found) return;
  assert.equal(first.install.variant, 'cpu');
  assert.equal(first.install.build, 'portable');
  assert.match(first.warning!, /runs on the CPU, not on the NVIDIA RTX A6000/);
  assert.match(first.warning!, /portable CUDA 12 build: llama-server could not start: .*older than this build needs \(GLIBC_2\.34; this system has glibc 2\.31\)/);
  assert.match(first.warning!, /official CUDA 12 build: skipped, needs glibc 2\.38 or newer/);
  const manifest = JSON.parse(await readFile(join(home, 'bin', 'manifest.json'), 'utf8'));
  assert.equal(manifest.latest, first.install.path);
  // The failed build is removed and remembered, so a new release does not download it again.
  assert.match(manifest.failed['portable-cuda12'].reason, /could not start/);
  assert.equal(manifest.installs.length, 1);
  assert.deepEqual((await readdir(join(home, 'bin'))).filter((name) => name.includes('cuda')), []);
  const again = await ensureLlamaServer({ ...onLinux, env, fix: true, release: { ...assets('blind'), tag_name: 'b2' } });
  assert.equal(again.found && again.install.path, first.install.path);
  assert.match(again.found ? again.warning! : '', /portable CUDA 12 build: failed here on .*sublera doctor --fix tries it again/);

  // doctor --fix tries it again; a build that finds no GPU is not used either.
  const second = await ensureLlamaServer({ ...onLinux, env, fix: true, retryFailed: true, release: { ...assets('blind'), tag_name: 'b2' } });
  assert.equal(second.found, true);
  if (!second.found) return;
  assert.equal(second.install.path, first.install.path);
  assert.match(second.warning!, /its CUDA backend found no GPU/);

  // Without --fix, the working CPU engine is used as it is.
  const plain = await ensureLlamaServer({ ...onLinux, env });
  assert.equal(plain.found && plain.install.path, first.install.path);

  // A host that asks for one kind of build gets it, also when another kind is installed.
  const forcedEnv = { ...env, SUBLERA_LLAMA_VARIANT: 'cuda' };
  const refused = await ensureLlamaServer({ ...onLinux, env: forcedEnv });
  assert.equal(refused.found, false);
  assert.match(refused.found ? '' : refused.reason, /SUBLERA_LLAMA_VARIANT=cuda/);
});

test('a GPU build that finds the GPU becomes the engine', { skip: process.platform === 'win32' }, async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-gpu-engine-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const gpu = zipWith('llama-server', fakeEngine(['CUDA0: NVIDIA RTX A6000 (48669 MiB, 48000 MiB free)']));
  const { server, base, requests } = await serveFiles({ '/gpu.zip': gpu });
  t.after(() => { server.close(); });
  const env = { ...process.env, SUBLET_HOME: home, LLAMA_SERVER: '', SUBLERA_LLAMA_VARIANT: '', SUBLET_LLAMA_VARIANT: '' };
  const release = { tag_name: 'engine-b1', assets: [{ name: 'sublera-llama-b1-linux-cuda-12.8-x64.zip', browser_download_url: `${base}/gpu.zip` }] };
  const result = await ensureLlamaServer({ ...onLinux, env, fix: true, release });
  assert.equal(result.found, true);
  if (!result.found) return;
  assert.equal(result.warning, undefined);
  assert.equal(result.install.variant, 'cuda');
  assert.equal(result.install.build, 'portable');
  assert.equal((await findExistingLlama({ env }))?.path, result.install.path);
  assert.equal((await ensureLlamaServer({ ...onLinux, env, fix: true, release })).found, true);
  assert.equal(requests(), 1);
});

test('no usable build gives every reason in one message', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-no-engine-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const env = { ...process.env, SUBLET_HOME: home, LLAMA_SERVER: '', SUBLERA_LLAMA_VARIANT: '', SUBLET_LLAMA_VARIANT: '' };
  const urls: string[] = [];
  const result = await ensureLlamaServer({
    ...onLinux, env, fix: true,
    fetchImpl: async (url) => {
      urls.push(String(url));
      return /tags\/engine-/.test(String(url)) ? new Response('{}', { status: 404 }) : new Response(JSON.stringify([]), { status: 200 });
    },
  });
  assert.equal(result.found, false);
  if (result.found) return;
  assert.match(result.reason, /^No llama\.cpp build can run on this computer\./);
  assert.match(result.reason, /portable CUDA 12 build: No portable CUDA 12 build is published yet for linux\/x64/);
  assert.match(result.reason, /official Vulkan build: skipped, needs glibc 2\.34 or newer/);
  assert.match(result.reason, /set LLAMA_SERVER/);
  // The pinned engine release is asked for first, then the release list, each once.
  assert.deepEqual(urls, [
    'https://api.github.com/repos/somemusic69-sudo/Sublera-cli/releases/tags/engine-b12503',
    'https://api.github.com/repos/somemusic69-sudo/Sublera-cli/releases?per_page=100',
  ]);
});

test('a GitHub rate limit is reported as such, not as a missing build', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-rate-limit-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const env = { ...process.env, SUBLET_HOME: home, LLAMA_SERVER: '', SUBLERA_LLAMA_VARIANT: '', SUBLET_LLAMA_VARIANT: '' };
  const result = await ensureLlamaServer({
    ...onLinux, env, fix: true,
    fetchImpl: async () => new Response('{"message":"API rate limit exceeded for 203.0.113.9."}', { status: 403 }),
  });
  assert.equal(result.found, false);
  assert.match(result.found ? '' : result.reason, /portable CUDA 12 build: Could not read the Sublera engine releases: the GitHub API rate limit was reached/);
});

test('a portable build is found in a newer engine release from the release list', { skip: process.platform === 'win32' }, async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublet-engine-list-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const bytes = zipWith('llama-server', fakeEngine(['CUDA0: NVIDIA RTX A6000 (48669 MiB, 48000 MiB free)']));
  const env = { ...process.env, SUBLET_HOME: home, LLAMA_SERVER: '', SUBLERA_LLAMA_VARIANT: '', SUBLET_LLAMA_VARIANT: '' };
  const list = [
    { tag_name: 'v9.9.9', assets: [{ name: 'sublera-cli-v9.9.9.cdx.json', browser_download_url: 'https://x/sbom' }] },
    { tag_name: 'engine-b99999', assets: [{ name: 'sublera-llama-b99999-linux-cuda-12.4-x64.zip', browser_download_url: 'https://x/engine.zip', digest: `sha256:${sha256(bytes)}` }] },
  ];
  const result = await ensureLlamaServer({
    ...onLinux, env, fix: true,
    fetchImpl: async (url) => String(url).includes('/tags/') ? new Response('{}', { status: 404 })
      : String(url).includes('?per_page=100') ? new Response(JSON.stringify(list), { status: 200 })
        : String(url) === 'https://x/engine.zip' ? new Response(bytes, { status: 200, headers: { 'content-length': String(bytes.length) } })
          : new Response('[]', { status: 200 }),
  });
  assert.equal(result.found, true);
  if (!result.found) return;
  assert.equal(result.install.version, 'engine-b99999');
  assert.equal(result.install.verified, true);
});
