import assert from 'node:assert/strict';
import test from 'node:test';
import { pickReleaseAsset } from '../src/engine/binaryManager.js';
import {
  cudaProblem, detectEngineHost, engineCandidates, explainEngineFailure, parseDeviceList, parseNvidiaDriverInfo,
  type EngineCandidate, type EngineHost,
} from '../src/engine/engineHost.js';

const ubuntu24: EngineHost = {
  platform: 'linux', arch: 'x64', vendor: 'nvidia', glibc: '2.39',
  nvidiaDriver: 580, computeCapability: 8.6, vulkan: { loader: true, driver: true }, rocm: null,
};
const labels = (host: EngineHost, forced?: Parameters<typeof engineCandidates>[1]) => engineCandidates(host, forced).candidates.map((c) => c.label);
const skipped = (host: EngineHost) => Object.fromEntries(engineCandidates(host).skipped.map((s) => [s.label, s.reason]));

test('a current Linux with an NVIDIA GPU tries official CUDA builds first, then portable, Vulkan and CPU', () => {
  assert.deepEqual(labels(ubuntu24), [
    'official CUDA 12 build', 'official CUDA 13 build', 'portable CUDA 12 build',
    'official Vulkan build', 'official CPU build', 'portable CPU build',
  ]);
});

test('an older Linux image (glibc 2.31, no Vulkan driver) skips the official builds and uses the portable ones', () => {
  const host: EngineHost = { ...ubuntu24, glibc: '2.31', nvidiaDriver: 535, vulkan: { loader: false, driver: false } };
  assert.deepEqual(labels(host), ['portable CUDA 12 build', 'portable CPU build']);
  const why = skipped(host);
  assert.match(why['official CUDA 12 build']!, /needs glibc 2\.38 or newer; this system has glibc 2\.31/);
  assert.match(why['official CUDA 13 build']!, /glibc 2\.38/);
  assert.match(why['official Vulkan build']!, /glibc 2\.34/);
});

test('Ubuntu 22.04 without a Vulkan driver keeps the official CPU build and explains the Vulkan problem', () => {
  const host: EngineHost = { ...ubuntu24, glibc: '2.35', nvidiaDriver: 550, vulkan: { loader: true, driver: false } };
  assert.deepEqual(labels(host), ['portable CUDA 12 build', 'official CPU build', 'portable CPU build']);
  assert.match(skipped(host)['official Vulkan build']!, /no Vulkan driver \(ICD\) found; in a container, add "graphics" to NVIDIA_DRIVER_CAPABILITIES/);
});

test('driver and GPU generation decide which CUDA builds can run', () => {
  const pascal: EngineHost = { ...ubuntu24, computeCapability: 6.1 };
  assert.ok(labels(pascal).includes('official CUDA 12 build'));
  assert.match(skipped(pascal)['official CUDA 13 build']!, /compute capability 6\.1/);
  const oldDriver: EngineHost = { ...ubuntu24, nvidiaDriver: 470 };
  assert.match(skipped(oldDriver)['official CUDA 12 build']!, /needs NVIDIA driver 550 or newer; this system has driver 470/);
  assert.match(skipped(oldDriver)['portable CUDA 12 build']!, /driver 525/);
  assert.deepEqual(labels(oldDriver), ['official Vulkan build', 'official CPU build', 'portable CPU build']);
  const driver550: EngineHost = { ...ubuntu24, nvidiaDriver: 550 };
  assert.match(skipped(driver550)['official CUDA 13 build']!, /driver 580/);
});

test('a CUDA build is skipped when it has no GPU code the driver can run for this GPU', () => {
  const gpu = (computeCapability: number, nvidiaDriver: number) => ({ platform: 'linux' as const, computeCapability, nvidiaDriver });
  // Tesla P100 and Maxwell: only the portable build has native code for them.
  assert.equal(cudaProblem(gpu(6.0, 550), 'portable', 12), null);
  assert.equal(cudaProblem(gpu(5.2, 550), 'portable', 12), null);
  assert.match(cudaProblem(gpu(6.0, 550), 'official', 12)!, /needs NVIDIA driver 570 or newer for compute capability 6\.0/);
  assert.match(cudaProblem(gpu(6.0, 590), 'official', 13)!, /has no GPU code for compute capability 6\.0/);
  // T4 and A100 need a driver that can compile the official build's PTX; the portable build has native code.
  assert.match(cudaProblem(gpu(7.5, 560), 'official', 12)!, /driver 570 or newer for compute capability 7\.5/);
  assert.equal(cudaProblem(gpu(8.0, 570), 'official', 12), null);
  assert.equal(cudaProblem(gpu(8.0, 535), 'portable', 12), null);
  // RTX A6000 has native code in every build.
  assert.equal(cudaProblem(gpu(8.6, 550), 'official', 12), null);
  assert.equal(cudaProblem(gpu(8.6, 525), 'portable', 12), null);
  assert.match(cudaProblem(gpu(8.6, 535), 'official', 12)!, /driver 550 or newer/);
  // Blackwell: native code in the official CUDA 12 build, PTX in the portable one.
  assert.equal(cudaProblem(gpu(12.0, 575), 'official', 12), null);
  assert.equal(cudaProblem(gpu(12.0, 575), 'portable', 12), null);
  assert.equal(cudaProblem({ platform: 'linux', computeCapability: null, nvidiaDriver: null }, 'official', 13), null);
  const t4: EngineHost = { ...ubuntu24, nvidiaDriver: 535, computeCapability: 7.5 };
  assert.deepEqual(labels(t4), ['portable CUDA 12 build', 'official Vulkan build', 'official CPU build', 'portable CPU build']);
});

test('Windows, arm64, AMD, Apple, musl and a forced variant get their own build lists', () => {
  assert.deepEqual(labels({ ...ubuntu24, platform: 'win32', glibc: null, vulkan: null }), [
    'official CUDA 12 build', 'official CUDA 13 build', 'official Vulkan build', 'official CPU build',
  ]);
  assert.deepEqual(labels({ ...ubuntu24, arch: 'arm64' }), ['official CUDA 13 build', 'official Vulkan build', 'official CPU build']);
  const amd: EngineHost = { ...ubuntu24, vendor: 'amd', nvidiaDriver: null, computeCapability: null, vulkan: { loader: false, driver: false }, rocm: { driver: true, runtime: true } };
  assert.deepEqual(labels(amd), ['official ROCm build', 'official CPU build', 'portable CPU build']);
  assert.match(skipped({ ...amd, rocm: { driver: true, runtime: false } })['official ROCm build']!, /libamdhip64\.so\.7/);
  assert.deepEqual(labels({ ...ubuntu24, platform: 'darwin', arch: 'arm64', vendor: 'apple' }), ['official Metal build']);
  const musl: EngineHost = { ...ubuntu24, glibc: null };
  assert.deepEqual(labels(musl), []);
  assert.match(skipped(musl)['portable CPU build']!, /musl/);
  assert.deepEqual(labels(ubuntu24, 'cpu'), ['official CPU build', 'portable CPU build']);
  assert.deepEqual(labels({ ...ubuntu24, vendor: null }), ['official CPU build', 'portable CPU build']);
});

const B12503 = [
  'cudart-llama-b12503-bin-ubuntu-cuda-12.8-x64.tar.gz', 'cudart-llama-b12503-bin-ubuntu-cuda-13.5-arm64.tar.gz',
  'cudart-llama-b12503-bin-ubuntu-cuda-13.5-x64.tar.gz', 'cudart-llama-bin-win-cuda-12.4-x64.zip',
  'cudart-llama-bin-win-cuda-13.5-arm64.zip', 'cudart-llama-bin-win-cuda-13.5-x64.zip',
  'llama-b12503-bin-linux-arm64-snapdragon.tar.gz', 'llama-b12503-bin-macos-arm64.tar.gz', 'llama-b12503-bin-macos-x64.tar.gz',
  'llama-b12503-bin-ubuntu-26.04-s390x.tar.gz', 'llama-b12503-bin-ubuntu-arm64.tar.gz',
  'llama-b12503-bin-ubuntu-cuda-12.8-x64.tar.gz', 'llama-b12503-bin-ubuntu-cuda-13.5-arm64.tar.gz', 'llama-b12503-bin-ubuntu-cuda-13.5-x64.tar.gz',
  'llama-b12503-bin-ubuntu-openvino-2026.4.1-x64.tar.gz', 'llama-b12503-bin-ubuntu-rocm-10.0-x64.tar.gz',
  'llama-b12503-bin-ubuntu-sycl-fp16-x64.tar.gz', 'llama-b12503-bin-ubuntu-vulkan-arm64.tar.gz', 'llama-b12503-bin-ubuntu-vulkan-x64.tar.gz',
  'llama-b12503-bin-ubuntu-x64.tar.gz', 'llama-b12503-bin-win-cpu-arm64.zip', 'llama-b12503-bin-win-cpu-x64.zip',
  'llama-b12503-bin-win-cuda-12.4-x64.zip', 'llama-b12503-bin-win-cuda-13.5-arm64.zip', 'llama-b12503-bin-win-cuda-13.5-x64.zip',
  'llama-b12503-bin-win-rocm-10.0-x64.zip', 'llama-b12503-bin-win-vulkan-arm64.zip', 'llama-b12503-bin-win-vulkan-x64.zip',
  'llama-b12503-ui.tar.gz', 'llama-b12503-xcframework.zip',
  'sublera-llama-b12503-linux-cuda-12.8-x64.tar.gz', 'sublera-llama-b12503-linux-cpu-x64.tar.gz',
].map((name) => ({ name, browser_download_url: `http://x/${name}` }));

test('each build picks its own llama.cpp release asset and CUDA runtime', () => {
  const release = { tag_name: 'b12503', assets: B12503 };
  const pick = (platform: NodeJS.Platform, arch: string, candidate: Omit<EngineCandidate, 'label'>) => {
    const picked = pickReleaseAsset(release, platform, arch, { ...candidate, label: 'test' });
    return [picked.main?.name ?? null, ...picked.companions.map((a) => a.name)];
  };
  assert.deepEqual(pick('linux', 'x64', { variant: 'cuda', source: 'official', cuda: 12 }), ['llama-b12503-bin-ubuntu-cuda-12.8-x64.tar.gz', 'cudart-llama-b12503-bin-ubuntu-cuda-12.8-x64.tar.gz']);
  assert.deepEqual(pick('linux', 'x64', { variant: 'cuda', source: 'official', cuda: 13 }), ['llama-b12503-bin-ubuntu-cuda-13.5-x64.tar.gz', 'cudart-llama-b12503-bin-ubuntu-cuda-13.5-x64.tar.gz']);
  assert.deepEqual(pick('linux', 'arm64', { variant: 'cuda', source: 'official', cuda: 13 }), ['llama-b12503-bin-ubuntu-cuda-13.5-arm64.tar.gz', 'cudart-llama-b12503-bin-ubuntu-cuda-13.5-arm64.tar.gz']);
  assert.deepEqual(pick('linux', 'x64', { variant: 'vulkan', source: 'official' }), ['llama-b12503-bin-ubuntu-vulkan-x64.tar.gz']);
  assert.deepEqual(pick('linux', 'x64', { variant: 'rocm', source: 'official' }), ['llama-b12503-bin-ubuntu-rocm-10.0-x64.tar.gz']);
  assert.deepEqual(pick('linux', 'x64', { variant: 'cpu', source: 'official' }), ['llama-b12503-bin-ubuntu-x64.tar.gz']);
  assert.deepEqual(pick('linux', 'arm64', { variant: 'cpu', source: 'official' }), ['llama-b12503-bin-ubuntu-arm64.tar.gz']);
  assert.deepEqual(pick('linux', 'x64', { variant: 'cuda', source: 'portable', cuda: 12 }), ['sublera-llama-b12503-linux-cuda-12.8-x64.tar.gz']);
  assert.deepEqual(pick('linux', 'x64', { variant: 'cpu', source: 'portable' }), ['sublera-llama-b12503-linux-cpu-x64.tar.gz']);
  assert.deepEqual(pick('win32', 'x64', { variant: 'cuda', source: 'official', cuda: 12 }), ['llama-b12503-bin-win-cuda-12.4-x64.zip', 'cudart-llama-bin-win-cuda-12.4-x64.zip']);
  assert.deepEqual(pick('win32', 'arm64', { variant: 'vulkan', source: 'official' }), ['llama-b12503-bin-win-vulkan-arm64.zip']);
  assert.deepEqual(pick('win32', 'arm64', { variant: 'cpu', source: 'official' }), ['llama-b12503-bin-win-cpu-arm64.zip']);
  assert.deepEqual(pick('linux', 'arm64', { variant: 'cuda', source: 'official', cuda: 12 }), [null]);
  assert.deepEqual(pick('win32', 'x64', { variant: 'cuda', source: 'portable', cuda: 12 }), [null]);
});

test('loader errors are explained in plain words', () => {
  const glibc = "/home/u/.sublet/bin/llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by /home/u/.sublet/bin/libllama-server-impl.so)\n"
    + "/home/u/.sublet/bin/llama-server: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/u/.sublet/bin/libllama-server-impl.so)\n"
    + "/home/u/.sublet/bin/llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by /home/u/.sublet/bin/libllama.so.1)";
  assert.match(explainEngineFailure(glibc, { glibc: '2.31' })!, /older than this build needs \(GLIBC_2\.38, GLIBCXX_3\.4\.32; this system has glibc 2\.31\)/);
  assert.match(explainEngineFailure('llama-server: error while loading shared libraries: libvulkan.so.1: cannot open shared object file: No such file or directory')!, /Vulkan loader/);
  assert.match(explainEngineFailure('libggml-cuda.so: libcuda.so.1: cannot open shared object file: No such file or directory')!, /--gpus all/);
  assert.match(explainEngineFailure('error while loading shared libraries: libgomp.so.1: cannot open shared object file')!, /libgomp1/);
  assert.equal(explainEngineFailure('segmentation fault'), null);
});

test('only GPU lines of --list-devices count as devices', () => {
  assert.deepEqual(parseDeviceList('Available devices:\n  (none)\n'), []);
  assert.deepEqual(parseDeviceList('ggml_cuda_init: found 1 CUDA devices\nAvailable devices:\n  CUDA0: NVIDIA RTX A6000 (48669 MiB, 48000 MiB free)\n'), ['CUDA0: NVIDIA RTX A6000 (48669 MiB, 48000 MiB free)']);
  assert.deepEqual(parseDeviceList(''), []);
});

test('nvidia-smi driver and compute capability use the oldest GPU', async () => {
  assert.deepEqual(parseNvidiaDriverInfo('535.183.01, 8.6\n535.183.01, 7.0\n'), { nvidiaDriver: 535, computeCapability: 7 });
  assert.deepEqual(parseNvidiaDriverInfo('470.256.02\n'), { nvidiaDriver: 470, computeCapability: null });
  const calls: string[] = [];
  const host = await detectEngineHost({
    platform: 'win32', arch: 'x64', glibc: null,
    gpu: { primary: { vendor: 'nvidia' } as never, gpus: [], source: 'test', platform: 'win32', checkedAt: 0, warnings: [] },
    // An old driver rejects compute_cap; the second query still finds the driver.
    exec: async (_command, args = []) => {
      calls.push(args[0]!);
      return args[0]!.includes('compute_cap')
        ? { code: 2, stdout: '', stderr: 'Field "compute_cap" is not a valid field to query.', timedOut: false }
        : { code: 0, stdout: '470.256.02\n', stderr: '', timedOut: false };
    },
  });
  assert.deepEqual(calls, ['--query-gpu=driver_version,compute_cap', '--query-gpu=driver_version']);
  assert.equal(host.nvidiaDriver, 470);
  assert.equal(host.vulkan, null);
});
