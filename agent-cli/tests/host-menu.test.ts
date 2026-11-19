import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { catalogTable, findCatalogModel, fitsGpu, hostCommand, parsePriceAnswer, supplyText, type CatalogEntry } from '../src/hostMenu.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const entry = fileURLToPath(new URL('../src/index.ts', import.meta.url));

const CATALOG: CatalogEntry[] = [
  { id: 'tinyllama-1.1b', label: 'TinyLlama 1.1B', note: 'tiny', hfUrl: 'https://huggingface.co/x/tiny/resolve/main/tiny.gguf', sizeGb: 0.67, maxContext: 2048, minVramMb: 1600, hosts: 0, lowestPriceOutUsd: null, lowestPriceHourUsd: null },
  { id: 'llama-3.1-8b', label: 'Llama 3.1 8B', note: 'general chat', hfUrl: 'https://huggingface.co/x/llama/resolve/main/llama.gguf', sizeGb: 4.92, maxContext: 131072, minVramMb: 6800, hosts: 2, lowestPriceOutUsd: 0.2, lowestPriceHourUsd: 0.6 },
  { id: 'qwen2.5-32b', label: 'Qwen2.5 32B', note: 'large', hfUrl: 'https://huggingface.co/x/qwen/resolve/main/qwen.gguf', sizeGb: 19.85, maxContext: 32768, minVramMb: 23000, hosts: 1, lowestPriceOutUsd: 0.9, lowestPriceHourUsd: null },
];

test('the host menu finds models by id or name, knows what fits, and shows what other GPUs ask', () => {
  assert.equal(findCatalogModel('LLAMA-3.1-8B', CATALOG)?.id, 'llama-3.1-8b');
  assert.equal(findCatalogModel(' Llama 3.1 8B ', CATALOG)?.id, 'llama-3.1-8b');
  assert.equal(findCatalogModel('nothing', CATALOG), null);
  assert.equal(findCatalogModel('', CATALOG), null);
  assert.equal(fitsGpu(CATALOG[2]!, 15_360), false, 'a 32B model does not fit a 16 GB T4');
  assert.equal(fitsGpu(CATALOG[1]!, 15_360), true);
  assert.equal(fitsGpu(CATALOG[1]!, null), null);
  assert.equal(supplyText(CATALOG[0]!), 'none');
  assert.equal(supplyText(CATALOG[1]!), '2 GPUs, from $0.20 per 1M, $0.60/h');
  assert.equal(supplyText(CATALOG[2]!), '1 GPU, from $0.90 per 1M');
  const table = catalogTable(CATALOG, 15_360).split('\n');
  assert.match(table[0]!, /#\s+Model\s+Id\s+Size\s+Needs\s+Fits\s+Other GPUs now/);
  assert.match(table[2]!, /^2 {2}Llama 3\.1 8B\s+llama-3\.1-8b\s+4\.9 GB\s+6\.6 GB\s+yes\s+2 GPUs, from \$0\.20 per 1M, \$0\.60\/h$/);
  assert.match(table[3]!, /Qwen2\.5 32B.*\bno\b/);
  assert.match(catalogTable(CATALOG, null), /\s\?\s/);
  assert.equal(parsePriceAnswer('', 0.25), 0.25);
  assert.equal(parsePriceAnswer(' $0.4 ', 0.25), 0.4);
  assert.equal(parsePriceAnswer('0', 0.5), 0);
  assert.equal(parsePriceAnswer('51', 0.5), null);
  assert.equal(parsePriceAnswer('cheap', 0.5), null);
  assert.equal(hostCommand('llama-3.1-8b', 0.2, 0.6), 'npx sublera-cli host llama-3.1-8b --price 0.2 --price-hour 0.6');
});

function cli(home: string, args: string[], extra: NodeJS.ProcessEnv = {}): Promise<{ code: number; out: string; err: string }> {
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^(SUBLET_|SUBLERA_|https?_proxy$|no_proxy$)/i.test(key)));
  return new Promise((resolve, reject) => {
    execFile(process.execPath, ['--import', 'tsx', entry, ...args], { cwd: root, env: { ...env, SUBLET_HOME: home, ...extra }, timeout: 20_000 }, (error, out, err) => {
      if (error?.killed) { reject(new Error(`CLI hung: ${err}`)); return; }
      resolve({ code: typeof error?.code === 'number' ? error.code : error ? -1 : 0, out, err });
    });
  });
}

test('sublera host --list prints the models, and host takes a model id instead of a URL', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'sublera-host-menu-'));
  const server = createServer((req, res) => {
    if (req.url === '/api/catalog') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify({ models: CATALOG })); return; }
    res.writeHead(404); res.end();
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  t.after(async () => { server.close(); await rm(home, { recursive: true, force: true }); });

  const list = await cli(home, ['host', '--list', '--server', url, '--gpu', 'Tesla T4', '--vram', '15360', '--no-color']);
  assert.equal(list.code, 0, list.err);
  assert.match(list.out, /Your GPU: Tesla T4, 15 GB/);
  assert.match(list.out, /llama-3\.1-8b .*yes .*2 GPUs, from \$0\.20 per 1M, \$0\.60\/h/);
  assert.match(list.out, /Run: sublera host <id>/);

  const json = await cli(home, ['host', '--list', '--json', '--server', url]);
  assert.equal(json.code, 0, json.err);
  assert.deepEqual(JSON.parse(json.out).models.map((m: CatalogEntry) => m.id), ['tinyllama-1.1b', 'llama-3.1-8b', 'qwen2.5-32b']);

  const usage = await cli(home, ['host', '--non-interactive', '--server', url]);
  assert.equal(usage.code, 2, usage.err);
  assert.match(`${usage.out}${usage.err}`, /sublera host --list/);

  const unknown = await cli(home, ['host', 'no-such-model', '--non-interactive', '--server', url], { SUBLERA_KEY: 'sh_test' });
  assert.equal(unknown.code, 2, unknown.err);
  assert.match(`${unknown.out}${unknown.err}`, /Unknown model "no-such-model"/);

  const old = await cli(home, ['host', '--list', '--server', `${url}/old`]);
  assert.notEqual(old.code, 0);
  assert.match(`${old.out}${old.err}`, /has no model list\. Give a Hugging Face GGUF URL instead/);
});
