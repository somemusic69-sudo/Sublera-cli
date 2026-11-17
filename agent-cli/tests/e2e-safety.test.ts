import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

test('end-to-end test refuses to send test actions to a live website', () => {
  const script = fileURLToPath(new URL('../scripts/e2e.mjs', import.meta.url));
  const result = spawnSync(process.execPath, [script], {
    env: { ...process.env, SUBLET_E2E_SERVER: 'https://sublera.com', SUBLET_E2E_INBOX: 'unused' },
    encoding: 'utf8', timeout: 5_000,
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /refusing to use a live site/);
});
