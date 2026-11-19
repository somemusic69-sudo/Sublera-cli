import { fetchJson, HttpError } from './util/http.js';

/** One model from the server's `/api/catalog`: what a host can serve with `sublera host <id>`, and what GPUs ask for it now. */
export type CatalogEntry = {
  id: string;
  label: string;
  note: string;
  hfUrl: string;
  sizeGb: number;
  maxContext: number;
  minVramMb: number;
  hosts: number;
  lowestPriceOutUsd: number | null;
  lowestPriceHourUsd: number | null;
};

export async function fetchCatalog(server: string): Promise<CatalogEntry[]> {
  try {
    const data = await fetchJson<{ models?: unknown }>(`${server}/api/catalog`, { timeoutMs: 15_000 });
    return Array.isArray(data.models) ? (data.models as CatalogEntry[]).filter((m) => m && typeof m.id === 'string' && typeof m.hfUrl === 'string') : [];
  } catch (error) {
    if (error instanceof HttpError && error.status === 404) {
      throw new Error(`${server} has no model list. Give a Hugging Face GGUF URL instead: sublera host <url>`);
    }
    throw error;
  }
}

/** A catalog model by its id (`llama-3.1-8b`) or its name (`Llama 3.1 8B`), in any letter case. */
export function findCatalogModel(name: string, catalog: CatalogEntry[]): CatalogEntry | null {
  const key = name.trim().toLowerCase();
  if (!key) return null;
  return catalog.find((m) => m.id.toLowerCase() === key) ?? catalog.find((m) => m.label.toLowerCase() === key) ?? null;
}

/** true or false when the GPU memory is known; null when it is not. */
export function fitsGpu(entry: Pick<CatalogEntry, 'minVramMb'>, vramMb: number | null): boolean | null {
  return vramMb && vramMb > 0 ? entry.minVramMb <= vramMb : null;
}

const money = (usd: number) => `$${usd.toFixed(2)}`;
const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`;

/** What other GPUs do with a model now: "none", or how many serve it and their lowest prices. */
export function supplyText(m: Pick<CatalogEntry, 'hosts' | 'lowestPriceOutUsd' | 'lowestPriceHourUsd'>): string {
  if (!m.hosts) return 'none';
  const prices = [m.lowestPriceOutUsd != null ? `from ${money(m.lowestPriceOutUsd)} per 1M` : '', m.lowestPriceHourUsd != null ? `${money(m.lowestPriceHourUsd)}/h` : '']
    .filter(Boolean).join(', ');
  return `${m.hosts} ${m.hosts === 1 ? 'GPU' : 'GPUs'}${prices ? `, ${prices}` : ''}`;
}

/** The numbered model table of `sublera host`. */
export function catalogTable(catalog: CatalogEntry[], vramMb: number | null): string {
  const rows = catalog.map((m, i) => {
    const fit = fitsGpu(m, vramMb);
    return [String(i + 1), m.label, m.id, `${m.sizeGb.toFixed(1)} GB`, gb(m.minVramMb), fit == null ? '?' : fit ? 'yes' : 'no', supplyText(m)];
  });
  const head = ['#', 'Model', 'Id', 'Size', 'Needs', 'Fits', 'Other GPUs now'];
  const widths = head.map((h, c) => Math.max(h.length, ...rows.map((r) => r[c]!.length)));
  const line = (cells: string[]) => cells.map((cell, c) => (c === 0 ? cell.padStart(widths[c]!) : c === cells.length - 1 ? cell : cell.padEnd(widths[c]!))).join('  ');
  return [line(head), ...rows.map(line)].join('\n');
}

/** A price typed at a prompt: Enter keeps the default; anything else must be a number from 0 to 50 (USD). */
export function parsePriceAnswer(raw: string, fallback: number): number | null {
  const text = raw.trim().replace(/^\$/, '');
  if (!text) return fallback;
  const n = Number(text);
  return Number.isFinite(n) && n >= 0 && n <= 50 ? n : null;
}

/** The command that does the same as the menu, for scripts and next time. */
export function hostCommand(id: string, price: number, priceHour: number): string {
  return `npx sublera-cli host ${id} --price ${price} --price-hour ${priceHour}`;
}
