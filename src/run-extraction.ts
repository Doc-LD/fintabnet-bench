#!/usr/bin/env tsx
/**
 * Run DocLD table extraction on FinTabNet samples with concurrent requests.
 *
 * Usage:
 *   npx tsx src/run-extraction.ts
 *   npx tsx src/run-extraction.ts --limit 100 --concurrency 10
 *   npx tsx src/run-extraction.ts --skip-existing
 *
 * Set DOCLD_API_KEY in fintabnet-bench/.env or repo root .env.
 */

import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const benchRoot = path.resolve(__dirname, '..');
try {
  const { config } = await import('dotenv');
  config({ path: path.join(benchRoot, '.env') });
  config({ path: path.join(benchRoot, '..', '.env') });
} catch {
  // dotenv optional
}
import { extractTableFromImage } from './docld-provider.js';

interface ManifestEntry {
  id: string;
  imagePath: string;
  labelPath: string;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    dataDir: path.resolve('benchmark/data'),
    predictionsDir: path.resolve('benchmark/predictions'),
    limit: Infinity,
    concurrency: 8,
    skipExisting: false,
    retries: 2,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--data':
        config.dataDir = path.resolve(next);
        i++;
        break;
      case '--predictions':
        config.predictionsDir = path.resolve(next);
        i++;
        break;
      case '--limit':
        config.limit = parseInt(next, 10);
        i++;
        break;
      case '--concurrency':
        config.concurrency = parseInt(next, 10);
        i++;
        break;
      case '--skip-existing':
        config.skipExisting = true;
        break;
      case '--retries':
        config.retries = parseInt(next, 10);
        i++;
        break;
    }
  }
  return config;
}

function loadManifest(dataDir: string): ManifestEntry[] {
  const manifestPath = path.join(dataDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Manifest not found at ${manifestPath}. Run: npm run download`);
  }
  return JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
}

function isValidPrediction(predictionsDir: string, id: string): boolean {
  const predPath = path.join(predictionsDir, `${id}.html`);
  if (!fs.existsSync(predPath)) return false;
  const content = fs.readFileSync(predPath, 'utf-8');
  return !content.startsWith('<!-- empty') && !content.startsWith('<!-- error');
}

async function extractWithRetry(
  imagePath: string,
  retries: number
): Promise<string> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const html = await extractTableFromImage(imagePath);
      return html;
    } catch (err: any) {
      lastError = err;
      if (attempt < retries) {
        const delay = Math.min(2000 * Math.pow(2, attempt), 30000);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError;
}

async function processBatch(
  entries: ManifestEntry[],
  config: ReturnType<typeof parseArgs>,
  stats: { completed: number; errors: number; skipped: number; total: number },
  startTime: number
) {
  const promises = entries.map(async (entry) => {
    const imagePath = path.join(config.dataDir, entry.imagePath);
    const outPath = path.join(config.predictionsDir, `${entry.id}.html`);

    if (config.skipExisting && isValidPrediction(config.predictionsDir, entry.id)) {
      stats.skipped++;
      return;
    }

    try {
      const html = await extractWithRetry(imagePath, config.retries);
      if (html && html.length > 0) {
        fs.writeFileSync(outPath, html, 'utf-8');
        stats.completed++;
      } else {
        fs.writeFileSync(outPath, '<!-- empty -->', 'utf-8');
        stats.errors++;
      }
    } catch (err: any) {
      console.error(`  [${entry.id}] Error: ${err?.message || err}`);
      fs.writeFileSync(
        outPath,
        `<!-- error: ${(err?.message || err).replace(/--/g, '-')} -->`,
        'utf-8'
      );
      stats.errors++;
    }

    const done = stats.completed + stats.errors + stats.skipped;
    if (done % 5 === 0 || done === stats.total) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const rate = (done / parseFloat(elapsed)).toFixed(2);
      const eta = done > 0
        ? (((stats.total - done) / parseFloat(rate))).toFixed(0)
        : '?';
      console.log(
        `  [${done}/${stats.total}] ${elapsed}s | ` +
        `ok=${stats.completed} err=${stats.errors} skip=${stats.skipped} | ` +
        `${rate}/s | ETA: ${eta}s`
      );
    }
  });

  await Promise.all(promises);
}

async function main() {
  const config = parseArgs();

  if (!process.env.DOCLD_API_KEY) {
    console.error('Error: DOCLD_API_KEY is required. Set it in .env');
    process.exit(1);
  }

  const manifest = loadManifest(config.dataDir);
  const samples = manifest.slice(0, config.limit);

  fs.mkdirSync(config.predictionsDir, { recursive: true });

  console.log('FinTabNet Extraction');
  console.log('====================');
  console.log(` Data dir:     ${config.dataDir}`);
  console.log(` Predictions:  ${config.predictionsDir}`);
  console.log(` Samples:      ${samples.length}`);
  console.log(` Concurrency:  ${config.concurrency}`);
  console.log(` Skip existing: ${config.skipExisting}`);
  console.log(` API: ${process.env.DOCLD_BASE_URL || 'https://docld.com'}`);
  console.log();

  const stats = { completed: 0, errors: 0, skipped: 0, total: samples.length };
  const startTime = Date.now();

  for (let i = 0; i < samples.length; i += config.concurrency) {
    const batch = samples.slice(i, i + config.concurrency);
    await processBatch(batch, config, stats, startTime);
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log('\n====================');
  console.log(` Done in ${elapsed}s`);
  console.log(` Completed: ${stats.completed}`);
  console.log(` Errors: ${stats.errors}`);
  console.log(` Skipped: ${stats.skipped}`);
  console.log(` Predictions: ${config.predictionsDir}`);
}

main().catch((err) => {
  console.error('Extraction failed:', err);
  process.exit(1);
});
