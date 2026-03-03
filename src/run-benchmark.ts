#!/usr/bin/env tsx
/**
 * Score table extraction predictions against FinTabNet ground truth.
 *
 * Usage:
 *   npx tsx src/run-benchmark.ts --predictions ./benchmark/predictions
 *   npx tsx src/run-benchmark.ts --predictions ./preds --limit 50 --output results.json
 */

import fs from 'node:fs';
import path from 'node:path';
import {
  tableScore,
  htmlTableToArray,
  markdownTableToArray,
  type TableBenchResult,
  summarizeBenchmark,
} from './table-scorer.js';

interface ManifestEntry {
  id: string;
  imagePath: string;
  labelPath: string;
}

interface RunConfig {
  dataDir: string;
  outputPath: string;
  limit: number;
  predictionsDir: string;
}

function parseArgs(): RunConfig {
  const args = process.argv.slice(2);
  const config: RunConfig = {
    dataDir: path.resolve('benchmark/data'),
    outputPath: path.resolve('benchmark/results/results.json'),
    limit: Infinity,
    predictionsDir: '',
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--data':
        config.dataDir = path.resolve(next);
        i++;
        break;
      case '--output':
        config.outputPath = path.resolve(next);
        i++;
        break;
      case '--limit':
        config.limit = parseInt(next, 10);
        i++;
        break;
      case '--predictions':
        config.predictionsDir = path.resolve(next);
        i++;
        break;
    }
  }

  if (!config.predictionsDir) {
    config.predictionsDir = path.resolve('benchmark/predictions');
  }
  return config;
}

function loadManifest(dataDir: string): ManifestEntry[] {
  const manifestPath = path.join(dataDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    throw new Error(
      `Manifest not found at ${manifestPath}. Run: npm run download`
    );
  }
  return JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
}

function loadGroundTruth(dataDir: string, entry: ManifestEntry): string[][] {
  const labelPath = path.join(dataDir, entry.labelPath);
  const raw = fs.readFileSync(labelPath, 'utf-8');
  return htmlTableToArray(raw);
}

function loadPrediction(
  predictionsDir: string,
  id: string
): string[][] | null {
  for (const ext of ['.html', '.htm', '.md']) {
    const predPath = path.join(predictionsDir, `${id}${ext}`);
    if (!fs.existsSync(predPath)) continue;
    const raw = fs.readFileSync(predPath, 'utf-8');
    if (raw.includes('<!-- error:')) return null;
    if (ext === '.md') return markdownTableToArray(raw);
    return htmlTableToArray(raw);
  }
  return null;
}

async function main() {
  const config = parseArgs();

  console.log('FinTabNet Benchmark Scorer');
  console.log('=========================');
  console.log(` Data dir: ${config.dataDir}`);
  console.log(` Predictions: ${config.predictionsDir}`);
  console.log(` Output: ${config.outputPath}`);
  console.log(` Limit: ${config.limit === Infinity ? 'all' : config.limit}`);
  console.log();

  const manifest = loadManifest(config.dataDir);
  const samples = manifest.slice(0, config.limit);

  const startTime = Date.now();
  const results: TableBenchResult[] = [];

  for (let i = 0; i < samples.length; i++) {
    const entry = samples[i];
    const result: TableBenchResult = {
      id: entry.id,
      score: 0,
      predictedRows: 0,
      predictedCols: 0,
      groundTruthRows: 0,
      groundTruthCols: 0,
    };

    try {
      const gt = loadGroundTruth(config.dataDir, entry);
      result.groundTruthRows = gt.length;
      result.groundTruthCols = gt.length > 0 ? gt[0].length : 0;

      const predicted = loadPrediction(config.predictionsDir, entry.id);
      if (!predicted || predicted.length === 0) {
        result.error = 'No prediction found';
        results.push(result);
        continue;
      }

      result.predictedRows = predicted.length;
      result.predictedCols = predicted.length > 0 ? predicted[0].length : 0;
      result.score = tableScore(gt, predicted);
    } catch (err: any) {
      result.error = err?.message || String(err);
    }

    results.push(result);

    if ((i + 1) % 50 === 0 || i + 1 === samples.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const scored = results.filter((r) => !r.error);
      const avg =
        scored.length > 0
          ? (
              (scored.reduce((a, b) => a + b.score, 0) / scored.length) *
              100
            ).toFixed(1)
          : '–';
      console.log(` [${i + 1}/${samples.length}] ${elapsed}s — running avg: ${avg}%`);
    }
  }

  const summary = summarizeBenchmark(results);
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  console.log('\n========== Results ==========');
  console.log(` Total samples: ${summary.totalSamples}`);
  console.log(` Scored: ${summary.scoredSamples}`);
  console.log(` Errors: ${summary.errorSamples}`);
  console.log(` Mean accuracy: ${(summary.meanAccuracy * 100).toFixed(1)}%`);
  console.log(` Median: ${(summary.medianAccuracy * 100).toFixed(1)}%`);
  console.log(
    ` P25 / P75: ${(summary.p25Accuracy * 100).toFixed(1)}% / ${(summary.p75Accuracy * 100).toFixed(1)}%`
  );
  console.log(
    ` Min / Max: ${(summary.minAccuracy * 100).toFixed(1)}% / ${(summary.maxAccuracy * 100).toFixed(1)}%`
  );
  console.log(` Time: ${elapsed}s`);

  const outDir = path.dirname(config.outputPath);
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(
    config.outputPath,
    JSON.stringify(
      { ...summary, elapsedSeconds: parseFloat(elapsed) },
      null,
      2
    )
  );
  console.log(`\nResults written to ${config.outputPath}`);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
