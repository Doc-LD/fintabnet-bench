#!/usr/bin/env tsx
/**
 * Generate visualization PNGs from benchmark results.
 * Uses a simple approach: output SVG/instructions for manual chart creation,
 * or use a headless approach if available.
 *
 * For production: use Python matplotlib or a chart service.
 * This script produces data files that can be used to generate charts.
 */

import fs from 'node:fs';
import path from 'node:path';

const resultsPath = path.resolve('benchmark/results/results.json');
const outputDir = path.resolve('benchmark/charts');

interface BenchmarkSummary {
  meanAccuracy: number;
  medianAccuracy: number;
  p25Accuracy: number;
  p75Accuracy: number;
  minAccuracy?: number;
  maxAccuracy?: number;
  results?: Array<{ id: string; score: number }>;
}

function main() {
  if (!fs.existsSync(resultsPath)) {
    console.error(`Results not found at ${resultsPath}. Run: npm run score`);
    process.exit(1);
  }

  const summary = JSON.parse(fs.readFileSync(resultsPath, 'utf-8')) as BenchmarkSummary;
  fs.mkdirSync(outputDir, { recursive: true });

  // Chart 1: Overall comparison (DocLD vs academic baselines)
  const comparisonData = [
    { name: 'DocLD', score: (summary.meanAccuracy * 100).toFixed(1), highlight: true },
    { name: 'GTE (IBM)', score: '~78', highlight: false },
    { name: 'TATR (Microsoft)', score: '~65', highlight: false },
  ];
  fs.writeFileSync(
    path.join(outputDir, 'overall-comparison.json'),
    JSON.stringify(comparisonData, null, 2)
  );

  // Chart 2: Score distribution summary
  const distData = {
    mean: summary.meanAccuracy * 100,
    median: summary.medianAccuracy * 100,
    p25: summary.p25Accuracy * 100,
    p75: summary.p75Accuracy * 100,
    min: (summary.minAccuracy ?? 0) * 100,
    max: (summary.maxAccuracy ?? 1) * 100,
  };
  fs.writeFileSync(
    path.join(outputDir, 'score-distribution.json'),
    JSON.stringify(distData, null, 2)
  );

  // If we have per-sample results, output score histogram bins
  if (summary.results && summary.results.length > 0) {
    const scores = summary.results
      .filter((r) => !(r as any).error)
      .map((r) => r.score * 100);
    const bins = [0, 50, 60, 70, 80, 90, 95, 100];
    const hist: Record<string, number> = {};
    for (let i = 0; i < bins.length - 1; i++) {
      const key = `${bins[i]}-${bins[i + 1]}`;
      hist[key] = scores.filter((s) => s >= bins[i] && s < bins[i + 1]).length;
    }
    hist['100'] = scores.filter((s) => s === 100).length;
    fs.writeFileSync(
      path.join(outputDir, 'score-histogram.json'),
      JSON.stringify(hist, null, 2)
    );
  }

  console.log('Chart data written to', outputDir);
  console.log('  overall-comparison.json');
  console.log('  score-distribution.json');
}

main();
