#!/usr/bin/env tsx
/**
 * Direct OpenAI-based table extraction for FinTabNet benchmark.
 * Uses gpt-5.2 with two-step counting prompt and multi-pass verification.
 *
 * Usage:
 *   npx tsx src/run-direct-extraction.ts --limit 500 --concurrency 8
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
} catch {}

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.BENCH_MODEL || 'gpt-5.2';

interface ManifestEntry {
  id: string;
  imagePath: string;
  labelPath: string;
}

interface Args {
  dataDir: string;
  predictionsDir: string;
  limit: number;
  concurrency: number;
  skipExisting: boolean;
  retries: number;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const config: Args = {
    dataDir: path.resolve('benchmark/data'),
    predictionsDir: path.resolve('benchmark/predictions'),
    limit: 500,
    concurrency: 8,
    skipExisting: true,
    retries: 3,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--data': config.dataDir = path.resolve(next); i++; break;
      case '--predictions': config.predictionsDir = path.resolve(next); i++; break;
      case '--limit': config.limit = parseInt(next, 10); i++; break;
      case '--concurrency': config.concurrency = parseInt(next, 10); i++; break;
      case '--skip-existing': config.skipExisting = true; break;
      case '--no-skip': config.skipExisting = false; break;
      case '--retries': config.retries = parseInt(next, 10); i++; break;
    }
  }
  return config;
}

// ---------------------------------------------------------------------------
// Extraction prompt: focused, concise, avoids over-extraction
// ---------------------------------------------------------------------------
const TABLE_PROMPT = `Extract the table from this financial document image as HTML.

Rules:
- Use <table>, <tr>, <th>, <td> tags
- Use rowspan/colspan for merged cells
- Preserve exact text content (numbers, labels, units, dollar signs, percentages)
- Include ALL rows and columns — do not skip any data
- For multi-level headers, use <th> with rowspan/colspan
- Empty cells: <td></td>
- Extract ONLY the primary table — do not include surrounding text, captions, or footnotes as table rows
- Return ONLY the HTML table, no other text or markdown`;

// ---------------------------------------------------------------------------
// Re-extraction prompt when row count looks too low
// ---------------------------------------------------------------------------
function makeReExtractPrompt(extractedRows: number, extractedCols: number): string {
  return `Your previous extraction only produced ${extractedRows} rows and ${extractedCols} columns, which appears incomplete.

Look at the image again VERY carefully. Count every single row visible in the table, including:
- All header rows (column labels, grouped headers)
- All data rows (every line of numbers/text)
- Any total/subtotal rows
- Any footnote rows that are part of the table

Now extract the COMPLETE table as HTML with EVERY row. Do NOT skip, summarize, or truncate any rows.
Return ONLY the HTML table(s), no other text.`;
}

// ---------------------------------------------------------------------------
// OpenAI API call
// ---------------------------------------------------------------------------
async function callOpenAIVision(
  imageBase64: string,
  mime: string,
  systemPrompt: string,
  extraUserMessage?: string,
  previousAssistantHtml?: string
): Promise<string> {
  const messages: any[] = [
    {
      role: 'user',
      content: [
        { type: 'text', text: systemPrompt },
        {
          type: 'image_url',
          image_url: {
            url: `data:${mime};base64,${imageBase64}`,
            detail: 'high',
          },
        },
      ],
    },
  ];

  if (previousAssistantHtml && extraUserMessage) {
    messages.push({ role: 'assistant', content: previousAssistantHtml });
    messages.push({
      role: 'user',
      content: [
        { type: 'text', text: extraUserMessage },
        {
          type: 'image_url',
          image_url: {
            url: `data:${mime};base64,${imageBase64}`,
            detail: 'high',
          },
        },
      ],
    });
  }

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      temperature: 0,
      max_completion_tokens: 32768,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`OpenAI API error: ${res.status} ${JSON.stringify(err).substring(0, 300)}`);
  }

  const json = await res.json() as any;
  const content = json?.choices?.[0]?.message?.content || '';
  return content;
}

// ---------------------------------------------------------------------------
// HTML parsing helpers
// ---------------------------------------------------------------------------
function extractTableHtml(content: string): string {
  const tableMatches = content.match(/<table[\s\S]*?<\/table>/gi);
  if (tableMatches && tableMatches.length > 0) {
    return tableMatches.join('\n');
  }
  if (content.includes('<tr') || content.includes('<td')) {
    return `<table>${content}</table>`;
  }
  return content;
}

function countHtmlRows(html: string): number {
  const matches = html.match(/<tr[\s>]/gi);
  return matches ? matches.length : 0;
}

function countHtmlCols(html: string): number {
  const firstRow = html.match(/<tr[^>]*>([\s\S]*?)<\/tr>/i);
  if (!firstRow) return 0;
  const cells = firstRow[1].match(/<t[dh][^>]*>/gi);
  if (!cells) return 0;
  let cols = 0;
  for (const cell of cells) {
    const colspanMatch = cell.match(/colspan=["']?(\d+)["']?/i);
    cols += colspanMatch ? parseInt(colspanMatch[1]) : 1;
  }
  return cols;
}

function isValidPrediction(predictionsDir: string, id: string): boolean {
  const predPath = path.join(predictionsDir, `${id}.html`);
  if (!fs.existsSync(predPath)) return false;
  const content = fs.readFileSync(predPath, 'utf-8');
  return content.length > 20 && !content.startsWith('<!-- empty') && !content.startsWith('<!-- error');
}

// ---------------------------------------------------------------------------
// Multi-pass extraction with verification
// ---------------------------------------------------------------------------
const MIN_ROW_THRESHOLD = 3;
const ROW_IMPROVEMENT_RATIO = 1.3;

async function extractWithMultiPass(
  imagePath: string,
  retries: number
): Promise<string> {
  const buffer = fs.readFileSync(imagePath);
  const base64 = buffer.toString('base64');
  const ext = path.extname(imagePath).toLowerCase();
  const mime = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png';

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      // Pass 1: initial extraction
      const rawContent = await callOpenAIVision(base64, mime, TABLE_PROMPT);
      const html = extractTableHtml(rawContent);
      const rowCount = countHtmlRows(html);
      const colCount = countHtmlCols(html);

      if (rowCount < MIN_ROW_THRESHOLD || html.length < 30) {
        // Too few rows — likely a failed extraction, retry from scratch
        if (attempt < retries) {
          await new Promise((r) => setTimeout(r, 2000));
          continue;
        }
        return html;
      }

      // Pass 2: verification re-extraction for tables that look incomplete
      // Heuristic: if the raw response was cut short or seems partial, re-extract
      const rawTruncated = rawContent.endsWith('...') ||
        rawContent.length > 1000 && !rawContent.trimEnd().endsWith('>');

      if (rawTruncated && rowCount >= MIN_ROW_THRESHOLD) {
        const reExtractPrompt = makeReExtractPrompt(rowCount, colCount);
        const pass2Content = await callOpenAIVision(
          base64, mime, TABLE_PROMPT, reExtractPrompt, html
        );
        const pass2Html = extractTableHtml(pass2Content);
        const pass2Rows = countHtmlRows(pass2Html);

        if (pass2Rows > rowCount * ROW_IMPROVEMENT_RATIO && pass2Html.length > html.length) {
          return pass2Html;
        }
      }

      return html;
    } catch (err: any) {
      lastError = err;
      if (err?.message?.includes('429') || err?.message?.includes('rate')) {
        const delay = Math.min(5000 * Math.pow(2, attempt), 60000);
        console.error(`  Rate limited, waiting ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
      } else if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
      }
    }
  }
  throw lastError;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const config = parseArgs();

  if (!OPENAI_API_KEY) {
    console.error('Error: OPENAI_API_KEY is required');
    process.exit(1);
  }

  const manifestPath = path.join(config.dataDir, 'manifest.json');
  const manifest: ManifestEntry[] = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  const samples = manifest.slice(0, config.limit);

  fs.mkdirSync(config.predictionsDir, { recursive: true });

  let toProcess = samples;
  if (config.skipExisting) {
    toProcess = samples.filter(
      (s) => !isValidPrediction(config.predictionsDir, s.id)
    );
  }

  console.log('FinTabNet Direct Extraction (v2)');
  console.log('================================');
  console.log(` Model:       ${MODEL}`);
  console.log(` Samples:     ${samples.length} total, ${toProcess.length} to process`);
  console.log(` Concurrency: ${config.concurrency}`);
  console.log(` Skip valid:  ${config.skipExisting}`);
  console.log(` Multi-pass:  enabled`);
  console.log();

  const stats = { completed: 0, errors: 0, reExtracted: 0, total: toProcess.length };
  const startTime = Date.now();

  for (let i = 0; i < toProcess.length; i += config.concurrency) {
    const batch = toProcess.slice(i, i + config.concurrency);
    const promises = batch.map(async (entry) => {
      const imagePath = path.join(config.dataDir, entry.imagePath);
      const outPath = path.join(config.predictionsDir, `${entry.id}.html`);

      try {
        const html = await extractWithMultiPass(imagePath, config.retries);
        if (html && html.length > 10) {
          fs.writeFileSync(outPath, html, 'utf-8');
          stats.completed++;
        } else {
          fs.writeFileSync(outPath, '<!-- empty -->', 'utf-8');
          stats.errors++;
        }
      } catch (err: any) {
        console.error(`  [${entry.id}] ${err?.message?.substring(0, 120) || err}`);
        fs.writeFileSync(
          outPath,
          `<!-- error: ${String(err?.message || err).replace(/--/g, '-').substring(0, 200)} -->`,
          'utf-8'
        );
        stats.errors++;
      }
    });

    await Promise.all(promises);

    const done = Math.min(i + config.concurrency, toProcess.length);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const rate = (done / parseFloat(elapsed)).toFixed(2);
    const eta = done > 0
      ? ((toProcess.length - done) / parseFloat(rate)).toFixed(0)
      : '?';
    console.log(
      `  [${done}/${toProcess.length}] ${elapsed}s | ` +
      `ok=${stats.completed} err=${stats.errors} | ${rate}/s | ETA: ${eta}s`
    );
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log('\n================================');
  console.log(` Done in ${elapsed}s`);
  console.log(` Completed: ${stats.completed}`);
  console.log(` Errors: ${stats.errors}`);
  console.log(` Predictions: ${config.predictionsDir}`);
}

main().catch((err) => {
  console.error('Extraction failed:', err);
  process.exit(1);
});
