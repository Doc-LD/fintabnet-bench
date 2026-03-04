/**
 * Table similarity scorer — faithful port of RD-TableBench grading.py.
 * Same Needleman-Wunsch parameters for consistency with RD-TableBench.
 */

const S_ROW_MATCH = 5;
const G_ROW = -3;
const S_CELL_MATCH = 1;
const P_CELL_MISMATCH = -1;
const G_COL = -1;

function levenshteinDistance(a: string, b: string): number {
  const la = a.length;
  const lb = b.length;
  if (la === 0) return lb;
  if (lb === 0) return la;

  let prev = new Array(lb + 1);
  let curr = new Array(lb + 1);

  for (let j = 0; j <= lb; j++) prev[j] = j;

  for (let i = 1; i <= la; i++) {
    curr[0] = i;
    for (let j = 1; j <= lb; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[lb];
}

function cellMatchScore(cell1: string | null, cell2: string | null): number {
  if (cell1 === null || cell2 === null) return P_CELL_MISMATCH;
  if (cell1 === cell2) return S_CELL_MATCH;

  const dist = levenshteinDistance(cell1, cell2);
  const maxLen = Math.max(cell1.length, cell2.length);
  const normalizedDist = maxLen === 0 ? 0 : dist / maxLen;
  const similarity = 1 - normalizedDist;
  return P_CELL_MISMATCH + similarity * (S_CELL_MATCH - P_CELL_MISMATCH);
}

function needlemanWunschColumns(
  seq1: string[],
  seq2: string[],
  gapPenalty: number
): { score: number } {
  const m = seq1.length;
  const n = seq2.length;

  const score: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0)
  );

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const match = score[i - 1][j - 1] + cellMatchScore(seq1[i - 1], seq2[j - 1]);
      const del = score[i - 1][j] + gapPenalty;
      const ins = score[i][j - 1] + gapPenalty;
      score[i][j] = Math.max(match, del, ins);
    }
  }

  let best = score[m][n];
  for (let j = 0; j <= n; j++) best = Math.max(best, score[m][j]);
  for (let i = 0; i <= m; i++) best = Math.max(best, score[i][n]);

  return { score: best };
}

export function normalizeCell(cell: string): string {
  let s = cell.normalize('NFC');
  s = s.replace(/\n/g, ' ');
  // Parenthesized negatives: (42.3) → 42.3 (common financial notation)
  s = s.replace(/\((\d[\d,.]*)\)/g, '$1');
  // Strip currency symbols and percent signs
  s = s.replace(/[$€£¥%]/g, '');
  // Remove commas between digits (thousands separators: 1,234 → 1234)
  s = s.replace(/(\d),(\d)/g, '$1$2');
  // Remove all hyphens (consistent with RD-TableBench grading.py)
  s = s.replace(/-/g, '');
  // Collapse all whitespace
  s = s.replace(/\s+/g, '');
  return s.toLowerCase();
}

export function tableScore(
  groundTruth: string[][],
  prediction: string[][]
): number {
  if (groundTruth.length === 0 && prediction.length === 0) return 1;
  if (groundTruth.length === 0 || prediction.length === 0) return 0;

  const gtRows = groundTruth.map((row) => row.map(normalizeCell));
  const predRows = prediction.map((row) => row.map(normalizeCell));

  const m = gtRows.length;
  const n = predRows.length;

  const rowMatchScores: number[][] = Array.from({ length: m }, () =>
    new Array(n).fill(0)
  );
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const colResult = needlemanWunschColumns(gtRows[i], predRows[j], G_COL);
      rowMatchScores[i][j] = colResult.score + S_ROW_MATCH;
    }
  }

  const score: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0)
  );
  const traceback: (string | null)[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(null)
  );

  for (let i = 1; i <= m; i++) traceback[i][0] = 'up';
  for (let j = 1; j <= n; j++) traceback[0][j] = 'left';

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const match = score[i - 1][j - 1] + rowMatchScores[i - 1][j - 1];
      const del = score[i - 1][j] + G_ROW;
      const ins = score[i][j - 1] + G_ROW;
      const maxScore = Math.max(match, del, ins);
      score[i][j] = maxScore;
      if (maxScore === match) traceback[i][j] = 'diag';
      else if (maxScore === del) traceback[i][j] = 'up';
      else traceback[i][j] = 'left';
    }
  }

  let maxScore = score[m][n];
  let maxI = m;
  let maxJ = n;
  for (let i = 0; i <= m; i++) {
    if (score[i][n] > maxScore) {
      maxScore = score[i][n];
      maxI = i;
      maxJ = n;
    }
  }
  for (let j = 0; j <= n; j++) {
    if (score[m][j] > maxScore) {
      maxScore = score[m][j];
      maxI = m;
      maxJ = j;
    }
  }

  let numAligned = 0;
  let ti = maxI;
  let tj = maxJ;
  while (ti > 0 || tj > 0) {
    const dir = traceback[ti][tj];
    if (ti > 0 && tj > 0 && dir === 'diag') {
      numAligned++;
      ti--;
      tj--;
    } else if (ti > 0 && (tj === 0 || dir === 'up')) {
      numAligned++;
      ti--;
    } else if (tj > 0 && (ti === 0 || dir === 'left')) {
      numAligned++;
      tj--;
    } else {
      break;
    }
  }

  if (numAligned === 0) return 0;

  const gtColCount = gtRows[0].length;
  const maxRowScore = S_ROW_MATCH + gtColCount * S_CELL_MATCH;
  const totalPossible = numAligned * maxRowScore;
  if (totalPossible === 0) return 0;

  return Math.max(0, Math.min(1, maxScore / totalPossible));
}

export interface TableBenchResult {
  id: string;
  score: number;
  predictedRows: number;
  predictedCols: number;
  groundTruthRows: number;
  groundTruthCols: number;
  error?: string;
}

export interface BenchmarkSummary {
  totalSamples: number;
  scoredSamples: number;
  errorSamples: number;
  meanAccuracy: number;
  medianAccuracy: number;
  p25Accuracy: number;
  p75Accuracy: number;
  minAccuracy: number;
  maxAccuracy: number;
  results: TableBenchResult[];
}

export function summarizeBenchmark(results: TableBenchResult[]): BenchmarkSummary {
  const scored = results.filter((r) => !r.error);
  const errors = results.filter((r) => !!r.error);
  const scores = scored.map((r) => r.score).sort((a, b) => a - b);

  const percentile = (sorted: number[], p: number): number => {
    if (sorted.length === 0) return 0;
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  };

  return {
    totalSamples: results.length,
    scoredSamples: scored.length,
    errorSamples: errors.length,
    meanAccuracy:
      scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0,
    medianAccuracy: percentile(scores, 50),
    p25Accuracy: percentile(scores, 25),
    p75Accuracy: percentile(scores, 75),
    minAccuracy: scores.length > 0 ? scores[0] : 0,
    maxAccuracy: scores.length > 0 ? scores[scores.length - 1] : 0,
    results,
  };
}

export function htmlTableToArray(html: string): string[][] {
  const rows: string[][] = [];
  const spanTracker = new Map<string, string>();

  const rowMatches = html.match(/<tr[^>]*>[\s\S]*?<\/tr>/gi) || [];

  for (let rowIdx = 0; rowIdx < rowMatches.length; rowIdx++) {
    const currentRow: string[] = [];
    let colIdx = 0;

    while (spanTracker.has(`${rowIdx}-${colIdx}`)) {
      currentRow.push(spanTracker.get(`${rowIdx}-${colIdx}`)!);
      spanTracker.delete(`${rowIdx}-${colIdx}`);
      colIdx++;
    }

    const cellMatches =
      rowMatches[rowIdx].match(/<t[dh][^>]*>[\s\S]*?<\/t[dh]>/gi) || [];

    for (const cellHtml of cellMatches) {
      while (spanTracker.has(`${rowIdx}-${colIdx}`)) {
        currentRow.push(spanTracker.get(`${rowIdx}-${colIdx}`)!);
        spanTracker.delete(`${rowIdx}-${colIdx}`);
        colIdx++;
      }

      const contentMatch = cellHtml.match(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/i);
      const cellText = contentMatch
        ? contentMatch[1].replace(/<[^>]+>/g, '').trim()
        : '';

      const rowspanMatch = cellHtml.match(/rowspan=["']?(\d+)["']?/i);
      const colspanMatch = cellHtml.match(/colspan=["']?(\d+)["']?/i);
      const rowspan = rowspanMatch ? parseInt(rowspanMatch[1]) : 1;
      const colspan = colspanMatch ? parseInt(colspanMatch[1]) : 1;

      if (rowspan > 1) {
        for (let c = 0; c < colspan; c++) {
          for (let r = 1; r < rowspan; r++) {
            spanTracker.set(`${rowIdx + r}-${colIdx + c}`, cellText);
          }
        }
      }

      for (let c = 0; c < colspan; c++) {
        currentRow.push(cellText);
        colIdx++;
      }
    }

    while (spanTracker.has(`${rowIdx}-${colIdx}`)) {
      currentRow.push(spanTracker.get(`${rowIdx}-${colIdx}`)!);
      spanTracker.delete(`${rowIdx}-${colIdx}`);
      colIdx++;
    }

    rows.push(currentRow);
  }

  const maxCols = rows.reduce((max, row) => Math.max(max, row.length), 0);
  for (const row of rows) {
    while (row.length < maxCols) row.push('');
  }

  return rows;
}

export function markdownTableToArray(md: string): string[][] {
  const lines = md
    .trim()
    .split('\n')
    .filter((l) => l.trim());
  const result: string[][] = [];

  for (const line of lines) {
    if (/^\|[\s\-:]+\|$/.test(line.replace(/[^|\-:\s]/g, '').trim())) continue;
    const cells = line
      .split('|')
      .filter((_, i, arr) => i > 0 && i < arr.length - 1)
      .map((c) => c.trim());
    if (cells.length > 0) result.push(cells);
  }
  return result;
}
