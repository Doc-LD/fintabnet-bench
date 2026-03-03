#!/usr/bin/env tsx
/**
 * Validate that the table scorer produces correct results.
 */

import { tableScore, htmlTableToArray } from './table-scorer.js';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

// Perfect match
const gt1 = [
  ['A', 'B', 'C'],
  ['1', '2', '3'],
];
assert(tableScore(gt1, gt1) === 1, 'Perfect match should be 1');

// Minor typo
const pred1 = [
  ['A', 'B', 'C'],
  ['1', '2', '3'],
];
assert(tableScore(gt1, pred1) === 1, 'Identical should be 1');

const pred2 = [
  ['A', 'B', 'C'],
  ['1', '2', '3x'],
];
const s2 = tableScore(gt1, pred2);
assert(s2 > 0.8 && s2 < 1, `Minor typo should get partial credit, got ${s2}`);

// HTML conversion
const html = '<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>';
const arr = htmlTableToArray(html);
assert(arr.length === 2, `Expected 2 rows, got ${arr.length}`);
assert(arr[0].length === 2, `Expected 2 cols, got ${arr[0].length}`);
assert(tableScore(arr, arr) === 1, 'HTML round-trip should score 1');

console.log('All scorer tests passed.');
