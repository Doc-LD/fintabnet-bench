# DocLD-FinTabNet Benchmark

Evaluation toolkit for benchmarking **table extraction accuracy** on the [FinTabNet](https://paperswithcode.com/dataset/fintabnet) dataset — 112K+ financial tables from S&P 500 SEC filings. Use this repository to reproduce [DocLD](https://docld.com)’s FinTabNet results or to evaluate your own extraction pipeline on the same data and scoring methodology.

## Overview

[DocLD](https://docld.com) is a document intelligence API for table extraction, OCR, and structured data. This benchmark measures table extraction quality on **FinTabNet** — the standard benchmark for financial table extraction, with tables from S&P 500 annual reports that include multi-level headers, merged cells, and dense numeric data.

**Published results (March 2026):** DocLD achieves **82.1%** average table accuracy on 500 FinTabNet tables with **100% extraction success rate** (500/500), outperforming GTE (IBM) and TATR (Microsoft) under the same Needleman–Wunsch scoring. For details and comparisons, see the [DocLD-FinTabNet blog post](https://docld.com/blog/docld-fintabnet).

## What’s in this repository

- **Dataset loader** — Downloads [FinTabNet_OTSL](https://huggingface.co/datasets/docling-project/FinTabNet_OTSL) from HuggingFace and samples tables from the test split (default: 1,000; published results used 500).
- **DocLD extraction client** — Runs table extraction via the DocLD API and writes HTML predictions.
- **Scoring** — Same Needleman–Wunsch hierarchical alignment used in [RD-TableBench](https://github.com/Doc-LD/rd-tablebench) for cross-benchmark consistency.
- **Charts** — Scripts to generate visualizations for analysis or reporting.

## Benchmark results (DocLD)

| Metric   | Value   |
| -------- | ------- |
| Mean     | 82.1%   |
| Median   | 83.2%   |
| P90      | 99.7%   |
| Scored   | 500/500 |

Results and predictions are available on [HuggingFace](https://huggingface.co/datasets/docld/fintabnet-bench).

## Quick start

**Requirements:** Node.js, Python 3, and a [DocLD API key](https://docld.com) (for running extraction).

```bash
# Clone and install
git clone https://github.com/Doc-LD/fintabnet-bench.git
cd fintabnet-bench
npm install
pip install -r scripts/requirements.txt

# Download FinTabNet_OTSL (samples from test split)
npm run download

# Run extraction (uses DOCLD_API_KEY; requires API credits)
npm run extract

# Score predictions against ground truth
npm run score
```

To run extraction and scoring in one step: `npm run benchmark`.

## Configuration

Create a `.env` file in the project root:

```env
DOCLD_API_KEY=docld_...
DOCLD_BASE_URL=https://api.docld.com   # optional, for custom endpoint
```

## Scoring methodology

Scoring matches [RD-TableBench](https://github.com/Doc-LD/rd-tablebench):

- **Cell-level:** Levenshtein distance for partial credit on near-matches.
- **Row-level:** Needleman–Wunsch alignment with free end gaps.
- **Parameters:** `S_ROW_MATCH=5`, `G_ROW=-3`, `S_CELL_MATCH=1`, `P_CELL_MISMATCH=-1`, `G_COL=-1`.

Cell text is normalized (Unicode NFC, case-insensitive, whitespace/currency/thousands separators). Ground truth and predictions are expanded from HTML (rowspan/colspan) to 2D arrays before alignment.

## Dataset

[FinTabNet_OTSL](https://huggingface.co/datasets/docling-project/FinTabNet_OTSL) — FinTabNet in OTSL format. Each sample has an `image` (table crop) and `html` (ground truth). The loader uses the test split and can limit the number of samples (e.g. 500 for the published run, 1000 by default).

## References

| Resource | Link |
| -------- | ---- |
| **DocLD-FinTabNet blog** | [docld.com/blog/docld-fintabnet](https://docld.com/blog/docld-fintabnet) |
| **Results & predictions** | [HuggingFace — docld/fintabnet-bench](https://huggingface.co/datasets/docld/fintabnet-bench) |
| DocLD-TableBench (RD-TableBench) | [docld.com/blog/docld-tablebench](https://docld.com/blog/docld-tablebench) |
| RD-TableBench scoring | [github.com/Doc-LD/rd-tablebench](https://github.com/Doc-LD/rd-tablebench) |
| FinTabNet (GTE paper) | [arXiv:2005.00589](https://arxiv.org/abs/2005.00589) |
| Aligning benchmark datasets (TATR) | [arXiv:2303.00716](https://arxiv.org/abs/2303.00716) |

## License

MIT
