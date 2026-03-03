# DocLD-FinTabNet Benchmark

Evaluation toolkit for benchmarking table extraction accuracy on the [FinTabNet](https://paperswithcode.com/dataset/fintabnet) dataset — 112K+ financial tables from S&P 500 SEC filings.

This repo contains:

- A dataset loader that downloads [FinTabNet_OTSL](https://huggingface.co/datasets/docling-project/FinTabNet_OTSL) from HuggingFace and samples 1,000 tables
- A DocLD API provider for agentic table extraction
- The same Needleman-Wunsch table similarity scorer used in [RD-TableBench](https://github.com/Doc-LD/rd-tablebench)
- Chart generation for blog visualizations

## Quick Start

```bash
# Install dependencies
npm install

# Install Python deps for dataset (pip install datasets pillow)
pip install -r scripts/requirements.txt

# Download FinTabNet_OTSL (samples 1,000 from test split)
npm run download

# Run extraction (requires DOCLD_API_KEY and sufficient credits)
npm run extract

# Score predictions
npm run score

# Or run extract + score
npm run benchmark
```

## Configuration

Create `.env`:

```
DOCLD_API_KEY=docld_...
DOCLD_BASE_URL=https://api.docld.com   # optional
```

## Scoring Methodology

Uses the same Needleman-Wunsch hierarchical alignment as [RD-TableBench](https://github.com/reductoai/rd-tablebench):

- **Cell-level**: Levenshtein distance for partial credit on near-matches
- **Row-level**: Needleman-Wunsch with free end gaps
- **Parameters**: `S_ROW_MATCH=5`, `G_ROW=-3`, `S_CELL_MATCH=1`, `P_CELL_MISMATCH=-1`, `G_COL=-1`

## Dataset

[FinTabNet_OTSL](https://huggingface.co/datasets/docling-project/FinTabNet_OTSL) — FinTabNet in OTSL format with `image` and `html` (ground truth) per sample.

## References

- [DocLD-TableBench blog](https://docld.com/blog/docld-tablebench)
- [RD-TableBench](https://github.com/reductoai/rd-tablebench)
- [FinTabNet (GTE paper)](https://arxiv.org/abs/2005.00589)
- [Aligning benchmark datasets](https://arxiv.org/abs/2303.00716)

## Publishing to GitHub

To publish as Doc-LD/fintabnet-bench:

1. Create a new repo at https://github.com/organizations/Doc-LD/repositories
2. `cd fintabnet-bench && git init && git add . && git commit -m "Initial FinTabNet benchmark" && git remote add origin git@github.com:Doc-LD/fintabnet-bench.git && git push -u origin main`

## License

MIT
