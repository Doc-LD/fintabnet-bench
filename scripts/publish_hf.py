#!/usr/bin/env python3
"""
Publish FinTabNet benchmark results and predictions to HuggingFace.
Creates a dataset with images, ground truth, predictions, and scores.

Usage:
  python3 scripts/publish_hf.py
  HF_TOKEN=hf_xxx python3 scripts/publish_hf.py
"""

import json
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Run: pip install huggingface_hub")
    raise SystemExit(1)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "docld/fintabnet-bench"
RESULTS_PATH = Path("benchmark/results/results.json")
DATA_DIR = Path("benchmark/data")
PREDICTIONS_DIR = Path("benchmark/predictions")
CHARTS_DIR = Path("benchmark/charts")


def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN env var required")
        raise SystemExit(1)

    if not RESULTS_PATH.exists():
        print(f"Results not found at {RESULTS_PATH}")
        raise SystemExit(1)

    api = HfApi(token=HF_TOKEN)

    print(f"Creating/updating HuggingFace dataset: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            exist_ok=True,
            private=False,
        )
    except Exception as e:
        print(f"Repo creation note: {e}")

    results = json.load(open(RESULTS_PATH))
    scored = [r for r in results.get("results", []) if not r.get("error")]

    # Create dataset card
    readme = f"""---
license: mit
task_categories:
  - table-question-answering
  - image-to-text
tags:
  - table-extraction
  - benchmark
  - fintabnet
  - document-ai
  - docld
pretty_name: DocLD FinTabNet Benchmark
size_categories:
  - n<1K
---

# DocLD FinTabNet Benchmark Results

Benchmark results for [DocLD](https://docld.com) table extraction on the
[FinTabNet](https://paperswithcode.com/dataset/fintabnet) dataset.

## Results Summary

| Metric | Value |
|--------|-------|
| **Mean Accuracy** | {results['meanAccuracy']*100:.1f}% |
| **Median** | {results['medianAccuracy']*100:.1f}% |
| **P25 / P75** | {results['p25Accuracy']*100:.1f}% / {results['p75Accuracy']*100:.1f}% |
| **Min / Max** | {results['minAccuracy']*100:.1f}% / {results['maxAccuracy']*100:.1f}% |
| **Scored Samples** | {results['scoredSamples']} |
| **Total Samples** | {results['totalSamples']} |

## Methodology

- **Dataset**: [FinTabNet_OTSL](https://huggingface.co/datasets/docling-project/FinTabNet_OTSL) — {results['totalSamples']} samples from the test split
- **Extraction**: DocLD agentic table extraction (VLM-based, gpt-5-mini)
- **Scoring**: Needleman-Wunsch hierarchical alignment (same as [RD-TableBench](https://github.com/reductoai/rd-tablebench))
- **Output**: HTML tables with rowspan/colspan for merged cells

## Comparison

| Provider | FinTabNet Accuracy |
|----------|-------------------|
| **DocLD** | **{results['meanAccuracy']*100:.1f}%** |
| GTE (IBM) | ~78% |
| TATR (Microsoft) | ~65% |

## Files

- `results.json` — Full benchmark results with per-sample scores
- `predictions/` — HTML predictions for each sample
- `charts/` — Visualization PNGs

## Links

- [DocLD](https://docld.com)
- [Blog Post](https://docld.com/blog/docld-fintabnet)
- [Benchmark Code](https://github.com/Doc-LD/fintabnet-bench)
- [RD-TableBench Results](https://docld.com/blog/docld-tablebench)
"""

    readme_path = Path("/tmp/hf_readme.md")
    readme_path.write_text(readme)

    # Upload files
    print("Uploading README...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    print("Uploading results.json...")
    api.upload_file(
        path_or_fileobj=str(RESULTS_PATH),
        path_in_repo="results.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    # Upload predictions
    if PREDICTIONS_DIR.exists():
        pred_files = list(PREDICTIONS_DIR.glob("*.html"))
        valid = [f for f in pred_files if not f.read_text().startswith("<!-- e")]
        print(f"Uploading {len(valid)} predictions...")
        for f in valid:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"predictions/{f.name}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )

    # Upload charts
    if CHARTS_DIR.exists():
        for f in CHARTS_DIR.glob("*.png"):
            print(f"Uploading chart: {f.name}")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"charts/{f.name}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
