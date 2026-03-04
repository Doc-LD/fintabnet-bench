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
- **Extraction**: DocLD vision-based table extraction
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

    # Stage everything in a temp folder, then upload in a single commit
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="hf_fintabnet_"))
    try:
        (staging / "README.md").write_text(readme)
        shutil.copy2(RESULTS_PATH, staging / "results.json")

        if PREDICTIONS_DIR.exists():
            pred_out = staging / "predictions"
            pred_out.mkdir()
            pred_files = list(PREDICTIONS_DIR.glob("*.html"))
            valid = [f for f in pred_files if not f.read_text().startswith("<!-- e")]
            print(f"Staging {len(valid)} predictions...")
            for f in valid:
                shutil.copy2(f, pred_out / f.name)

        if CHARTS_DIR.exists():
            chart_out = staging / "charts"
            chart_out.mkdir()
            for f in CHARTS_DIR.glob("*.png"):
                shutil.copy2(f, chart_out / f.name)

        import time
        import re

        print("Uploading all files in a single commit...")
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                api.upload_folder(
                    folder_path=str(staging),
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    commit_message="Update benchmark results (82.2% on 451 samples)",
                )
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg and attempt < max_retries:
                    wait = 120
                    match = re.search(r"Retry after (\d+) seconds", err_msg)
                    if match:
                        wait = int(match.group(1)) + 10
                    print(f"Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
