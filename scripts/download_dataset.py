#!/usr/bin/env python3
"""
Download and prepare FinTabNet_OTSL dataset from HuggingFace.
Samples 1,000 tables from the test split and saves images + ground truth HTML.

Usage:
  python3 scripts/download_dataset.py
  python3 scripts/download_dataset.py --output ./benchmark/data --limit 1000

Requires: pip install datasets pillow
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark/data", help="Output directory")
    parser.add_argument("--limit", type=int, default=1000, help="Max samples to download")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Run: pip install datasets pillow")
        raise SystemExit(1)

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FinTabNet_OTSL from HuggingFace...")
    try:
        full_ds = load_dataset("docling-project/FinTabNet_OTSL")
        ds = full_ds["test"] if "test" in full_ds else full_ds[list(full_ds.keys())[0]]
    except Exception as e:
        print(f"Error: {e}")
        raise

    n = min(args.limit, len(ds))
    print(f"Sampling {n} tables from test split (total: {len(ds)})")

    manifest = []
    def cells_to_text(cell):
        if isinstance(cell, dict) and "tokens" in cell:
            return "".join(cell["tokens"]).strip()
        return ""

    def escape_html(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def rebuild_html(html_tokens, cells_flat):
        """Reconstruct HTML from structure tokens + cell content."""
        out = []
        cell_idx = 0
        i = 0
        while i < len(html_tokens):
            tok = html_tokens[i]
            if tok == "<td>" or tok == "<th>":
                out.append(tok)
                i += 1
                if i < len(html_tokens) and (html_tokens[i] == "</td>" or html_tokens[i] == "</th>"):
                    if cell_idx < len(cells_flat):
                        text = escape_html(cells_to_text(cells_flat[cell_idx]))
                        out.append(text)
                        cell_idx += 1
                    out.append(html_tokens[i])
                    i += 1
            else:
                out.append(tok)
                i += 1
        return "".join(out)

    for i in range(n):
        row = ds[i]
        sample_id = str(i)
        image = row.get("image")
        html_raw = row.get("html") or row.get("html_restored")
        cells = row.get("cells", [])

        if image is None:
            print(f"  Skip {i}: no image")
            continue
        if not html_raw and not cells:
            print(f"  Skip {i}: no html/cells")
            continue

        if isinstance(html_raw, list):
            # Flatten cells if nested (rows of cells)
            cells_flat = []
            for c in cells:
                if isinstance(c, list):
                    cells_flat.extend(c)
                else:
                    cells_flat.append(c)
            html = rebuild_html(html_raw, cells_flat)
            if html and not html.strip().startswith("<table"):
                html = f"<table>{html}</table>"
        else:
            html = html_raw
        if not html or len(html) < 10:
            print(f"  Skip {i}: html too short")
            continue

        image_path = f"images/{sample_id}.png"
        label_path = f"labels/{sample_id}.html"

        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(str(output_dir / image_path))

        with open(output_dir / label_path, "w", encoding="utf-8") as f:
            f.write(html)

        manifest.append({
            "id": sample_id,
            "imagePath": image_path,
            "labelPath": label_path,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written: {manifest_path}")
    print(f"  {len(manifest)} samples")

    if manifest:
        print(f"  Example: {manifest[0]['id']}")
        print(f"    image: {manifest[0]['imagePath']}")
        print(f"    label: {manifest[0]['labelPath']}")


if __name__ == "__main__":
    main()
