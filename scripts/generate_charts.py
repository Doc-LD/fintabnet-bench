#!/usr/bin/env python3
"""
Generate rich benchmark visualization PNGs for FinTabNet blog post.
Produces: bar chart, score distribution, histogram, percentile comparison,
          and a combined dashboard.
"""

import json
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("Run: pip install matplotlib numpy")
    raise SystemExit(1)

RESULTS_PATH = Path("benchmark/results/results.json")
OUTPUT_DIR = Path("benchmark/charts")
BLOG_IMAGES = Path(__file__).resolve().parent.parent.parent / "public/blog/images/docld-fintabnet"

# DocLD brand palette
BRAND_GOLD = "#c4956a"
BRAND_LIGHT = "#f5f3ef"
BRAND_DARK = "#4a4540"
BRAND_MID = "#6b6560"
BRAND_MUTED = "#c4c0b8"
BRAND_BG = "#faf9f7"

# RD-TableBench competitor scores (from our published blog)
RD_TABLEBENCH_SCORES = {
    "DocLD": 92.4,
    "Reducto": 90.2,
    "Azure DI": 82.7,
    "Textract": 80.9,
    "Sonnet 3.5": 80.7,
    "GPT-4o": 76.0,
    "LlamaParse": 74.6,
    "GCloud DI": 64.6,
    "Unstructured": 60.2,
}

# IDP Leaderboard (Nanonets, table extraction)
IDP_SCORES = {
    "Claude Sonnet 4": 93.44,
    "Claude 3.7 Sonnet": 91.23,
    "Gemini 2.5 Pro": 79.51,
    "GPT-4.1": 74.34,
    "Gemini 2.0 Flash": 71.32,
    "GPT-4o": 64.30,
}


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 600,
        'axes.labelsize': 12,
        'axes.facecolor': BRAND_BG,
        'figure.facecolor': BRAND_BG,
        'axes.edgecolor': '#d6d3cd',
        'axes.grid': True,
        'grid.color': '#e8e5e0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
    })


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def chart_1_overall_comparison(data, docld_score):
    """Bar chart: DocLD vs competitors on FinTabNet."""
    providers = ["DocLD", "GTE (IBM)", "TATR (Microsoft)"]
    scores = [docld_score, 78.0, 65.0]
    colors = [BRAND_GOLD, BRAND_MUTED, BRAND_MUTED]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(providers, scores, color=colors, edgecolor="none", width=0.55, zorder=3)
    ax.set_ylabel("Accuracy (%)", fontweight=500, color=BRAND_DARK)
    ax.set_ylim(50, 105)
    ax.set_title("FinTabNet — Average Table Accuracy", fontweight=700, color=BRAND_DARK, pad=16)
    ax.tick_params(axis='x', labelsize=12, colors=BRAND_DARK)
    ax.tick_params(axis='y', labelsize=10, colors=BRAND_MID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{s:.1f}%", ha="center", va="bottom", fontsize=13, fontweight=700,
                color=BRAND_DARK)
    plt.tight_layout()
    return fig


def chart_2_score_distribution(data):
    """Box + violin plot of per-sample scores."""
    results = data.get("results", [])
    scores = [r["score"] * 100 for r in results if not r.get("error")]
    if not scores:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    vp = ax.violinplot([scores], positions=[1], showmeans=True, showmedians=True)
    for body in vp['bodies']:
        body.set_facecolor(BRAND_GOLD)
        body.set_alpha(0.3)
        body.set_edgecolor(BRAND_DARK)
    vp['cmeans'].set_color(BRAND_GOLD)
    vp['cmeans'].set_linewidth(2)
    vp['cmedians'].set_color(BRAND_DARK)
    vp['cmedians'].set_linewidth(2)
    for key in ['cbars', 'cmaxes', 'cmins']:
        vp[key].set_color(BRAND_MID)
        vp[key].set_linewidth(1)

    bp = ax.boxplot([scores], positions=[1], widths=0.15, patch_artist=True,
                    showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
    bp['boxes'][0].set_facecolor(BRAND_LIGHT)
    bp['boxes'][0].set_edgecolor(BRAND_DARK)
    bp['medians'][0].set_color(BRAND_GOLD)
    bp['medians'][0].set_linewidth(2)

    ax.set_ylabel("Score (%)", fontweight=500, color=BRAND_DARK)
    ax.set_xticklabels(["DocLD"])
    ax.set_title("Per-Sample Score Distribution (n=" + str(len(scores)) + ")",
                 fontweight=700, color=BRAND_DARK, pad=16)
    ax.set_ylim(0, 108)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mean = np.mean(scores)
    median = np.median(scores)
    ax.annotate(f"Mean: {mean:.1f}%\nMedian: {median:.1f}%",
                xy=(1.3, mean), fontsize=10, color=BRAND_DARK,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=BRAND_LIGHT, edgecolor=BRAND_MUTED))

    plt.tight_layout()
    return fig


def chart_3_histogram(data):
    """Score distribution histogram with KDE-like smoothing."""
    results = data.get("results", [])
    scores = [r["score"] * 100 for r in results if not r.get("error")]
    if not scores:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.arange(0, 105, 5)
    n, _, patches = ax.hist(scores, bins=bins, color=BRAND_GOLD, edgecolor=BRAND_LIGHT,
                           alpha=0.85, zorder=3)
    for patch in patches:
        patch.set_linewidth(0.5)

    mean = np.mean(scores)
    ax.axvline(mean, color=BRAND_DARK, linestyle='--', linewidth=2, zorder=4,
               label=f'Mean: {mean:.1f}%')

    p90 = np.percentile(scores, 90)
    ax.axvline(p90, color=BRAND_MID, linestyle=':', linewidth=1.5, zorder=4,
               label=f'P90: {p90:.1f}%')

    ax.set_xlabel("Score (%)", fontweight=500, color=BRAND_DARK)
    ax.set_ylabel("Number of Samples", fontweight=500, color=BRAND_DARK)
    ax.set_title("Score Distribution — DocLD on FinTabNet",
                 fontweight=700, color=BRAND_DARK, pad=16)
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def chart_4_cross_benchmark(data, docld_fintabnet_score):
    """Cross-benchmark comparison: DocLD across RD-TableBench and FinTabNet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: RD-TableBench
    providers_rd = list(RD_TABLEBENCH_SCORES.keys())
    scores_rd = list(RD_TABLEBENCH_SCORES.values())
    colors_rd = [BRAND_GOLD if p == "DocLD" else BRAND_MUTED for p in providers_rd]

    y_pos = np.arange(len(providers_rd))
    ax1.barh(y_pos, scores_rd, color=colors_rd, edgecolor="none", height=0.6, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(providers_rd, fontsize=10)
    ax1.set_xlabel("Accuracy (%)", fontweight=500, color=BRAND_DARK)
    ax1.set_title("RD-TableBench", fontweight=700, color=BRAND_DARK, pad=12)
    ax1.set_xlim(50, 100)
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for i, s in enumerate(scores_rd):
        ax1.text(s + 0.5, i, f"{s:.1f}%", va='center', fontsize=9,
                fontweight=600 if providers_rd[i] == "DocLD" else 400,
                color=BRAND_DARK)

    # Right: FinTabNet
    providers_ft = ["DocLD", "GTE (IBM)", "TATR (Microsoft)"]
    scores_ft = [docld_fintabnet_score, 78.0, 65.0]
    colors_ft = [BRAND_GOLD, BRAND_MUTED, BRAND_MUTED]

    y_pos2 = np.arange(len(providers_ft))
    ax2.barh(y_pos2, scores_ft, color=colors_ft, edgecolor="none", height=0.6, zorder=3)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(providers_ft, fontsize=10)
    ax2.set_xlabel("Accuracy (%)", fontweight=500, color=BRAND_DARK)
    ax2.set_title("FinTabNet", fontweight=700, color=BRAND_DARK, pad=12)
    ax2.set_xlim(50, 100)
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for i, s in enumerate(scores_ft):
        ax2.text(s + 0.5, i, f"{s:.1f}%", va='center', fontsize=9,
                fontweight=600 if providers_ft[i] == "DocLD" else 400,
                color=BRAND_DARK)

    fig.suptitle("DocLD — Leading Table Extraction Across Benchmarks",
                 fontsize=15, fontweight=700, color=BRAND_DARK, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def chart_5_percentile_breakdown(data, docld_score):
    """Summary stats as a clean infographic-style chart."""
    results = data.get("results", [])
    scores = [r["score"] * 100 for r in results if not r.get("error")]
    if not scores:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    stats = {
        'Mean': np.mean(scores),
        'Median': np.median(scores),
        'P25': np.percentile(scores, 25),
        'P75': np.percentile(scores, 75),
        'P90': np.percentile(scores, 90),
        'Min': np.min(scores),
        'Max': np.max(scores),
    }

    positions = np.linspace(5, 95, len(stats))
    for i, (label, val) in enumerate(stats.items()):
        x = positions[i]
        color = BRAND_GOLD if label in ('Mean', 'Median') else BRAND_MID
        ax.text(x, 1.0, f"{val:.1f}%", ha='center', va='center',
                fontsize=16, fontweight=700, color=color)
        ax.text(x, 0.3, label, ha='center', va='center',
                fontsize=10, fontweight=500, color=BRAND_DARK)

    n = len(scores)
    errs = len([r for r in results if r.get("error")])
    ax.text(50, -0.3, f"n={n} scored samples  •  {errs} errors  •  {len(results)} total",
            ha='center', va='center', fontsize=9, color=BRAND_MID)

    fig.suptitle("DocLD FinTabNet — Performance Summary",
                 fontsize=14, fontweight=700, color=BRAND_DARK, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def chart_6_idp_comparison(docld_fintabnet_score):
    """Compare DocLD against IDP Leaderboard providers."""
    all_scores = {"DocLD (FinTabNet)": docld_fintabnet_score}
    all_scores.update(IDP_SCORES)

    providers = list(all_scores.keys())
    scores = list(all_scores.values())
    colors = [BRAND_GOLD if "DocLD" in p else BRAND_MUTED for p in providers]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(providers))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor="none", height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(providers, fontsize=10)
    ax.set_xlabel("Accuracy (%)", fontweight=500, color=BRAND_DARK)
    ax.set_title("Table Extraction — DocLD vs. Foundation Models\n(IDP Leaderboard + FinTabNet)",
                 fontweight=700, color=BRAND_DARK, pad=12)
    ax.set_xlim(50, 100)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, s in enumerate(scores):
        ax.text(s + 0.3, i, f"{s:.1f}%", va='center', fontsize=9,
                fontweight=600 if "DocLD" in providers[i] else 400,
                color=BRAND_DARK)
    plt.tight_layout()
    return fig


def main():
    if not RESULTS_PATH.exists():
        print(f"Results not found at {RESULTS_PATH}. Run: npm run score")
        raise SystemExit(1)

    setup_style()
    data = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BLOG_IMAGES.mkdir(parents=True, exist_ok=True)

    docld_score = data["meanAccuracy"] * 100
    print(f"DocLD FinTabNet score: {docld_score:.1f}%")

    charts = [
        ("overall-comparison.png", chart_1_overall_comparison(data, docld_score)),
        ("score-distribution.png", chart_2_score_distribution(data)),
        ("score-histogram.png", chart_3_histogram(data)),
        ("cross-benchmark.png", chart_4_cross_benchmark(data, docld_score)),
        ("performance-summary.png", chart_5_percentile_breakdown(data, docld_score)),
        ("idp-comparison.png", chart_6_idp_comparison(docld_score)),
    ]

    for name, fig in charts:
        if fig is None:
            print(f"Skipped {name} (no data)")
            continue
        out = OUTPUT_DIR / name
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BRAND_BG)
        plt.close(fig)
        print(f"Saved {out}")

        # Copy to blog images
        import shutil
        dest = BLOG_IMAGES / name
        shutil.copy(out, dest)
        print(f"  → {dest}")


if __name__ == "__main__":
    main()
