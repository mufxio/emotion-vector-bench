"""
Cross-model comparison synthesis.

Reads validation_results.json for all models in results/ and produces:
- A summary table (printed)
- A multi-panel plot showing PC1 valence separation, within/cross cluster diff, and cross-layer stability across models
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def load_results(model_dir: Path):
    rf = model_dir / "validation_results.json"
    if not rf.exists():
        return None
    with open(rf) as f:
        return json.load(f)


def main():
    models = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        r = load_results(d)
        if r is None:
            continue
        models.append((d.name, r))

    print("=" * 85)
    print("Cross-Model Comparison: Emotion Vector Geometry")
    print("=" * 85)
    print(f"\n{'Model':<28} | {'Layer':<10} | {'Within':>7} | {'Cross':>7} | {'Diff':>6} | {'PC1 sep':>9} | {'X-layer':>7}")
    print("-" * 95)

    # Aggregate per-model max-PC1 result
    summary = []
    for name, r in models:
        layer_keys = [k for k in r.keys() if k.startswith("layer_")]
        # Find the layer with the highest valence separation
        best_layer = max(layer_keys, key=lambda lk: r[lk]["pca"]["valence_separation_pc1"])
        c = r[best_layer]["cosine"]
        p = r[best_layer]["pca"]
        # Handle both old ("within_minus_cross") and new ("diff") field names
        diff = c.get("diff", c.get("within_minus_cross", c["within_cluster_mean"] - c["cross_cluster_mean"]))
        xl = r.get("cross_layer_stability", {}).get("mean_correlation", "?")
        xl_str = f"{xl:.3f}" if isinstance(xl, (int, float)) else xl
        print(f"{name:<28} | {best_layer:<10} | {c['within_cluster_mean']:>7.3f} | "
              f"{c['cross_cluster_mean']:>7.3f} | {diff:>6.3f} | "
              f"{p['valence_separation_pc1']:>9.2f} | {xl_str:>7}")
        summary.append({
            "model": name,
            "best_layer": best_layer,
            "within": c["within_cluster_mean"],
            "cross": c["cross_cluster_mean"],
            "diff": diff,
            "pc1_sep": p["valence_separation_pc1"],
            "xlayer": xl if isinstance(xl, (int, float)) else None,
        })

    # Multi-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = [s["model"] for s in summary]
    short_names = [n.replace("-instruct", "").replace("-v0.3", "") for n in model_names]

    # Panel 1: PC1 valence separation
    ax = axes[0]
    pc1_vals = [s["pc1_sep"] for s in summary]
    bars = ax.bar(short_names, pc1_vals, color=["#1f77b4" if "qwen" in n else "#d62728" if "llama" in n else "#2ca02c" for n in short_names])
    ax.axhline(1.0, color="gray", linestyle="--", label="Pass threshold")
    ax.set_ylabel("PC1 valence separation (higher = cleaner positive/negative axis)")
    ax.set_title("Valence axis strength per model")
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Within-cluster vs cross-cluster diff
    ax = axes[1]
    diff_vals = [s["diff"] for s in summary]
    ax.bar(short_names, diff_vals, color=["#1f77b4" if "qwen" in n else "#d62728" if "llama" in n else "#2ca02c" for n in short_names])
    ax.axhline(0.05, color="gray", linestyle="--", label="Pass threshold")
    ax.set_ylabel("Within-cluster − cross-cluster cosine diff")
    ax.set_title("Cluster cohesion per model")
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Cross-layer stability
    ax = axes[2]
    xl_vals = [s["xlayer"] for s in summary]
    ax.bar(short_names, xl_vals, color=["#1f77b4" if "qwen" in n else "#d62728" if "llama" in n else "#2ca02c" for n in short_names])
    ax.axhline(0.7, color="gray", linestyle="--", label="Pass threshold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean correlation of cosine matrices across layers")
    ax.set_title("Cross-layer stability per model")
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = REPO_ROOT / "results" / "_comparison.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"\nSaved cross-model plot → {out}")

    # Save numeric summary
    with open(REPO_ROOT / "results" / "_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
