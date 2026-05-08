"""
Test whether the emotion vector geometry recovers the *arousal* dimension
(in addition to valence). Together they form the "affective circumplex"
from human psychology.

Methodology:
  - Tag each emotion with high/low arousal based on Russell's affective grid
  - Run PCA on emotion vectors (we already have these)
  - For each PC, compute how cleanly it separates high-arousal from low-arousal emotions
  - Find the PC that best captures arousal (typically PC2 in Anthropic's results)
  - Plot the affective circumplex: valence-PC vs arousal-PC

Usage:
    python arousal_axis.py --model qwen2.5-7b-instruct
    python arousal_axis.py --model all
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

# Russell's affective circumplex tags for our 20 emotions.
# High arousal = active, intense. Low arousal = passive, depleted.
# Mid emotions (which mix or are ambiguous) are excluded from the high/low test.
AROUSAL = {
    "high": ["furious", "panicked", "terrified", "distressed", "joyful"],
    "low":  ["weary", "depressed", "hopeless", "resigned", "content"],
    # 10 emotions not used in the high/low arousal contrast (still plotted):
    # desperate, grief-stricken, ashamed, lonely, anxious, resentful,
    # contemptuous, frustrated, paranoid, proud
}

# Valence tags (already used in our previous PC1 test)
VALENCE = {
    "positive": ["joyful", "content", "proud"],
    "negative": [
        "desperate", "grief-stricken", "ashamed", "lonely",
        "furious", "resentful", "contemptuous", "frustrated",
        "terrified", "anxious", "panicked", "distressed",
        "depressed", "weary", "hopeless", "resigned", "paranoid",
    ],
}

# Color by Anthropic cluster
CLUSTERS = {
    "Despair & Shame": ["desperate", "grief-stricken", "ashamed", "lonely"],
    "Hostile Anger": ["furious", "resentful", "contemptuous", "frustrated"],
    "Fear & Overwhelm": ["terrified", "anxious", "panicked", "distressed"],
    "Depleted Disengagement": ["depressed", "weary", "hopeless", "resigned"],
    "Vigilant Suspicion": ["paranoid"],
    "Positive Anchors": ["joyful", "content", "proud"],
}
EMOTION_TO_CLUSTER = {e: c for c, es in CLUSTERS.items() for e in es}
CLUSTER_COLORS = {
    "Despair & Shame": "#1f77b4",
    "Hostile Anger": "#d62728",
    "Fear & Overwhelm": "#ff7f0e",
    "Depleted Disengagement": "#7f7f7f",
    "Vigilant Suspicion": "#9467bd",
    "Positive Anchors": "#2ca02c",
}


def load_vectors_at_layer(path, layer_key):
    data = np.load(path)
    prefix = f"{layer_key}__"
    return {k.replace(prefix, ""): data[k] for k in data.files if k.startswith(prefix)}


def axis_separation(coords_1d, group_a_emotions, group_b_emotions, all_emotions):
    """How well does this 1D projection separate group A from group B?
    Returns absolute difference of means."""
    a_idx = [all_emotions.index(e) for e in group_a_emotions if e in all_emotions]
    b_idx = [all_emotions.index(e) for e in group_b_emotions if e in all_emotions]
    if not a_idx or not b_idx:
        return 0.0
    a_mean = np.mean([coords_1d[i] for i in a_idx])
    b_mean = np.mean([coords_1d[i] for i in b_idx])
    return float(abs(a_mean - b_mean)), float(a_mean), float(b_mean)


def analyze_model(model_slug):
    model_dir = RESULTS_ROOT / model_slug
    vec_path = model_dir / "denoised_vectors.npz"
    if not vec_path.exists():
        vec_path = model_dir / "raw_vectors.npz"
    if not vec_path.exists():
        return None

    print(f"\n=== {model_slug} ===")
    data = np.load(vec_path)
    layer_keys = sorted(set(k.split("__")[0] for k in data.files))

    results = {"model": model_slug, "layers": {}}

    for lk in layer_keys:
        vecs = load_vectors_at_layer(vec_path, lk)
        emotions = list(vecs.keys())
        M = np.stack([vecs[e] for e in emotions], axis=0)

        pca = PCA(n_components=min(5, len(emotions)))
        coords = pca.fit_transform(M)  # shape [20, 5]

        # For each PC, compute valence separation and arousal separation
        pc_analysis = []
        for pc_idx in range(coords.shape[1]):
            val_sep, val_pos_mean, val_neg_mean = axis_separation(
                coords[:, pc_idx], VALENCE["positive"], VALENCE["negative"], emotions
            )
            ar_sep, ar_high_mean, ar_low_mean = axis_separation(
                coords[:, pc_idx], AROUSAL["high"], AROUSAL["low"], emotions
            )
            pc_analysis.append({
                "pc": pc_idx + 1,
                "explained_variance": float(pca.explained_variance_ratio_[pc_idx]),
                "valence_sep": val_sep,
                "arousal_sep": ar_sep,
            })

        # Find which PCs encode valence and arousal best
        best_valence_pc = max(range(len(pc_analysis)), key=lambda i: pc_analysis[i]["valence_sep"])
        best_arousal_pc = max(range(len(pc_analysis)), key=lambda i: pc_analysis[i]["arousal_sep"])

        # If the same PC wins both (rare but possible), use 2nd-best for arousal
        if best_arousal_pc == best_valence_pc:
            sorted_pcs = sorted(range(len(pc_analysis)), key=lambda i: pc_analysis[i]["arousal_sep"], reverse=True)
            best_arousal_pc = sorted_pcs[1] if len(sorted_pcs) > 1 else best_valence_pc

        v_pc = best_valence_pc + 1
        a_pc = best_arousal_pc + 1
        v_sep = pc_analysis[best_valence_pc]["valence_sep"]
        a_sep = pc_analysis[best_arousal_pc]["arousal_sep"]
        v_var = pc_analysis[best_valence_pc]["explained_variance"]
        a_var = pc_analysis[best_arousal_pc]["explained_variance"]

        print(f"  {lk}: best valence on PC{v_pc} (sep {v_sep:.2f}, {v_var:.1%} var), "
              f"best arousal on PC{a_pc} (sep {a_sep:.2f}, {a_var:.1%} var)")

        results["layers"][lk] = {
            "pc_analysis": pc_analysis,
            "best_valence_pc": v_pc,
            "best_arousal_pc": a_pc,
            "valence_separation": v_sep,
            "arousal_separation": a_sep,
        }

        # Plot affective circumplex for this layer
        plots_dir = model_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        x_coords = coords[:, best_valence_pc]
        y_coords = coords[:, best_arousal_pc]
        for e, x, y in zip(emotions, x_coords, y_coords):
            cluster = EMOTION_TO_CLUSTER.get(e, "")
            color = CLUSTER_COLORS.get(cluster, "black")
            # Highlight high-arousal/low-arousal members
            edge = "black" if e in AROUSAL["high"] or e in AROUSAL["low"] else "none"
            ax.scatter(x, y, c=color, s=110, edgecolors=edge, linewidths=1.5)
            ax.annotate(e, (x, y), fontsize=9, ha="left", va="bottom")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel(f"Valence axis (PC{v_pc}, sep={v_sep:.2f}, {v_var:.1%} var)")
        ax.set_ylabel(f"Arousal axis (PC{a_pc}, sep={a_sep:.2f}, {a_var:.1%} var)")
        ax.set_title(f"Affective Circumplex — {model_slug} {lk}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"circumplex_{lk}.png", dpi=120)
        plt.close()

    out_path = model_dir / "arousal_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model slug, or 'all'")
    args = ap.parse_args()

    if args.model == "all":
        all_results = {}
        for d in sorted(RESULTS_ROOT.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            r = analyze_model(d.name)
            if r is not None:
                all_results[d.name] = r

        print("\n" + "=" * 95)
        print("CROSS-MODEL AFFECTIVE CIRCUMPLEX SUMMARY (best layer per model)")
        print("=" * 95)
        print(f"{'Model':<28} | {'Best layer':<10} | {'Valence on':<10} | {'Sep':>6} | {'Arousal on':<10} | {'Sep':>6}")
        print("-" * 95)
        for slug, r in all_results.items():
            best_layer = max(
                r["layers"].keys(),
                key=lambda lk: r["layers"][lk]["valence_separation"] + r["layers"][lk]["arousal_separation"]
            )
            ld = r["layers"][best_layer]
            print(f"{slug:<28} | {best_layer:<10} | PC{ld['best_valence_pc']:<8} | "
                  f"{ld['valence_separation']:>6.2f} | PC{ld['best_arousal_pc']:<8} | "
                  f"{ld['arousal_separation']:>6.2f}")
    else:
        analyze_model(args.model)


if __name__ == "__main__":
    main()
