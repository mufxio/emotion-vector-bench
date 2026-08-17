"""
Validate emotion vector geometry: cosine clustering, PCA valence axis,
cross-layer stability. Saves heatmap + PCA scatter plots.

Usage:
    python validate.py --model qwen2.5-7b-instruct
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

# Anthropic-style cluster mapping (from emotions.json)
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


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def check_cosine_matrix(vectors):
    emotions = list(vectors.keys())
    n = len(emotions)
    sim = np.zeros((n, n))
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            sim[i, j] = cosine(vectors[e1], vectors[e2])
    within, cross = [], []
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            if i >= j:
                continue
            (within if EMOTION_TO_CLUSTER.get(e1) == EMOTION_TO_CLUSTER.get(e2) else cross).append(sim[i, j])
    return {
        "matrix": sim,
        "emotions": emotions,
        "within_cluster_mean": float(np.mean(within)),
        "cross_cluster_mean": float(np.mean(cross)),
        "diff": float(np.mean(within) - np.mean(cross)),
        "passed": np.mean(within) > np.mean(cross) + 0.05,
    }


def check_pca(vectors):
    """PC1 valence separation, measured scale-free.

    CORRECTION 2026-08-17: this previously ran PCA on the raw mean-difference
    vectors and reported the PC1 separation in those raw units. Because activation
    magnitude varies by an order of magnitude across model families, that number was
    dominated by vector scale rather than by geometry -- across the five models in
    results/, raw separation correlated with mean vector L2 norm at r = 0.9896
    (R^2 = 0.979). The resulting "18x spread in valence-axis strength" was an
    artifact of units, not a finding; normalized, the same spread is 1.40x.

    The vectors are now unit-normalized before PCA, so separation is measured
    between directions. `valence_separation_pc1_raw` is retained for
    backward comparison but must not be compared across models.
    """
    emotions = list(vectors.keys())
    M_raw = np.stack([vectors[e] for e in emotions], axis=0)

    # Unit-normalize each emotion vector: compare directions, not magnitudes.
    norms = np.linalg.norm(M_raw, axis=1, keepdims=True)
    M = M_raw / np.clip(norms, 1e-12, None)

    pca = PCA(n_components=min(5, len(emotions)))
    coords = pca.fit_transform(M)

    def _sep(c):
        pos = [c[emotions.index(e), 0] for e in CLUSTERS["Positive Anchors"] if e in emotions]
        neg_em = [e for cl, es in CLUSTERS.items() for e in es if cl != "Positive Anchors"]
        neg = [c[emotions.index(e), 0] for e in neg_em if e in emotions]
        return float(abs(np.mean(pos) - np.mean(neg))), float(np.mean(pos)), float(np.mean(neg))

    sep, pos_mean, neg_mean = _sep(coords)
    raw_sep, _, _ = _sep(pca.fit_transform(M_raw))

    # Scale-free effect size: separation in units of PC1 spread. Comparable
    # across models regardless of activation magnitude.
    pc1_spread = float(np.std(coords[:, 0])) or 1e-12
    sep_d = sep / pc1_spread

    return {
        "pc_explained_variance": pca.explained_variance_ratio_.tolist(),
        "coords_pc1_pc2": coords[:, :2].tolist(),
        "emotions": emotions,
        "pos_mean_pc1": pos_mean,
        "neg_mean_pc1": neg_mean,
        "valence_separation_pc1": sep,          # unit-normalized -- cross-model comparable
        "valence_separation_pc1_raw": raw_sep,  # legacy, NOT comparable across models
        "valence_separation_pc1_effect": sep_d,
        "mean_vector_l2": float(np.mean(norms)),
        # Criterion is on the scale-free effect size: valence must separate by at
        # least one PC1 standard deviation. The old `sep > 1.0` bar was in raw
        # activation units and was therefore a different test for every model.
        "passed": sep_d > 1.0,
    }


def check_layer_stability(vectors_per_layer):
    layer_keys = sorted(vectors_per_layer.keys())
    if len(layer_keys) < 2:
        return {"passed": False, "reason": "need ≥2 layers"}
    matrices = {lk: check_cosine_matrix(vectors_per_layer[lk])["matrix"] for lk in layer_keys}
    correlations = {}
    for i, l1 in enumerate(layer_keys):
        for j, l2 in enumerate(layer_keys):
            if i >= j:
                continue
            iu = np.triu_indices_from(matrices[l1], k=1)
            r = np.corrcoef(matrices[l1][iu], matrices[l2][iu])[0, 1]
            correlations[f"{l1}-{l2}"] = float(r)
    mean_corr = float(np.mean(list(correlations.values())))
    return {
        "pairwise_correlations": correlations,
        "mean_correlation": mean_corr,
        "passed": mean_corr > 0.7,
    }


def make_plots(vectors, cosine_info, pca_info, layer_key, out_dir):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cosine_info["matrix"], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cosine_info["emotions"])))
    ax.set_yticks(range(len(cosine_info["emotions"])))
    ax.set_xticklabels(cosine_info["emotions"], rotation=45, ha="right")
    ax.set_yticklabels(cosine_info["emotions"])
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Cosine similarity ({layer_key})")
    plt.tight_layout()
    plt.savefig(plots_dir / f"cosine_{layer_key}.png", dpi=120)
    plt.close()

    # PCA scatter
    coords = np.array(pca_info["coords_pc1_pc2"])
    fig, ax = plt.subplots(figsize=(10, 8))
    for e, (x, y) in zip(pca_info["emotions"], coords):
        c = CLUSTER_COLORS.get(EMOTION_TO_CLUSTER.get(e, ""), "black")
        ax.scatter(x, y, c=c, s=80)
        ax.annotate(e, (x, y), fontsize=9, ha="left", va="bottom")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    pcvar = pca_info["pc_explained_variance"]
    ax.set_xlabel(f"PC1 ({pcvar[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pcvar[1]:.1%} var)")
    ax.set_title(f"Emotion vectors PCA ({layer_key})")
    plt.tight_layout()
    plt.savefig(plots_dir / f"pca_{layer_key}.png", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model slug")
    ap.add_argument("--use", choices=["raw", "denoised"], default="denoised")
    args = ap.parse_args()

    model_dir = RESULTS_ROOT / args.model
    vec_path = model_dir / f"{args.use}_vectors.npz"
    if not vec_path.exists():
        vec_path = model_dir / "raw_vectors.npz"
        print(f"Falling back to raw vectors (no denoised available)")
    print(f"Using vectors from: {vec_path}")

    data = np.load(vec_path)
    layer_keys = sorted(set(k.split("__")[0] for k in data.files))
    vectors_per_layer = {lk: load_vectors_at_layer(vec_path, lk) for lk in layer_keys}

    results = {}
    for lk in layer_keys:
        print(f"\n=== {lk} ===")
        vecs = vectors_per_layer[lk]
        cos = check_cosine_matrix(vecs)
        pca = check_pca(vecs)
        results[lk] = {"cosine": cos, "pca": pca}
        print(f"  [Check 1] within={cos['within_cluster_mean']:.3f}, cross={cos['cross_cluster_mean']:.3f}, diff={cos['diff']:.3f} {'✓' if cos['passed'] else '✗'}")
        print(f"  [Check 2] PC1 valence sep={pca['valence_separation_pc1']:.2f} {'✓' if pca['passed'] else '✗'}")
        make_plots(vecs, cos, pca, lk, model_dir)

    print("\n=== Cross-layer stability ===")
    stab = check_layer_stability(vectors_per_layer)
    print(f"  Mean correlation: {stab.get('mean_correlation', 'N/A')} {'✓' if stab.get('passed') else '✗'}")

    print("\n" + "=" * 60)
    print("AUDIT GATE SUMMARY")
    print("=" * 60)
    for lk in layer_keys:
        r = results[lk]
        passes = sum([r["cosine"]["passed"], r["pca"]["passed"]])
        print(f"  {lk}: {passes}/2 (cos: {'✓' if r['cosine']['passed'] else '✗'}, pca: {'✓' if r['pca']['passed'] else '✗'})")
    print(f"  Cross-layer: {'✓' if stab.get('passed') else '✗'}")

    summary = {
        lk: {
            "cosine": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r["cosine"].items()},
            "pca": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r["pca"].items()},
        }
        for lk, r in results.items()
    }
    summary["cross_layer_stability"] = stab
    summary["model"] = args.model
    summary["vectors_used"] = args.use
    with open(model_dir / "validation_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {model_dir / 'validation_results.json'}")


if __name__ == "__main__":
    main()
