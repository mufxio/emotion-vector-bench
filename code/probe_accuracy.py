"""
Linear probe accuracy + bootstrap confidence intervals for emotion-vector validation.

For each layer in each model:
  - Train a logistic regression classifier on activations -> emotion labels
  - Evaluate on held-out test set with multiple train/test splits
  - Bootstrap CI for accuracy and within/cross cosine diff

This is the canonical comparison metric in the interpretability literature.
Anthropic reports 60-83% on a 15-way classification (chance = 6.7%).

Usage:
    python probe_accuracy.py --model qwen2.5-7b-instruct
    python probe_accuracy.py --model all   # run on all models
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

# Anthropic-style cluster mapping (for cluster-aware analysis)
CLUSTERS = {
    "Despair & Shame": ["desperate", "grief-stricken", "ashamed", "lonely"],
    "Hostile Anger": ["furious", "resentful", "contemptuous", "frustrated"],
    "Fear & Overwhelm": ["terrified", "anxious", "panicked", "distressed"],
    "Depleted Disengagement": ["depressed", "weary", "hopeless", "resigned"],
    "Vigilant Suspicion": ["paranoid"],
    "Positive Anchors": ["joyful", "content", "proud"],
}
EMOTION_TO_CLUSTER = {e: c for c, es in CLUSTERS.items() for e in es}


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def compute_emotion_vectors(activations, emotions):
    """Anthropic mean-of-means: v_E = mean(E) - mean(all)"""
    mean_all = activations.mean(axis=0)
    return {e: activations[emotions == e].mean(axis=0) - mean_all for e in np.unique(emotions)}


def cosine_within_cross(vectors):
    """Compute within-cluster vs cross-cluster cosine on a set of emotion vectors."""
    emotions = list(vectors.keys())
    within, cross = [], []
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            if i >= j:
                continue
            c = cosine(vectors[e1], vectors[e2])
            (within if EMOTION_TO_CLUSTER.get(e1) == EMOTION_TO_CLUSTER.get(e2) else cross).append(c)
    return float(np.mean(within)), float(np.mean(cross)), float(np.mean(within) - np.mean(cross))


def probe_accuracy(activations, emotions, n_folds=5, seed=42):
    """Train logistic regression on activations -> emotion. Return cross-val accuracy."""
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    y = emotions

    accs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed, n_jobs=1)
        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[test_idx], y[test_idx])
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs)), accs


def bootstrap_within_cross(activations, emotions, n_resamples=100, seed=42):
    """Bootstrap CI on within-cross cosine diff by resampling stories."""
    rng = np.random.default_rng(seed)
    n = len(activations)
    diffs = []
    withins = []
    crosses = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        boot_acts = activations[idx]
        boot_emos = emotions[idx]
        # Need at least 2 stories per emotion for the cluster to be meaningful
        unique_emos, counts = np.unique(boot_emos, return_counts=True)
        if (counts < 2).any():
            continue
        vecs = compute_emotion_vectors(boot_acts, boot_emos)
        w, c, d = cosine_within_cross(vecs)
        withins.append(w)
        crosses.append(c)
        diffs.append(d)
    return {
        "within_mean": float(np.mean(withins)),
        "within_ci_low": float(np.percentile(withins, 2.5)),
        "within_ci_high": float(np.percentile(withins, 97.5)),
        "cross_mean": float(np.mean(crosses)),
        "cross_ci_low": float(np.percentile(crosses, 2.5)),
        "cross_ci_high": float(np.percentile(crosses, 97.5)),
        "diff_mean": float(np.mean(diffs)),
        "diff_ci_low": float(np.percentile(diffs, 2.5)),
        "diff_ci_high": float(np.percentile(diffs, 97.5)),
        "n_resamples": len(diffs),
    }


def permutation_test(activations, emotions, n_permutations=100, seed=42):
    """Test if within-cross diff is significantly larger than chance.

    Shuffle emotion labels, recompute diff. p-value = fraction of shuffled diffs >= actual diff.
    """
    rng = np.random.default_rng(seed)
    real_vecs = compute_emotion_vectors(activations, emotions)
    _, _, real_diff = cosine_within_cross(real_vecs)

    null_diffs = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(emotions)
        vecs = compute_emotion_vectors(activations, shuffled)
        _, _, d = cosine_within_cross(vecs)
        null_diffs.append(d)

    p_value = float(np.mean([d >= real_diff for d in null_diffs]))
    return {
        "real_diff": float(real_diff),
        "null_mean": float(np.mean(null_diffs)),
        "null_std": float(np.std(null_diffs)),
        "null_max": float(np.max(null_diffs)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def analyze_model(model_slug):
    model_dir = RESULTS_ROOT / model_slug
    activations_path = model_dir / "activations.npz"
    if not activations_path.exists():
        print(f"  Skipping {model_slug}: no activations.npz")
        return None

    print(f"\n=== {model_slug} ===")
    data = np.load(activations_path, allow_pickle=True)
    emotions = data["emotions"]
    layer_keys = sorted([k for k in data.files if k.startswith("layer_")])
    n_emotions = len(np.unique(emotions))
    chance = 1.0 / n_emotions

    print(f"  Stories: {len(emotions)}, emotions: {n_emotions}, chance accuracy: {chance:.3f}")

    results = {"model": model_slug, "n_emotions": int(n_emotions), "chance": chance, "layers": {}}

    for lk in layer_keys:
        print(f"  --- {lk} ---")
        acts = data[lk]

        # 5-fold cross-val probe
        mean_acc, std_acc, fold_accs = probe_accuracy(acts, emotions, n_folds=5)
        print(f"    Probe accuracy: {mean_acc:.3f} ± {std_acc:.3f} (chance: {chance:.3f}, factor: {mean_acc/chance:.1f}x)")

        # Bootstrap CI on within-cross diff (50 resamples — keep it fast)
        boot = bootstrap_within_cross(acts, emotions, n_resamples=50)
        print(f"    Within-cross diff: {boot['diff_mean']:.3f} [{boot['diff_ci_low']:.3f}, {boot['diff_ci_high']:.3f}]")

        # Permutation test (50 shuffles)
        perm = permutation_test(acts, emotions, n_permutations=50)
        print(f"    Permutation test: real={perm['real_diff']:.3f}, null mean={perm['null_mean']:.4f}, p={perm['p_value']:.3f}")

        results["layers"][lk] = {
            "probe_accuracy_mean": mean_acc,
            "probe_accuracy_std": std_acc,
            "probe_accuracy_factor_over_chance": mean_acc / chance,
            "fold_accuracies": fold_accs,
            "bootstrap": boot,
            "permutation": perm,
        }

    out_path = model_dir / "probe_results.json"
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

        print("\n" + "=" * 80)
        print("CROSS-MODEL PROBE ACCURACY SUMMARY")
        print("=" * 80)
        print(f"{'Model':<28} | {'Best layer':<10} | {'Best acc':>9} | {'Factor':>7}")
        print("-" * 70)
        for slug, r in all_results.items():
            best_layer = max(r["layers"], key=lambda lk: r["layers"][lk]["probe_accuracy_mean"])
            best_acc = r["layers"][best_layer]["probe_accuracy_mean"]
            factor = r["layers"][best_layer]["probe_accuracy_factor_over_chance"]
            print(f"{slug:<28} | {best_layer:<10} | {best_acc:>9.3f} | {factor:>6.1f}x")
    else:
        analyze_model(args.model)


if __name__ == "__main__":
    main()
