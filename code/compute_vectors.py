"""
Compute emotion vectors using Anthropic's mean-of-means + PCA denoising.

For each emotion E and each layer:
    v_E = mean(E's stories activations) - mean(all stories activations)

Then denoise: PCA on neutral_activations, project out top components covering
50% of variance from each emotion vector.

Usage:
    python compute_vectors.py --model qwen2.5-7b-instruct
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
VARIANCE_THRESHOLD = 0.50


def compute_vectors(activations: np.ndarray, emotions: np.ndarray) -> dict[str, np.ndarray]:
    mean_all = activations.mean(axis=0)
    return {
        emotion: activations[emotions == emotion].mean(axis=0) - mean_all
        for emotion in np.unique(emotions)
    }


def fit_pca_from_neutral(neutral_activations: np.ndarray):
    pca = PCA()
    pca.fit(neutral_activations)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, VARIANCE_THRESHOLD)) + 1
    return pca.components_[:k], pca.explained_variance_ratio_[:k]


def project_out(v: np.ndarray, components: np.ndarray) -> np.ndarray:
    out = v.copy()
    for c in components:
        out = out - (np.dot(out, c) / np.dot(c, c)) * c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model slug (e.g. qwen2.5-7b-instruct)")
    args = ap.parse_args()

    model_dir = RESULTS_ROOT / args.model
    activations_path = model_dir / "activations.npz"
    neutral_path = model_dir / "neutral_activations.npz"

    print(f"Loading {activations_path}...")
    data = np.load(activations_path, allow_pickle=True)
    emotions = data["emotions"]
    layer_keys = [k for k in data.files if k.startswith("layer_")]

    has_neutral = neutral_path.exists()
    if has_neutral:
        neutral_data = np.load(neutral_path)

    raw_vectors = {}
    denoised_vectors = {}
    pca_info = {}

    for lkey in layer_keys:
        print(f"\n=== {lkey} ===")
        acts = data[lkey]
        vecs = compute_vectors(acts, emotions)
        raw_vectors[lkey] = vecs

        norms = [np.linalg.norm(v) for v in vecs.values()]
        print(f"  Vector norms: min={min(norms):.2f}, max={max(norms):.2f}, mean={np.mean(norms):.2f}")

        if has_neutral:
            components, explained = fit_pca_from_neutral(neutral_data[lkey])
            print(f"  PCA: top {len(components)} comps cover {explained.sum():.1%} of neutral variance")
            pca_info[lkey] = {"k": len(components), "explained_variance_ratio": explained.tolist()}
            denoised_vecs = {e: project_out(v, components) for e, v in vecs.items()}
            denoised_vectors[lkey] = denoised_vecs

    flat_raw = {f"{lkey}__{e}": v for lkey, vecs in raw_vectors.items() for e, v in vecs.items()}
    np.savez_compressed(model_dir / "raw_vectors.npz", **flat_raw)
    print(f"\nSaved raw vectors → {model_dir / 'raw_vectors.npz'}")

    if has_neutral:
        flat_d = {f"{lkey}__{e}": v for lkey, vecs in denoised_vectors.items() for e, v in vecs.items()}
        np.savez_compressed(model_dir / "denoised_vectors.npz", **flat_d)
        with open(model_dir / "pca_info.json", "w") as f:
            json.dump(pca_info, f, indent=2)
        print(f"Saved denoised vectors → {model_dir / 'denoised_vectors.npz'}")


if __name__ == "__main__":
    main()
