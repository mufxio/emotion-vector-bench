"""Generate a composite hero image for the README — cross-model bar chart +
a representative affective circumplex side by side.

Output: results/_hero.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

# Load cross-model summary
with open(RESULTS / "_comparison.json") as f:
    comp = json.load(f)

# Sort: Qwen family first (by scale), then Llama, Mistral
order = [
    "qwen2.5-1.5b-instruct",
    "qwen2.5-7b-instruct",
    "qwen3-8b",
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
]
comp_sorted = sorted(comp, key=lambda x: order.index(x["model"]) if x["model"] in order else 99)

short_names = {
    "qwen2.5-1.5b-instruct": "Qwen2.5-1.5B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
    "qwen3-8b": "Qwen3-8B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "mistral-7b-instruct-v0.3": "Mistral-7B",
}
colors = {
    "qwen2.5-1.5b-instruct": "#9ecae1",
    "qwen2.5-7b-instruct": "#3182bd",
    "qwen3-8b": "#08519c",
    "llama-3.1-8b-instruct": "#d62728",
    "mistral-7b-instruct-v0.3": "#2ca02c",
}

fig = plt.figure(figsize=(15, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)

# Panel 1: cross-model PC1 valence comparison bar chart
ax1 = fig.add_subplot(gs[0])
names = [short_names[s["model"]] for s in comp_sorted]
pc1_vals = [s["pc1_sep"] for s in comp_sorted]
bar_colors = [colors[s["model"]] for s in comp_sorted]
bars = ax1.barh(names, pc1_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, pc1_vals):
    ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}",
             ha="left", va="center", fontsize=11, fontweight="bold")
ax1.set_xlim(0, max(pc1_vals) * 1.15)
ax1.set_xlabel("PC1 valence separation (positive emotions vs negative emotions)", fontsize=11)
ax1.set_title("Same corpus, 5 models — 18× spread in valence axis strength",
              fontsize=12, fontweight="bold", loc="left", pad=12)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.invert_yaxis()
ax1.grid(axis="x", alpha=0.25, linestyle="--")

# Panel 2: Qwen3-8B affective circumplex — best example
# Load denoised vectors and recompute the circumplex projection
qwen3_vecs_path = RESULTS / "qwen3-8b" / "denoised_vectors.npz"
data = np.load(qwen3_vecs_path)
# Use layer_28 — the best layer
layer_key = "layer_28"
prefix = f"{layer_key}__"
vectors = {k.replace(prefix, ""): data[k] for k in data.files if k.startswith(prefix)}
emotions = list(vectors.keys())
M = np.stack([vectors[e] for e in emotions], axis=0)
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
coords = pca.fit_transform(M)

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

ax2 = fig.add_subplot(gs[1])
# Use PC1 (valence) and PC3 (arousal in qwen3)
x = coords[:, 0]
y = coords[:, 2]
for emo, xi, yi in zip(emotions, x, y):
    cluster = EMOTION_TO_CLUSTER.get(emo, "")
    color = CLUSTER_COLORS.get(cluster, "black")
    ax2.scatter(xi, yi, c=color, s=150, edgecolors="black", linewidths=0.7, zorder=3)
    ax2.annotate(emo, (xi, yi), fontsize=8.5, ha="left", va="bottom", xytext=(4, 3),
                 textcoords="offset points")
ax2.axhline(0, color="gray", linewidth=0.5, zorder=1)
ax2.axvline(0, color="gray", linewidth=0.5, zorder=1)
ax2.set_xlabel("Valence (PC1) →", fontsize=11)
ax2.set_ylabel("Arousal (PC3) →", fontsize=11)
ax2.set_title("Affective circumplex emerges (Qwen3-8B, layer 28)",
              fontsize=12, fontweight="bold", loc="left", pad=12)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(alpha=0.25, linestyle="--")

plt.suptitle(
    "emotion-vector-bench  •  Anthropic-style emotion geometry across 5 open-weight LLMs  •  one command, runs on a Mac mini",
    fontsize=10, color="#444", y=0.99,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = RESULTS / "_hero.png"
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved hero image → {out}")
