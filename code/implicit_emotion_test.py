"""
Implicit-emotion scenario test.

Scenarios that evoke specific emotions WITHOUT using the emotion word or direct
synonyms — emotion is conveyed only through situational content, behavior, body
language. This tests whether the emotion vectors encode emotion CONCEPTS or just
emotion WORDS.

Anthropic's Table 2 has 12 such scenarios. We use 10 of our own targeting a mix.

Process:
  1. For each scenario, run through the model, extract activation at probe layer
  2. Project onto each of the 20 emotion vectors (cosine similarity)
  3. Rank emotions by activation
  4. Check: is the target emotion in the top-K predictions?

Top-1 accuracy = the target won outright
Top-3 accuracy = the target was in the top 3
Top-5 accuracy = the target was in the top 5
Random chance: top-1 = 5%, top-3 = 15%, top-5 = 25%

Usage:
    python implicit_emotion_test.py --model qwen2.5-7b-instruct
    python implicit_emotion_test.py --model all
"""

import transformers.modeling_utils as _modeling_utils
_modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nnsight import LanguageModel

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
START_TOKEN = 50

# Implicit-emotion scenarios are loaded from corpus/implicit_scenarios.jsonl
# (a transparent corpus artifact, version-controlled).
def load_scenarios():
    path = REPO_ROOT / "corpus" / "implicit_scenarios.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]

SCENARIOS = None  # populated in main()


def model_slug(model_name):
    return model_name.split("/")[-1].lower()


def load_model_for(slug):
    """Resolve a slug to a full HF model name and load it."""
    slug_to_hf = {
        "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    }
    hf_name = slug_to_hf.get(slug)
    if hf_name is None:
        raise ValueError(f"Unknown slug: {slug}")
    use_bf16 = "llama-3" in slug or "mistral" in slug
    print(f"  Loading {hf_name} ({'bf16' if use_bf16 else 'fp16'})...", flush=True)
    return LanguageModel(
        hf_name,
        device_map={"": "mps"},
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )


def extract_activation(model, text, layer):
    saved = {}
    with model.trace(text):
        saved[layer] = model.model.layers[layer].output[0].save()
    act = saved[layer]
    n_tokens = act.shape[0]
    start = min(START_TOKEN, n_tokens // 2)
    return act[start:, :].float().mean(dim=0).detach().cpu().numpy()


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def best_layer_for_model(slug):
    """Choose the layer with strongest valence separation (from arousal_results.json)."""
    with open(RESULTS_ROOT / slug / "arousal_results.json") as f:
        r = json.load(f)
    return max(r["layers"].keys(), key=lambda lk: r["layers"][lk]["valence_separation"])


def load_emotion_vectors(slug, layer_key):
    """Load denoised emotion vectors for a model at a specific layer."""
    path = RESULTS_ROOT / slug / "denoised_vectors.npz"
    data = np.load(path)
    prefix = f"{layer_key}__"
    return {k.replace(prefix, ""): data[k] for k in data.files if k.startswith(prefix)}


def analyze_model(slug):
    print(f"\n=== {slug} ===")
    layer_key = best_layer_for_model(slug)
    layer_idx = int(layer_key.replace("layer_", ""))
    print(f"  Using layer {layer_idx} (best valence separation)")
    vectors = load_emotion_vectors(slug, layer_key)

    model = load_model_for(slug)
    print(f"  Running {len(SCENARIOS)} scenarios through model...", flush=True)

    scenario_results = []
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    for sc in SCENARIOS:
        target = sc["target"]
        text = sc["text"]
        act = extract_activation(model, text, layer_idx)
        # Project onto each emotion vector
        scores = {emo: cosine(act, v) for emo, v in vectors.items()}
        # Rank
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        target_rank = next(i for i, (e, _) in enumerate(ranked) if e == target) + 1
        in_top1 = target_rank == 1
        in_top3 = target_rank <= 3
        in_top5 = target_rank <= 5
        if in_top1: correct_top1 += 1
        if in_top3: correct_top3 += 1
        if in_top5: correct_top5 += 1

        scenario_results.append({
            "target": target,
            "target_rank": target_rank,
            "top5_emotions": [e for e, _ in ranked[:5]],
            "top5_scores": [s for _, s in ranked[:5]],
        })

        marker = "✓" if in_top1 else ("●" if in_top3 else ("◐" if in_top5 else "✗"))
        top3_str = ", ".join(e for e, _ in ranked[:3])
        print(f"    {marker} target={target:<16} | rank #{target_rank:<2} | top3: {top3_str}")

    n = len(SCENARIOS)
    print(f"\n  Top-1 accuracy: {correct_top1}/{n} = {correct_top1/n:.1%} (chance: 5%)")
    print(f"  Top-3 accuracy: {correct_top3}/{n} = {correct_top3/n:.1%} (chance: 15%)")
    print(f"  Top-5 accuracy: {correct_top5}/{n} = {correct_top5/n:.1%} (chance: 25%)")

    out = {
        "model": slug,
        "layer": layer_key,
        "scenarios": scenario_results,
        "top1_accuracy": correct_top1 / n,
        "top3_accuracy": correct_top3 / n,
        "top5_accuracy": correct_top5 / n,
        "n_scenarios": n,
    }
    with open(RESULTS_ROOT / slug / "implicit_emotion_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {RESULTS_ROOT / slug / 'implicit_emotion_results.json'}")

    # Free model before next
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import gc
    gc.collect()
    return out


def main():
    global SCENARIOS
    SCENARIOS = load_scenarios()
    print(f"Loaded {len(SCENARIOS)} implicit-emotion scenarios from corpus", flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model slug, or 'all'")
    args = ap.parse_args()

    if args.model == "all":
        all_results = {}
        for d in sorted(RESULTS_ROOT.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            r = analyze_model(d.name)
            all_results[d.name] = r

        print("\n" + "=" * 80)
        print("CROSS-MODEL IMPLICIT-EMOTION SUMMARY")
        print("=" * 80)
        print(f"{'Model':<28} | {'Layer':<10} | {'Top-1':>6} | {'Top-3':>6} | {'Top-5':>6}")
        print("-" * 70)
        for slug, r in all_results.items():
            print(f"{slug:<28} | {r['layer']:<10} | {r['top1_accuracy']:>6.0%} | "
                  f"{r['top3_accuracy']:>6.0%} | {r['top5_accuracy']:>6.0%}")
        print(f"\n(chance: 5% / 15% / 25%)")
    else:
        analyze_model(args.model)


if __name__ == "__main__":
    main()
