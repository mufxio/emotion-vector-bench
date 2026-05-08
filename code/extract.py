"""
Extract residual stream activations from a target model on the frozen story corpus.

Usage:
    python extract.py --model Qwen/Qwen2.5-7B-Instruct
    python extract.py --model meta-llama/Llama-3.1-8B-Instruct --quantize 4bit

For each story:
- Forward pass through the model with hooks on residual stream at multiple layers
- Average activations across token positions starting from token 50 (Anthropic protocol)
- Save per-layer numpy arrays of shape [n_stories, hidden_dim]

Checkpointed: saves partial state every CHECKPOINT_EVERY stories. Resume-safe.
Calls torch.mps.empty_cache() between checkpoints to fight MPS allocator fragmentation.

Output: results/{model_slug}/activations.npz, neutral_activations.npz
"""

# Disable MPS allocator warmup BEFORE importing transformers
# (it tries to allocate one giant buffer that exceeds Apple's MPS single-buffer limit)
import transformers.modeling_utils as _modeling_utils
_modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from nnsight import LanguageModel

# Static config
START_TOKEN = 50
CHECKPOINT_EVERY = 50

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
STORIES_DIR = CORPUS_DIR / "stories"
NEUTRAL_PATH = CORPUS_DIR / "neutral_dialogues.jsonl"
RESULTS_ROOT = REPO_ROOT / "results"


def model_slug(model_name: str) -> str:
    """Convert e.g. 'Qwen/Qwen2.5-7B-Instruct' to 'qwen2.5-7b-instruct'."""
    return model_name.split("/")[-1].lower()


def pick_target_layers(n_layers: int) -> list[int]:
    """Return four layers spread through the model: ~28%, ~50%, ~64%, ~80%."""
    return [
        int(n_layers * 0.28),
        int(n_layers * 0.50),
        int(n_layers * 0.64),
        int(n_layers * 0.80),
    ]


def load_corpus():
    rows = []
    for jsonl_path in sorted(STORIES_DIR.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def extract_for_text(model, text: str, layers: list[int]) -> dict[int, np.ndarray]:
    saved = {}
    with model.trace(text):
        for l in layers:
            saved[l] = model.model.layers[l].output[0].save()
    out = {}
    for l in layers:
        act = saved[l]
        n_tokens = act.shape[0]
        start = min(START_TOKEN, n_tokens // 2)
        avg = act[start:, :].float().mean(dim=0).detach().cpu().numpy()
        out[l] = avg
    del saved
    return out


def save_checkpoint(checkpoint_path, activations, completed, emotions, topics, story_idxs, perspectives, n_total, layers):
    save_dict = {f"layer_{l}": activations[l] for l in layers}
    save_dict.update({
        "completed": np.array([completed]),
        "n_total": np.array([n_total]),
        "layers": np.array(layers),
        "emotions": np.array(emotions),
        "topics": np.array(topics),
        "story_idxs": np.array(story_idxs),
        "perspectives": np.array(perspectives),
    })
    tmp = checkpoint_path.parent / "extraction_checkpoint_tmp.npz"
    np.savez_compressed(str(tmp).replace(".npz", ""), **save_dict)
    tmp.replace(checkpoint_path)


def load_checkpoint(checkpoint_path, layers):
    if not checkpoint_path.exists():
        return 0, None, None, None, None, None
    cp = np.load(checkpoint_path, allow_pickle=True)
    completed = int(cp["completed"][0])
    activations = {l: cp[f"layer_{l}"] for l in layers}
    emotions = list(cp["emotions"])
    topics = list(cp["topics"])
    story_idxs = list(cp["story_idxs"])
    perspectives = list(cp["perspectives"])
    print(f"Resumed from checkpoint: {completed} stories already done", flush=True)
    return completed, activations, emotions, topics, story_idxs, perspectives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--quantize", choices=["none", "4bit", "8bit"], default="none",
                    help="Use bitsandbytes quantization (4bit/8bit) for larger models")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = ap.parse_args()

    slug = model_slug(args.model)
    out_dir = RESULTS_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Extraction — {args.model}")
    print(f"Output dir: {out_dir}")
    print("=" * 60, flush=True)

    print(f"\nLoading {args.model}...", flush=True)
    t0 = time.time()
    # Use bfloat16 if model was trained in bf16 (Llama-3.1, Mistral newer)
    use_bf16 = "llama-3" in args.model.lower() or "llama_3" in args.model.lower() or "mistral" in args.model.lower()
    load_kwargs = {
        "device_map": {"": args.device},
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        "low_cpu_mem_usage": True,
    }
    print(f"Using dtype: {'bfloat16' if use_bf16 else 'float16'}", flush=True)
    if args.quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        load_kwargs.pop("torch_dtype", None)
    elif args.quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("torch_dtype", None)

    model = LanguageModel(args.model, **load_kwargs)
    print(f"Constructed in {time.time() - t0:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    target_layers = pick_target_layers(n_layers)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim", flush=True)
    print(f"Target layers: {target_layers}", flush=True)

    rows = load_corpus()
    n = len(rows)
    print(f"Loaded {n} stories from corpus", flush=True)

    # Resume-safe checkpoint
    checkpoint_path = out_dir / "extraction_checkpoint.npz"
    completed, prev_acts, emotions, topics, story_idxs, perspectives = load_checkpoint(checkpoint_path, target_layers)

    if prev_acts is None:
        activations = {l: np.zeros((n, hidden_dim), dtype=np.float32) for l in target_layers}
        emotions, topics, story_idxs, perspectives = [], [], [], []
    else:
        activations = prev_acts

    print(f"\nStarting from index {completed}/{n}", flush=True)
    t0 = time.time()
    last_checkpoint_time = t0
    for i in range(completed, n):
        row = rows[i]
        out = extract_for_text(model, row["text"], target_layers)
        for l in target_layers:
            activations[l][i] = out[l]
        emotions.append(row["emotion"])
        topics.append(row["topic"])
        story_idxs.append(row["story_index"])
        perspectives.append(row.get("perspective", "unknown"))

        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt_time = time.time()
            recent_rate = CHECKPOINT_EVERY / (ckpt_time - last_checkpoint_time)
            last_checkpoint_time = ckpt_time
            remaining = n - (i + 1)
            eta = remaining / max(recent_rate, 0.001)
            print(f"  [{i+1:>4}/{n}] rate={recent_rate:.2f}/s, eta={eta:.0f}s", flush=True)
            save_checkpoint(checkpoint_path, activations, i+1, emotions, topics, story_idxs, perspectives, n, target_layers)
            if args.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

    print(f"\nTotal extraction time: {time.time() - t0:.0f}s", flush=True)

    # Save final activations (small subset — just labels and vectors, not full tensors)
    final_path = out_dir / "activations.npz"
    save_dict = {f"layer_{l}": activations[l] for l in target_layers}
    save_dict.update({
        "layers": np.array(target_layers),
        "emotions": np.array(emotions),
        "topics": np.array(topics),
        "story_idxs": np.array(story_idxs),
        "perspectives": np.array(perspectives),
    })
    np.savez_compressed(final_path, **save_dict)
    print(f"Saved → {final_path}", flush=True)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Neutral dialogues
    if NEUTRAL_PATH.exists():
        print(f"\nProcessing neutral dialogues...", flush=True)
        with open(NEUTRAL_PATH) as f:
            neutral_rows = [json.loads(l) for l in f]
        n_neutral = len(neutral_rows)
        neutral_acts = {l: np.zeros((n_neutral, hidden_dim), dtype=np.float32) for l in target_layers}
        for i, row in enumerate(neutral_rows):
            out = extract_for_text(model, row["text"], target_layers)
            for l in target_layers:
                neutral_acts[l][i] = out[l]
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_neutral}]", flush=True)
                if args.device == "mps" and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        save_dict_n = {f"layer_{l}": neutral_acts[l] for l in target_layers}
        save_dict_n["layers"] = np.array(target_layers)
        neutral_out = out_dir / "neutral_activations.npz"
        np.savez_compressed(neutral_out, **save_dict_n)
        print(f"Saved → {neutral_out}", flush=True)


if __name__ == "__main__":
    main()
