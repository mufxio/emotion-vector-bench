"""
emotion-vector-bench — unified entry point.

Single command: takes a HuggingFace model name, runs the full evaluation pipeline,
produces a consolidated report.

Usage:
    python run_bench.py --model Qwen/Qwen3-8B
    python run_bench.py --model meta-llama/Llama-3.1-8B-Instruct

Pipeline stages (each is a separate script that can also be run standalone):
    1. extract.py              — residual-stream activations on the corpus
    2. compute_vectors.py      — emotion vectors (mean-of-means + PCA denoise)
    3. validate.py             — basic geometry checks (cosine, PCA, layer stability)
    4. probe_accuracy.py       — linear probe accuracy + bootstrap CIs + permutation tests
    5. arousal_axis.py         — PC2/3 arousal recovery + affective circumplex
    6. implicit_emotion_test.py — scenarios that evoke emotion without naming it
    7. generate_report.py      — synthesize REPORT.md per-model

Inputs:
    - Model name (HuggingFace ID)
    - corpus/  — frozen 3000 stories, 50 neutral dialogues, 10 implicit scenarios

Outputs (under results/{model_slug}/):
    - activations.npz              — raw residual-stream activations
    - neutral_activations.npz       — for PCA denoising
    - raw_vectors.npz, denoised_vectors.npz, pca_info.json
    - validation_results.json      — cosine, PCA, layer stability
    - probe_results.json           — accuracy + bootstrap CIs + permutation tests
    - arousal_results.json         — affective circumplex
    - implicit_emotion_results.json — top-1/3/5 accuracy on implicit scenarios
    - REPORT.md                    — human-readable synthesis
    - plots/                       — heatmaps, PCA scatter, circumplex
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def run_stage(stage_name: str, script: str, args: list[str]) -> bool:
    """Run a pipeline stage. Returns True on success, False on failure."""
    print("\n" + "=" * 80)
    print(f"  STAGE: {stage_name}")
    print("=" * 80, flush=True)
    cmd = [sys.executable, "-u", str(CODE_DIR / script)] + args
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ {stage_name} FAILED (exit code {result.returncode}, {elapsed:.0f}s)", flush=True)
        return False
    print(f"\n  ✓ {stage_name} done ({elapsed:.0f}s)", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Run the full emotion-vector-bench pipeline on a target model.",
    )
    ap.add_argument("--model", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3-8B)")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--quantize", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--skip-extraction", action="store_true",
                    help="Skip extraction (assumes activations.npz already exists)")
    args = ap.parse_args()

    slug = model_slug(args.model)
    out_dir = REPO_ROOT / "results" / slug

    print("\n" + "█" * 80)
    print(f"  emotion-vector-bench → {args.model}")
    print(f"  Output: {out_dir}")
    print("█" * 80, flush=True)

    pipeline_t0 = time.time()
    stages_done = []

    # Stage 1: Extract activations (the slow one)
    if not args.skip_extraction:
        ok = run_stage(
            "1/6 Extract activations (~30-90 min depending on model size)",
            "extract.py",
            ["--model", args.model, "--device", args.device, "--quantize", args.quantize],
        )
        if not ok: sys.exit(1)
        stages_done.append("extract")
    else:
        print("\n  Skipping extraction (--skip-extraction)", flush=True)

    # Stage 2: Compute emotion vectors
    ok = run_stage(
        "2/6 Compute emotion vectors (mean-of-means + PCA denoise)",
        "compute_vectors.py",
        ["--model", slug],
    )
    if not ok: sys.exit(1)
    stages_done.append("vectors")

    # Stage 3: Basic validation (cosine, PCA, layer stability)
    ok = run_stage(
        "3/6 Validate geometry (cosine clustering + PCA + layer stability)",
        "validate.py",
        ["--model", slug],
    )
    if not ok: sys.exit(1)
    stages_done.append("validate")

    # Stage 4: Probe accuracy + bootstrap + permutation
    ok = run_stage(
        "4/6 Probe accuracy + bootstrap CIs + permutation tests",
        "probe_accuracy.py",
        ["--model", slug],
    )
    if not ok: sys.exit(1)
    stages_done.append("probe")

    # Stage 5: Arousal axis recovery (PC2/3)
    ok = run_stage(
        "5/6 Affective circumplex (valence + arousal axes)",
        "arousal_axis.py",
        ["--model", slug],
    )
    if not ok: sys.exit(1)
    stages_done.append("arousal")

    # Stage 6: Implicit-emotion scenarios
    ok = run_stage(
        "6/6 Implicit-emotion scenario test",
        "implicit_emotion_test.py",
        ["--model", slug],
    )
    if not ok: sys.exit(1)
    stages_done.append("implicit")

    # Generate consolidated report
    ok = run_stage(
        "Generate REPORT.md",
        "generate_report.py",
        ["--model", slug],
    )
    if not ok:
        print("  (Report generation failed, but all data is saved)", flush=True)

    total_elapsed = time.time() - pipeline_t0
    print("\n" + "█" * 80)
    print(f"  PIPELINE COMPLETE — {args.model}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Stages done: {', '.join(stages_done)}")
    print(f"  Results: {out_dir}")
    print(f"  See REPORT.md for human-readable synthesis")
    print("█" * 80, flush=True)


if __name__ == "__main__":
    main()
