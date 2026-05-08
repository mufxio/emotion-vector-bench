# Cross-Model Findings — Emotion Vector Geometry on Open-Weight LLMs

*Written 2026-05-06. Companion document to the [emotion-vector-bench repo](../README.md).*

We ran the same 6-stage pipeline (extraction → vectors → validation → probes → arousal → implicit-emotion) on five open-weight language models from three different labs. All five pass the basic Anthropic-style geometry tests. They also reveal something we didn't expect: **two genuinely different ways to encode emotion**, both valid, neither dominant.

This document is the synthesis. The per-model details are in `results/{model_slug}/REPORT.md`.

## TL;DR

| Model | Probe acc | PC1 valence sep | Implicit top-3 | Geometric profile |
|---|---|---|---|---|
| Qwen2.5-1.5B | 89.7% | 7.30 | 20% | Compressed valence |
| Qwen2.5-7B | 91.8% | 12.06 | 60% | Strong valence |
| **Qwen3-8B** | 91.0% | **29.19** | 40% | Dominant valence axis |
| Llama-3.1-8B | **92.1%** | 2.71 | **60%** | Distributed clusters |
| Mistral-7B-v0.3 | 91.6% | 1.57 | 50% | Distributed clusters |

Permutation tests: p < 0.001 across every model, every layer.
Cross-layer stability: 0.96-0.99 across all five.

**Headline finding:** Qwen models compress emotion onto a strong valence axis. Llama and Mistral spread emotion across many cluster-defining directions without a single dominant valence axis. Both encode emotion richly enough to support 90%+ probe accuracy. They organize it differently.

## What we tested

The pipeline applies Anthropic's "Emotion Concepts" methodology (Sofroniew et al. 2026) to open-weight models. For each model, we:

1. **Extracted** residual-stream activations on a frozen corpus of 3000 emotion-evoking stories spanning 20 emotions × 30 topics.
2. **Computed** emotion vectors as `mean(stories of E) − mean(all stories)` at four sampled layers.
3. **Denoised** via PCA on neutral-dialogue activations (project out top components covering 50% variance).
4. **Validated** with three checks Anthropic ran qualitatively, plus three we added for rigor:
   - Cosine clustering: do similar emotions point similar directions?
   - PC1 valence: does positive vs negative emerge as the dominant organizing axis?
   - Cross-layer stability: is the geometry consistent across model depth?
   - **Linear probe accuracy**: can a logistic regression classify which of 20 emotions a story is about?
   - **Bootstrap CIs**: confidence intervals on cluster cohesion via story resampling.
   - **Permutation tests**: shuffle emotion labels and see if "structure" persists (it shouldn't).
5. **Recovered** the affective circumplex: which PC encodes valence, which encodes arousal.
6. **Probed implicit-emotion scenarios**: 10 scenarios that evoke an emotion without naming it. Does the right vector activate?

Full code is at `code/run_bench.py`.

## What we found

### 1. All five models pass with statistical significance

Probe accuracy on a 20-way emotion classification (chance = 5%):

| Model | Best layer | Probe accuracy | Factor over chance |
|---|---|---|---|
| Llama-3.1-8B | 16/32 | 92.1% ± 0.8% | 18.4× |
| Qwen2.5-7B | 18/28 | 91.8% ± 0.9% | 18.4× |
| Mistral-7B-v0.3 | 16/32 | 91.6% ± 0.8% | 18.3× |
| Qwen3-8B | 23/36 | 91.0% ± 0.8% | 18.2× |
| Qwen2.5-1.5B | 17/28 | 89.7% ± 1.0% | 17.9× |

Permutation tests confirmed the cluster signal is real: real within-cross diff (~0.20) was 50-100× larger than what shuffled-label baselines produced. p < 0.001 universally.

This is the strongest evidence that emotion-vector geometry isn't a quirk of large frontier models — every model in this scale class has emotion linearly accessible at high accuracy.

### 2. Valence axis strength varies dramatically

PC1 valence separation (positive emotions minus negative emotions on PC1):

```
Qwen3-8B:        ████████████████████████████████ 29.19
Qwen2.5-7B:      █████████████ 12.06
Qwen2.5-1.5B:    ████████ 7.30
Llama-3.1-8B:    ███ 2.71
Mistral-7B-v0.3: █▌ 1.57
```

The 18× spread is the most striking cross-model finding. Within the Qwen family there's a clear scale ladder (1.5B → 7B → 8B = 7.3 → 12.1 → 29.2). Across families the difference is starker still: Qwen3-8B has 18× cleaner valence separation than Mistral-7B at the same parameter scale.

### 3. Within-cluster cohesion goes the OTHER way

Within-cluster minus cross-cluster cosine difference (higher = tighter clusters):

| Model | Within−cross diff |
|---|---|
| Llama-3.1-8B | 0.252 |
| Mistral-7B-v0.3 | 0.247 |
| Qwen3-8B | 0.233 |
| Qwen2.5-1.5B | 0.222 |
| Qwen2.5-7B | 0.215 |

Llama and Mistral have **tighter** within-cluster cohesion than the Qwen family. So the same models that have flat valence axes have sharper individual clusters.

### 4. Two geometric profiles

Putting findings 2 and 3 together, we see two distinct ways the same task gets solved:

**Profile A — Dominant valence axis (Qwen family):**
- Strong PC1 separating positive from negative
- Looser within-cluster cohesion (~0.215-0.233)
- Like organizing a library by genre first (fiction/non-fiction), then topic

**Profile B — Distributed clusters (Llama, Mistral):**
- Weak PC1 valence separation (1.57-2.71)
- Tighter within-cluster cohesion (~0.247-0.252)
- Like organizing a library by topic, with no top-level genre split

Both produce 91-92% probe accuracy. Both pass cross-layer stability. Both recover the arousal axis somewhere in PC2 or PC3. The 20 emotions are distinguishable in both. They just live in different shapes.

### 5. Implicit-emotion test reveals practical implications

The implicit-emotion scenarios (10 scenarios that evoke specific emotions without naming them):

| Model | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Llama-3.1-8B | 20% | 60% | 80% |
| Qwen2.5-7B | 20% | 60% | 70% |
| Mistral-7B-v0.3 | 10% | 50% | 60% |
| Qwen3-8B | 20% | 40% | 60% |
| Qwen2.5-1.5B | 20% | 20% | 20% |

(Chance: 5% / 15% / 25%)

Notable: **Llama matches the best Qwen model on this practical test**, despite having the flattest valence axis. The "distributed clusters" profile is enough to identify implicit emotions when the cluster cohesion is tight.

Qwen 1.5B is sharply worse, suggesting smaller models have insufficient resolution for situational inference even when their basic geometry passes (89.7% probe accuracy is high, but Top-3 implicit accuracy is only 20%).

### 6. Affective circumplex: valence and arousal both recover

Anthropic found PC1 = valence and PC2 = arousal in Sonnet 4.5, recovering Russell's 1980 affective circumplex. Our results:

- **Valence is always PC1** in all five models — the most consistent finding across families.
- **Arousal lands on PC2 or PC3** depending on model. Mistral has it on PC2 (matching Anthropic). Most others on PC3, meaning a non-arousal dimension captures more variance second-most. This could be agency/dominance (Russell-Mehrabian PAD model), self-vs-other focus, or a corpus artifact (positive emotions in our corpus are positive *reframes* of negative topics, which may show up structurally).

Either way, both core axes of the human affective circumplex are recoverable in every model. The 2D layout in `plots/circumplex_layer_X.png` for each model shows this visually.

## What this means

### For interpretability research

Anthropic's emotion-vector methodology generalizes across model families. You don't need a frontier model to study emotion concepts — Qwen 7B on a Mac mini gives you 91.8% probe accuracy on 20-way emotion classification, which is more than enough to do steering, residue detection, or feature analysis.

The 18× spread in valence-axis strength is a meaningful difference between architectures that hasn't been documented before (to our knowledge). It suggests training-data composition or RLHF mix matters substantially for how emotion gets encoded, even when downstream task accuracy is similar.

### For downstream applications

Choosing a model for emotion-related interpretability work depends on what you need:

- **Sentiment/valence detection (positive vs negative residue)** → Qwen3-8B. Cleanest valence axis means even crude probes pick up sign of emotion accurately.
- **Fine-grained emotion identification (which of N emotions)** → Llama-3.1-8B. Tightest within-cluster cohesion + strongest probe accuracy.
- **Long-form scenario inference (implicit emotion in stories)** → Llama-3.1-8B or Qwen2.5-7B (60% Top-3).
- **All-around** → Qwen2.5-7B. Consistent in the top half across every metric.
- **Smallest workable model** → Qwen2.5-1.5B passes basic tests but loses substantially on situational tasks. 7B class is the practical floor.

### For methodology

The paper threshold-based audit gates we used initially ("PC1 sep > 1.0") were eyeball judgments. The bootstrap CIs and permutation tests show those judgments held up — but quantitative metrics (probe accuracy, p-values) are what enable principled cross-model comparison.

## Honest caveats

1. **Corpus generator bias.** All 3000 stories were written by Sonnet 4.5. Sonnet has narrative defaults per emotion (e.g., desperate = phone-calling loops). These may inflate within-cluster cohesion across all models tested with this corpus.

2. **Positive-anchor confound.** Our 3 positive emotions (joyful, content, proud) are written as positive *reframes of negative life events* (relief from eviction, vindication after rejection, etc.). They're not pure positive scenarios. This may suppress PC1 valence separation in models that don't internally normalize "positive interpretation of bad event" as fully positive.

3. **PC2 ≠ arousal in most of our models.** Anthropic found arousal on PC2; we found it on PC3 in 4 of 5 models. PC2 is some other dimension (maybe agency, maybe a corpus artifact). Worth investigating but doesn't undermine the core findings.

4. **Implicit-emotion test is small (n=10 scenarios).** Top-1 accuracy of 20% means 2 of 10 scenarios got the exact target. Single-scenario noise dominates. We use Top-3 / Top-5 because the right *family* of emotion is what matters most for downstream work.

5. **Open-weight scale gap.** All five models are 1.5B-9B. Anthropic's Sonnet 4.5 is much larger. Geometric structure may sharpen at frontier scale beyond what we observe here.

6. **No causal validation.** We measured representational structure, not steering effects. Anthropic's full paper additionally demonstrates that emotion vectors causally influence behavior via steering. We do not test this. (The vectors are saved at `results/{model}/denoised_vectors.npz` if anyone wants to.)

## Reproducibility

To reproduce on any model:

```bash
git clone <this repo>
cd emotion-vector-bench
pip install -r requirements.txt
python code/run_bench.py --model <huggingface model id>
```

Result lands in `results/{model_slug}/REPORT.md`.

The full corpus (stories, neutral dialogues, implicit scenarios) is in `corpus/`. The reference results are in `results/`. Activations files (~150MB each) are gitignored but regenerable from the corpus.

## Source

Sofroniew, Kauvar, Saunders et al., "Emotion Concepts and their Function in a Large Language Model," *Transformer Circuits Thread*, April 2026.

This work extends their methodology to a comparative open-weight setting. The corpus is ours (Anthropic published the recipe but not the data). The statistical methods (bootstrap CIs, permutation tests, probe accuracy as primary metric) are added on top of their qualitative validation.

## Citation

```
emotion-vector-bench: A standardized corpus and reproducible recipe for testing 
emotion-vector geometry across open-weight language models. 2026.
```
