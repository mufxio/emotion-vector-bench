# Methodology

This document describes exactly what we did, what's in the corpus, and what limitations we know about. The methodology is Anthropic's; the corpus is ours.

## Source

Sofroniew, Kauvar, Saunders, Chen, Henighan, Hydrie, Citro, Pearce, Tarng, Gurnee, Batson, Zimmerman, Rivoire, Fish, Olah, Lindsey. "Emotion Concepts and their Function in a Large Language Model." *Transformer Circuits Thread*, April 2026. https://transformer-circuits.pub/2026/emotions/index.html

We follow the methodology in their Section 1.1 ("Finding emotion vectors") and Section 2.1 ("Geometry of emotion space"). We use the verbatim prompt from their Appendix Section 6.5.

## Corpus design

### Emotions (n=20)

Drawn from Anthropic's full 171-emotion list (their page 58). We picked a balanced subset across 6 of their 10 k-means clusters:

| Cluster | Emotions |
|---|---|
| Despair & Shame (4) | desperate, grief-stricken, ashamed, lonely |
| Hostile Anger (4) | furious, resentful, contemptuous, frustrated |
| Fear & Overwhelm (4) | terrified, anxious, panicked, distressed |
| Depleted Disengagement (4) | depressed, weary, hopeless, resigned |
| Vigilant Suspicion (1) | paranoid |
| Positive Anchors (3) | joyful, content, proud |

The 17 negative + 3 positive split was deliberate: positive anchors give PCA a valence axis to recover. Three positive emotions span the positive subspace minimally (high-arousal joyful, low-arousal content, agency-focused proud).

### Topics (n=30)

Subset of Anthropic's 100-topic list, balanced across 6 domains:

- **Relational/Family** (8): family religion change, partner secrets, ex at wedding, friend moving, sibling inheritance, teen secret social, roommate journal, adult child returning
- **Professional/Career** (6): company sold, training replacement, junior pay disparity, manuscript rejected, song stolen, cutting a player
- **Financial/Property** (5): eviction, crime-scene house, towed car, fallen tree, secret-rich neighbor
- **Educational** (3): scholarship denied, plagiarism accusation, lower grade
- **Identity/Discovery** (3): adoption via DNA, half-sibling, plagiarized author
- **Impersonal/Misc** (5): flight delay, escaping dog, wrong-recipient text, college closing, mixed medical records

Domain diversity matters: it averages out the topic-correlated component of activations, leaving emotion structure dominant.

### Stories: 5 per (emotion × topic) = 3000 total

For each (emotion, topic) pair, we asked Sonnet 4.5 (via the Claude Code Agent tool) to generate 5 short stories using Anthropic's exact prompt:

```
Write 5 different stories based on the following premise.
Topic: {topic}
The story should follow a character who is feeling {emotion}.
Format: <NEW STORY> [story] <NEW STORY> [story] etc.
Mix of third-person and first-person narration.
NEVER use the word '{emotion}' or direct synonyms.
Convey only through actions/body language/dialogue/thoughts/context.
```

Stories are 80-120 words each (some early waves up to 138). Format: JSONL, one story per line, schema:
```json
{"emotion": "...", "topic": "...", "story_index": 0-4, "perspective": "first|third", "text": "..."}
```

### Neutral dialogues: 50

For PCA denoising. Generated with Anthropic's verbatim neutral-dialogues prompt (their pages 60-61): two-character Person/AI exchanges on neutral topics (code, factual questions, work tasks, brainstorming) with strict no-emotion constraint.

## Extraction methodology

For each story:
1. Tokenize, forward pass through target model
2. Hook the residual stream at four sampled layers (~28%, 50%, 64%, 80% through the model)
3. Average activations across token positions starting from token 50 (Anthropic's protocol; skip the prompt boilerplate)
4. Save per-layer numpy arrays of shape `[3000, hidden_dim]`

Same procedure for the 50 neutral dialogues, used downstream for PCA denoising.

## Vector computation

For each emotion E and each layer:

```
v_E = mean(stories where emotion=E) − mean(all stories)
```

This subtracts the "story-ness" baseline shared across all emotions, leaving the emotion-specific direction.

### Denoising (Anthropic's procedure)

PCA on the 50 neutral-dialogue activations at each layer. Keep the top components covering 50% of variance (typically 9-10 components). Project these out of each emotion vector:

```
v_E_denoised = v_E − Σ proj_c(v_E)  for each top component c
```

This removes generic-text directions (e.g., "this is text in English") that aren't emotion-specific.

## Validation

Four checks per layer:

### 1. Cosine clustering
Compute all 20×20 pairwise cosines between emotion vectors. Within-cluster pairs (e.g., desperate↔grief-stricken) should have higher cosine than cross-cluster pairs. Pass: within-cluster mean exceeds cross-cluster mean by at least 0.05.

### 2. PCA valence axis on PC1
Run PCA on the 20-vector matrix. Positive emotions (joyful, content, proud) should separate clearly from negatives on the first principal component. Pass: |mean(pos PC1) − mean(neg PC1)| > 1.0.

### 3. Cross-layer stability
Compute cosine matrices at all sampled layers. Pairwise correlate them (Pearson on upper triangles). Pass: mean pairwise correlation > 0.7.

### 4. (Optional) Implicit-emotion scenarios
Construct ~8 scenarios that evoke emotions without naming them (Anthropic's Table 2 style). Project their activations onto each emotion vector. Pass: the scenario evokes its target emotion vector strongly. *(Not yet implemented in `validate.py`.)*

Audit gate: pass at least 3 of 4. Fail at any layer triggers debug.

## Honest limitations

These are real and worth knowing about:

1. **Generator template repetition.** Sonnet has go-to patterns per emotion (e.g., desperate→phone-calling loops, furious→controlled stillness). Within an emotion, narrative templates recur. This inflates within-cluster cosine similarities. Real but expected.

2. **Cultural register narrow.** All stories read as contemporary American middle-class. No diversity in era, culture, class, or socioeconomic context.

3. **Topic skew toward negative life events.** Positive anchor stories (joyful/content/proud) are positive *reframes* of negative topics (relief, vindication, schadenfreude, freedom-from-constraint). This creates a structural difference between positive-anchor and negative-emotion stories — they're not just different valences, they're different in their topic-emotion alignment. May appear as a confound in PCA.

4. **Word-count drift.** Wave 1 (desperate, grief-stricken, ashamed, lonely, furious) had stories 90-138 words. Later waves tightened to 80-120. Minor inconsistency across files.

5. **Perspective splits drifted.** 14 emotions hit 75/75 first/third-person. 6 came in at 90/60. Both perspectives present, slight skew.

6. **Two files needed post-hoc cleanup.**
   - `panicked.jsonl`: agent emitted stray `</text>` XML tags. Fixed via string replacement.
   - `terrified.jsonl` and `weary.jsonl`: each had one duplicate row. Deduplicated.

7. **Open-weight scale gap.** Reference results are on Qwen2.5-7B-Instruct (28 layers, 3584 hidden dim). Anthropic worked on Sonnet 4.5 (closed). Structure may be noisier or coarser at our scale.

## What the methodology doesn't claim

- **The model "feels" anything.** Emotion vectors are functional patterns of activation that correlate with emotion concepts. They are not subjective experience claims.
- **Generality across all language models.** Methodology works on multiple architectures (we test Qwen, Llama, Phi, Gemma) but specific magnitudes vary.
- **Causal interventions.** This repo extracts and validates the existence of vectors. Anthropic's full paper additionally demonstrates causal effects via steering. We do not currently include steering experiments.

## Hardware

Reference results were extracted on a Mac mini M4 Pro (24GB unified memory, 16 GPU cores). Model loaded in fp16 on MPS. See `docs/HARDWARE_NOTES.md` for Mac-specific gotchas (MPS allocator warmup workaround, device_map quirks, checkpointing pattern).

## Time budget

For a 7B-class model on M4 Pro:
- Extraction: ~70 min for 3000 stories + 50 dialogues
- Vector computation: ~30 sec
- Validation: ~1-2 min including plot generation
- Total: ~75 min wall clock

For a 14B-class model (4-bit quantized): roughly 1.5×.
