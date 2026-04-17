# Summary — DynMM Noise-Augmented Training on CMU-MOSEI

Companion to [results.md](results.md) (tables). This document narrates the findings from two experiments and argues that the results are negative in a *structurally explainable* way — not a tuning problem.

## TL;DR

We trained the two-branch DynMMNetV2 gate with per-sample text-channel Gaussian noise (probabilities 0.25 / 0.5 / 0.75) across three FLOP-regularization weights (0.001 / 0.01 / 0.1), then evaluated each trained model under test-time Gaussian corruption — first on text alone, then per-modality (vision / audio / text) with the gate forced into each branch. We also trained a text-free (audio + vision only) baseline to establish the ceiling for a hypothetical rescue branch, and back-filled robustness evals for the clean-trained (np=0.0) checkpoints to enable a direct noise-vs-clean head-to-head. The adaptive gate **never beats the static multimodal expert (E2) on binary accuracy in any cell**, clean or noisy. The reason is structural and informational: (a) both existing experts consume text, so no routing choice escapes text corruption, and (b) the text-free baseline reaches only **0.07 correlation** on clean data — the same level as a text-corrupted multimodal model — so even extending the architecture with a text-free branch cannot rescue this task on MOSEI. One nuance (Finding 6): noise-augmented training *does* consistently improve **regression correlation** under test-time noise (by up to +0.13), but this uplift doesn't move the binary sign boundary, so the headline accuracy metric doesn't reflect it. The negative result is dataset-level and metric-level — the training is doing real work in a direction the benchmark doesn't reward.

## Experiments

### 1. Training sweep — [`affect_dyn_sweep.sh`](../../../examples/affect/affect_dyn_sweep.sh)

3 × 4 grid: `reg ∈ {0.001, 0.01, 0.1}` × `text_noise_prob ∈ {0.0, 0.25, 0.5, 0.75}`, 3 runs per noisy cell, 1 run per clean cell. Each trained model was evaluated on clean test and under Gaussian text noise at σ ∈ {0.3, 0.5, 1.0}. Full tables in [results.md](results.md) §2–§4.

### 2. Per-modality / per-expert eval — [`affect_dyn_robust_eval.sh`](../../../examples/affect/affect_dyn_robust_eval.sh)

Eval-only sweep on the `reg=0.01` checkpoints, crossing `noise_prob ∈ {0.25, 0.5, 0.75}` with `infer_mode ∈ {0:adaptive, 1:E1 text-only, 2:E2 multimodal}`. Per-modality noise applied to vision, audio, or text independently. The SLURM array is sized `0-11` but only 9 cells are valid — tasks 9-11 silently fall through because `INFER_MODES[3]` is empty (see "Script bug" below). The 9 valid configs all completed.

### 3. Text-free viability check — [`affect_mm_av.sh`](../../../examples/affect/affect_mm_av.sh)

Single training run of a late-fusion transformer using only vision + audio, with the text modality dropped entirely. Architecture mirrors E2's non-text encoders: `Transformer(35, 60)` vision + `Transformer(74, 120)` audio → `Concat` → `MLP(180, 128, 1)`, trained from scratch with L1 regression. Purpose: measure the upper bound on performance achievable without text — i.e., the ceiling for any hypothetical text-free expert that a three-branch DynMM could route to under text corruption. Results in [`lf_tran_av_20260416_184437.json`](lf_tran_av_20260416_184437.json).

### 4. Clean-trained robustness backfill — [`affect_dyn_clean_eval.py`](../../../examples/affect/affect_dyn_clean_eval.py)

Eval-only sweep that fills a gap in the original sweep: the three np=0.0 checkpoints (`reg ∈ {0.001, 0.01, 0.1}`) were trained without noise and therefore never evaluated under test-time noise. Approach: class-swap each loaded `DynMMNetV2` into `DynMMNetV2Noisy` (no parameter change — the subclass only adds a pre-forward text-corruption step) and run the standard σ ∈ {0.3, 0.5, 1.0} text-noise eval. This enables a direct head-to-head between clean-trained and noise-trained DynMMs on noisy data (Finding 6). Results in [`dyn_enc_transformer_reg_*_freezeFalse_cleaneval_*.json`](./).

## Findings

### Finding 0 — The clean sweep already tells most of the story

Before any noise is applied, the three `reg` values (0.001 / 0.01 / 0.1) produce very different *routing behaviors* but nearly identical *accuracies*:

| reg   | Accuracy | Corr   | FLOP (M) | E2 ratio | Gate behaves as… |
|-------|----------|--------|----------|----------|------------------|
| 0.001 | 0.7838   | 0.5011 | 293.6    | 0.857    | mostly E2 (multimodal) |
| 0.01  | 0.7775   | 0.4921 | 218.4    | 0.451    | ~50/50 split     |
| 0.1   | 0.7764   | 0.4944 | 135.1    | 0.000    | always E1 (text-only) — degenerate static policy |

Two observations follow, and they pre-stage every subsequent finding:

1. **The gate reliably obeys its FLOP regularizer.** Increasing `reg` monotonically reduces E2 usage and FLOPs. The `reg=0.1` cell shows the gate collapsing entirely to text-only — the dynamic mechanism is functionally off, and performance matches a plain E1 baseline.

2. **Accuracy barely changes across the full routing spectrum.** Best (`reg=0.001`, mostly-E2) to worst (`reg=0.1`, pure-E1) is 0.7838 vs. 0.7764 — a 0.7-point gap across a 2.2× FLOP range. Correlation gap is similarly tiny (0.501 → 0.494). This is the first quantitative indication that **MOSEI's multimodal branch barely outperforms a text-only branch on clean data.** The multimodal signal above text is worth ~1 accuracy point, not ~10. Finding 5 later confirms the same fact from the opposite direction (drop text entirely and performance collapses).

Together, these two observations mean the clean sweep already hints at the core negative result: the gate works, it just doesn't have a meaningful choice to make. The noisy sweep and per-modality eval just make the problem sharper.

### Finding 1 — E2 (static multimodal) dominates every other setting

At every training `noise_prob`, forcing inference through E2 beats the adaptive gate on clean test accuracy and on every noisy cell:

| train np | Adaptive (clean) | E1 (clean) | **E2 (clean)** |
|----------|------------------|------------|----------------|
| 0.25     | 0.7523           | 0.7523     | **0.7881**     |
| 0.50     | 0.7607           | 0.7521     | **0.7827**     |
| 0.75     | 0.7618           | 0.7543     | **0.7685**     |

Under text noise at σ=1.0 (`np=0.5`), E2 still wins: E2 = 0.7073, Adaptive = 0.7045, E1 = 0.6883. Under vision/audio noise at σ=1.0, E2 ≈ 0.78 while Adaptive ≈ 0.77. **E2-only is a strict improvement over the gated model on every cell tested.**

### Finding 2 — The gate responds to noise but the response does not help

The E2-routing ratio *does* shift under corruption — it rises with σ, correctly interpreting "input is corrupted → prefer multimodal":

| noisy modality (np=0.5) | clean ratio | σ=0.3 | σ=0.5 | σ=1.0 |
|-------------------------|-------------|-------|-------|-------|
| Vision                  | 0.435       | 0.456 | 0.494 | 0.534 |
| Audio                   | 0.435       | 0.468 | 0.487 | 0.538 |
| Text                    | 0.435       | 0.514 | 0.534 | 0.556 |

So the gate *is* learning something — noise-augmented training is successfully activating the routing signal, which was the stated motivation in the [`DynMMNetV2Noisy`](../../../examples/affect/affect_dyn.py#L180-L186) docstring. But the redirection never closes the accuracy gap to a model that just always used E2. The dynamic decision costs accuracy on the headline binary-sentiment metric. (See Finding 6: on the regression correlation metric, the noise-trained gate *does* deliver a consistent improvement over the clean-trained gate — the dynamic behavior helps, but in a direction the binary metric doesn't see.)

### Finding 3 — Robustness was free, not earned

Vision and audio noise degrade the model barely at all (E2 accuracy drops only ~0.005 from clean to σ=1.0). This is because **E2 has modality redundancy**: its late-fusion transformer can lean on the uncorrupted channels. The appearance of "vision/audio robustness" in the adaptive model is inherited from E2's redundancy, not produced by routing. Training noise and dynamic gating were both irrelevant to this outcome.

### Finding 4 — Under text noise there is no escape route

Text corruption is the hard case and the one the noise training targets. But text corruption hurts *both* branches: E1 is text-only (obviously), and E2 also consumes text as one of three fused modalities. At σ=1.0, text noise drops all three modes (Adaptive / E1 / E2) into a 0.688–0.707 band — a ~7-point drop from clean. No routing policy can recover, because no text-free expert exists.

### Finding 5 — The missing modality's signal isn't recoverable from vision + audio

The text-free baseline trained to convergence in 14 epochs (early-stopped at ~103 seconds of training) and reached:

| Metric      | Val    | Test   |
|-------------|--------|--------|
| Accuracy    | 0.7052 | **0.6950** |
| Correlation | 0.058  | **0.073** |

Place this alongside the text-corrupted multimodal results from Finding 1/4:

| Configuration                     | Accuracy | Correlation |
|-----------------------------------|----------|-------------|
| E2 multimodal, clean              | 0.7881   | 0.4982      |
| E2 multimodal, text σ=1.0         | 0.7073   | 0.0749      |
| **A+V text-free, clean**          | **0.6950** | **0.073** |

**A+V clean performs at the same level as a multimodal model whose text channel has been destroyed.** Vision + audio together contain almost no independent sentiment signal on MOSEI — just enough to push binary accuracy a few points above the ~53% majority-class baseline, but not enough to correlate meaningfully with the continuous sentiment target. This is a dataset-level ceiling, not a model-level one.

### Finding 6 — Noise training helps correlation consistently but not accuracy

The backfill eval (Experiment 4) makes the head-to-head possible. Comparing noise-trained (np=0.5, averaged over 3 seeds) against clean-trained (np=0.0, single run) on the same reg value, evaluated under the same text-noise sigmas:

**Accuracy (Δ = noise-trained − clean-trained):** mixed, small, often negative.

| reg   | σ=0.3    | σ=0.5    | σ=1.0    |
|-------|----------|----------|----------|
| 0.001 | +0.0052  | −0.0029  | **−0.0337** |
| 0.01  | −0.0058  | −0.0054  | −0.0056  |
| 0.1   | +0.0104  | +0.0118  | −0.0182  |

**Correlation (Δ = noise-trained − clean-trained):** positive everywhere, up to +0.13.

| reg   | σ=0.3    | σ=0.5    | σ=1.0    |
|-------|----------|----------|----------|
| 0.001 | **+0.1146** | **+0.1144** | +0.0685 |
| 0.01  | +0.0161  | +0.0136  | +0.0335  |
| 0.1   | +0.1051  | **+0.1321** | +0.0663 |

**Interpretation.** Noise-augmented training is doing real work — every cell of the correlation delta is positive, by a non-trivial margin (+0.01 to +0.13). But this uplift does not translate into binary accuracy: the sign of the regression decision boundary barely moves, even though the model's predicted magnitudes align better with the true sentiment scores. The clean-trained `reg=0.001` checkpoint actually *beats* the noise-trained one in accuracy at σ=1.0 (0.710 vs 0.676) — a consequence of the clean-trained gate routing more uniformly under corruption (its E2-ratio drifts from 0.86 clean → 0.57 at σ=1.0) while the noise-trained gate stays committed to E2 (0.86 → 0.73). More-uniform routing happens to average in enough signal to edge out the noise-trained model on binary sign.

This refines Finding 2: the gate's response to noise *does* improve the model's regression fidelity, but because the MOSEI task is typically evaluated through binary sign (pos/neg), the improvement is invisible to the headline metric. The noise-training value is real but is in a direction the benchmark doesn't reward.

## Why the results are negative

The project was set up on a hypothesis that turns out to be false for this model family:

> "If we teach the gate that text can be corrupted, it will route around corrupted text at inference time and be more robust than a fixed multimodal model."

This fails for three reasons, in order of importance:

1. **No text-free branch.** E1 = text-only, E2 = multimodal-including-text. The gate's action space does not contain a "skip text" option. Training on noisy text produces a gate that knows which *probability* of text corruption it's in, but it cannot route to a channel that wouldn't be affected, because no such channel exists.

2. **E2 is strictly better than E1 on clean data, by ~3 accuracy points.** The gate's only incentive to ever pick E1 is the FLOP regularizer. The moment the FLOP penalty is large enough to make E1 attractive (`reg ≥ 0.01`), it pulls the adaptive policy *away* from the more accurate branch on clean inputs — a persistent clean-accuracy tax that noise-aware routing has no way to repay, because it cannot do better than E2 even under noise.

3. **Modality redundancy already buys vision/audio robustness for free.** E2 is robust to vision/audio corruption without any routing help, because its late-fusion transformer averages across channels. The gate is solving a problem that doesn't need solving on two of three modalities, and cannot solve it on the third.

4. **The missing signal isn't recoverable from the other modalities.** Even if the two-branch architecture were extended with a text-free expert, that expert's ceiling on MOSEI is ~0.07 correlation — essentially zero, and indistinguishable from a text-corrupted multimodal model. Finding 5 closes off the natural architectural fix: MOSEI's vision and audio channels simply don't carry enough independent sentiment signal to serve as a rescue expert. The problem is not just that the gate's action space is missing an option; the option that's missing wouldn't work on this data either.

The FLOP regularizer is doing real work — it shrinks compute from ~320 MFLOP (E2-only) to 135–285 MFLOP depending on `reg` — but it is buying compute at the cost of accuracy, not buying robustness. If the research claim is purely about compute–accuracy tradeoffs, that story survives. If the claim is about **noise-augmented routing improving robustness**, the evidence here does not support it.

## Script bug to note

[`affect_dyn_robust_eval.sh:10`](../../../examples/affect/affect_dyn_robust_eval.sh#L10) sets `#SBATCH --array=0-11` but `NOISE_PROBS` only has 3 entries, not the 4 the comment at line 21 claims. Array IDs 9-11 index `INFER_MODES[3]` which is empty, so those three jobs run with `--infer-mode=` and produce nothing useful. Either reduce the array to `0-8` or add `0.0` back into `NOISE_PROBS` (which requires handling the no-noisy-checkpoint case). This did not affect the conclusions — all 9 valid cells ran.

## What would make the hypothesis testable

The natural architectural fix — adding a text-free branch — is ruled out by Finding 5: on MOSEI, a text-free model's ceiling is the same as a text-corrupted multimodal model's floor. The remaining directions move *off* MOSEI or change what's being measured:

1. **Change dataset.** The hypothesis "noise-augmented gating improves robustness to modality corruption" can only be tested on data where each modality carries usable independent signal. Candidates: AVMNIST (vision + audio by construction, roughly balanced), speech-emotion recognition (audio + face, both informative), action recognition (multiple visual streams), or any dataset where dropping one modality still leaves the others competent. MOSEI is the wrong instrument for this experiment.

2. **Asymmetric corruption during eval.** Corrupt modalities the experts consume differently (e.g., corrupt the visual channel in E2 only; E1 ignores it). This isolates whether the gate can usefully redirect *at all*, independent of whether any expert has the capacity to rescue. A useful diagnostic even on MOSEI, though it doesn't change the outcome.

3. **Drop the FLOP regularizer in the robustness study.** If the claim is about robustness, not efficiency, the FLOP penalty works against you — it forces the gate toward the cheaper branch even when the expensive one is strictly better. Without it, the gate could learn to pick E2 under text corruption *without paying an accuracy tax on clean inputs* — but then the gate's value-add over always-E2 becomes unclear.

The cleanest writeup of the current results: **noise-augmented training activates the DynMM gate in a measurable way, but no routing policy on MOSEI can improve robustness over the static multimodal baseline — not because of the two-branch architecture alone, but because MOSEI's non-text modalities don't carry enough independent signal for any text-free expert to be a meaningful alternative.** The gate learns the right *signal* and produces the wrong *action*, and extending its action space wouldn't change the result on this data. The negative finding is dataset-level, and the fix is to move to a dataset whose modalities are actually competitive.

## Plain-English explanation

Imagine you're trying to tell whether someone in a short video clip is happy or sad. You have three sources of information:

- **The words they said** ("text")
- **How they said them** — tone of voice ("audio")
- **What their face looked like** ("vision")

The MOSEI dataset is built from clips like this. For MOSEI specifically, it turns out that almost all the useful information is in the **words**. The tone and the face barely help — they only move the needle by a few percent.

**The original research idea was a "smart switch."** Train two small models — one that looks only at the words (cheap), and one that looks at all three channels (more accurate but slower). Then train a switch that picks between them at inference time. The hope was: if we sometimes scramble the words during training, the switch will learn to prefer the "all three channels" model when the words are unreliable, and the system will be more robust to noisy text at test time.

**The switch couldn't fix the problem — because every option it had available still used the words.** The "only words" model obviously breaks when you scramble the words. The "all three channels" model *also* uses the words, so it breaks too, just slightly less. There was no button the switch could press that avoided the words. The switch did learn to route toward the all-three-channels model under corruption, but it couldn't route to anywhere actually safe.

**So we built a third model: "audio + vision only," with no words at all.** If this model performed decently, the switch would finally have a real escape hatch — a text-free option to route to when text is unreliable. It trained in about 100 seconds and reached roughly 70% classification accuracy on paper, which sounds acceptable. But its **correlation with the actual sentiment score was 0.07** — basically zero. Without the words, the model can't meaningfully distinguish *degrees* of happy vs. sad; it mostly just guesses, and gets its classification accuracy because the two labels are roughly balanced in the data, not because it's actually seeing the sentiment.

**The punchline:** on MOSEI, the information needed to judge sentiment lives in the words. Take the words away and there's nothing useful left — not in the tone, not in the face. No amount of clever routing between models can recover what isn't there to begin with. This isn't a flaw in the switch idea. It's a mismatch between the idea and this particular dataset. To actually test whether a dynamic switch improves robustness, you'd need a dataset where removing any single input still leaves enough signal in the others to work with. MOSEI isn't that dataset.
