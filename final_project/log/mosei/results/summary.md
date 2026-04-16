# Summary — DynMM Noise-Augmented Training on CMU-MOSEI

Companion to [results.md](results.md) (tables). This document narrates the findings from two experiments and argues that the results are negative in a *structurally explainable* way — not a tuning problem.

## TL;DR

We trained the two-branch DynMMNetV2 gate with per-sample text-channel Gaussian noise (probabilities 0.25 / 0.5 / 0.75) across three FLOP-regularization weights (0.001 / 0.01 / 0.1), then evaluated each trained model under test-time Gaussian corruption — first on text alone, then per-modality (vision / audio / text) with the gate forced into each branch. The adaptive gate **never beats the static multimodal expert (E2) on any cell**, clean or noisy. The reason is structural: both experts consume text, so no routing choice exists that escapes text corruption, which is the exact regime the noise training was designed to solve.

## Experiments

### 1. Training sweep — [`affect_dyn_sweep.sh`](../../../examples/affect/affect_dyn_sweep.sh)

3 × 4 grid: `reg ∈ {0.001, 0.01, 0.1}` × `text_noise_prob ∈ {0.0, 0.25, 0.5, 0.75}`, 3 runs per noisy cell, 1 run per clean cell. Each trained model was evaluated on clean test and under Gaussian text noise at σ ∈ {0.3, 0.5, 1.0}. Full tables in [results.md](results.md) §2–§4.

### 2. Per-modality / per-expert eval — [`affect_dyn_robust_eval.sh`](../../../examples/affect/affect_dyn_robust_eval.sh)

Eval-only sweep on the `reg=0.01` checkpoints, crossing `noise_prob ∈ {0.25, 0.5, 0.75}` with `infer_mode ∈ {0:adaptive, 1:E1 text-only, 2:E2 multimodal}`. Per-modality noise applied to vision, audio, or text independently. The SLURM array is sized `0-11` but only 9 cells are valid — tasks 9-11 silently fall through because `INFER_MODES[3]` is empty (see "Script bug" below). The 9 valid configs all completed.

## Findings

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

So the gate *is* learning something — noise-augmented training is successfully activating the routing signal, which was the stated motivation in the [`DynMMNetV2Noisy`](../../../examples/affect/affect_dyn.py#L180-L186) docstring. But the redirection never closes the accuracy gap to a model that just always used E2. The dynamic decision costs accuracy and delivers no robustness benefit.

### Finding 3 — Robustness was free, not earned

Vision and audio noise degrade the model barely at all (E2 accuracy drops only ~0.005 from clean to σ=1.0). This is because **E2 has modality redundancy**: its late-fusion transformer can lean on the uncorrupted channels. The appearance of "vision/audio robustness" in the adaptive model is inherited from E2's redundancy, not produced by routing. Training noise and dynamic gating were both irrelevant to this outcome.

### Finding 4 — Under text noise there is no escape route

Text corruption is the hard case and the one the noise training targets. But text corruption hurts *both* branches: E1 is text-only (obviously), and E2 also consumes text as one of three fused modalities. At σ=1.0, text noise drops all three modes (Adaptive / E1 / E2) into a 0.688–0.707 band — a ~7-point drop from clean. No routing policy can recover, because no text-free expert exists.

## Why the results are negative

The project was set up on a hypothesis that turns out to be false for this model family:

> "If we teach the gate that text can be corrupted, it will route around corrupted text at inference time and be more robust than a fixed multimodal model."

This fails for three reasons, in order of importance:

1. **No text-free branch.** E1 = text-only, E2 = multimodal-including-text. The gate's action space does not contain a "skip text" option. Training on noisy text produces a gate that knows which *probability* of text corruption it's in, but it cannot route to a channel that wouldn't be affected, because no such channel exists.

2. **E2 is strictly better than E1 on clean data, by ~3 accuracy points.** The gate's only incentive to ever pick E1 is the FLOP regularizer. The moment the FLOP penalty is large enough to make E1 attractive (`reg ≥ 0.01`), it pulls the adaptive policy *away* from the more accurate branch on clean inputs — a persistent clean-accuracy tax that noise-aware routing has no way to repay, because it cannot do better than E2 even under noise.

3. **Modality redundancy already buys vision/audio robustness for free.** E2 is robust to vision/audio corruption without any routing help, because its late-fusion transformer averages across channels. The gate is solving a problem that doesn't need solving on two of three modalities, and cannot solve it on the third.

The FLOP regularizer is doing real work — it shrinks compute from ~320 MFLOP (E2-only) to 135–285 MFLOP depending on `reg` — but it is buying compute at the cost of accuracy, not buying robustness. If the research claim is purely about compute–accuracy tradeoffs, that story survives. If the claim is about **noise-augmented routing improving robustness**, the evidence here does not support it.

## Script bug to note

[`affect_dyn_robust_eval.sh:10`](../../../examples/affect/affect_dyn_robust_eval.sh#L10) sets `#SBATCH --array=0-11` but `NOISE_PROBS` only has 3 entries, not the 4 the comment at line 21 claims. Array IDs 9-11 index `INFER_MODES[3]` which is empty, so those three jobs run with `--infer-mode=` and produce nothing useful. Either reduce the array to `0-8` or add `0.0` back into `NOISE_PROBS` (which requires handling the no-noisy-checkpoint case). This did not affect the conclusions — all 9 valid cells ran.

## What would make the hypothesis testable

Three directions, in rough order of promise:

1. **Three-branch model with a text-free expert.** The commented-out `DynMMNet` (3-branch) in [`affect_dyn.py:36`](../../../examples/affect/affect_dyn.py#L36) uses separate visual/audio/text encoders. A variant with one multimodal branch + one audio+vision-only branch (no text) would give the gate a real escape route under text corruption. This is the single most direct fix.

2. **Asymmetric corruption during eval.** Corrupt modalities the experts consume differently (e.g., corrupt the visual channel only in E2; E1 ignores it). This isolates whether the gate can usefully redirect at all, independent of whether the expert set is adequate.

3. **Drop the FLOP regularizer in the robustness study.** If the claim is about robustness, not efficiency, the FLOP penalty is working against you. Without it, the gate could learn to pick E2 when text is corrupted *without paying an accuracy tax on clean inputs* — but then it's unclear what the gate contributes over always-E2. This clarifies whether there's any gate-specific benefit at all once compute is removed from the objective.

The cleanest writeup of the current results is: **noise-augmented training activates the DynMM gate in a measurable way, but the two-branch architecture with shared text dependence admits no routing policy that improves robustness over the static multimodal baseline.** The gate is learning the right *signal* and producing the wrong *action* — because the right action is not in its action space.
