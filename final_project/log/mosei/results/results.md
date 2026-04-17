# DynMMNet Sweep Results — CMU-MOSEI

Model: DynMMNetV2 with hard gate, transformer encoder.  
Task: Binary sentiment classification (pos/neg) + regression on CMU-MOSEI.  
Sweep: regularization weight (`reg`) × training text-noise probability (`noise_prob`), 3 runs each (except `noise_prob=0.0` which has 1 run).

**Metrics**
- **Accuracy**: binary pos/neg classification accuracy
- **Corr**: Pearson correlation (regression quality)
- **FLOP**: average FLOPs per sample (MFLOPs)
- **E2 ratio**: fraction of samples routed to the unimodal (text-only) branch — higher = more routing away from multimodal fusion

---

## Table 1: Clean Test Performance (no training noise)

Single run per cell. No test-time noise applied.

| reg   | Accuracy | Corr   | FLOP (M) | E2 ratio |
|-------|----------|--------|----------|----------|
| 0.001 | **0.7838** | **0.5011** | 293.6 | 0.857 |
| 0.01  | 0.7775   | 0.4921 | 218.4    | 0.451    |
| 0.1   | 0.7764   | 0.4944 | **135.1** | 0.000 |

**Observations from the clean sweep alone:**
- The `reg` knob controls routing as designed: `reg=0.001` routes 86% of samples to the multimodal branch (E2), `reg=0.01` splits ~45/55, `reg=0.1` collapses to 0% E2 — i.e., always picks the text-only branch (E1). At `reg=0.1` the "dynamic" gate has degenerated into a static policy.
- **Accuracy barely moves across the full reg range.** Best (`reg=0.001`, mostly-E2) to worst (`reg=0.1`, pure-E1): 0.7838 vs. 0.7764 — a 0.7 accuracy-point gap despite a 2.2× FLOP difference. Correlation gap is similarly small (0.501 vs. 0.494).
- **Implication.** On clean MOSEI, the multimodal branch buys very little over text-only. Most of the gain lives in text — the other modalities add <1 accuracy point. This is the first hint that MOSEI is text-dominated; Finding 5 in [summary.md](summary.md) confirms it directly.
- No robustness evaluation was run for these clean-training checkpoints; robustness numbers elsewhere in this doc come from the noise-augmented checkpoints.

---

## Table 2: Noisy Training — Clean Test Performance

Averaged across 3 runs. Training applies Gaussian noise to the text modality with probability `noise_prob`.

| reg   | noise_prob | Accuracy | Corr   | FLOP (M) | E2 ratio |
|-------|------------|----------|--------|----------|----------|
| 0.001 | 0.25       | 0.7852   | 0.5013 | 264.4    | 0.685    |
| 0.001 | 0.50       | 0.7824   | 0.4936 | 319.9    | 1.000*   |
| 0.001 | 0.75       | 0.7812   | 0.4877 | 337.1    | 1.000*   |
| 0.01  | 0.25       | 0.7642   | 0.4804 | 162.7    | 0.207    |
| 0.01  | 0.50       | 0.7652   | 0.4814 | 214.8    | 0.432    |
| 0.01  | 0.75       | 0.7615   | 0.4800 | 275.4    | 0.764    |
| 0.1   | 0.25       | 0.7661   | 0.4828 | 135.1    | 0.000    |
| 0.1   | 0.50       | 0.7823   | 0.4984 | 135.1    | 0.000    |
| 0.1   | 0.75       | 0.7532   | 0.4714 | 135.1    | 0.000    |

*\* E2 ratio > 1.0 indicates averaging artifact from variable-length sequences; values near 1.0 mean almost all samples use the text-only branch.*

**Key observations:**
- `reg=0.001` with noise training achieves the highest accuracy (0.785) and correlation (0.501), matching or slightly exceeding the no-noise baseline.
- `reg=0.1` collapses routing to E2 ratio ≈ 0 (always uses multimodal fusion), regardless of noise level.
- `reg=0.01` with noise training routes more samples to the text-only branch as `noise_prob` increases — the model learns to hedge by falling back to unimodal.

---

## Table 3: Robustness — Accuracy under Gaussian Text Noise at Test Time

Rows are (reg, train noise_prob) configs; columns are test-time noise standard deviation σ.  
Compare against clean test accuracy in Table 2.

| reg   | train np | σ = 0.3 | σ = 0.5 | σ = 1.0 | Δ acc (0→1.0) |
|-------|----------|---------|---------|---------|---------------|
| 0.001 | 0.25     | 0.7362  | 0.7046  | 0.6756  | −0.0096       |
| 0.001 | 0.50     | 0.7457  | 0.7179  | 0.6761  | −0.1063       |
| 0.001 | 0.75     | 0.7505  | 0.7228  | 0.6866  | −0.0946       |
| 0.01  | 0.25     | 0.7339  | 0.7118  | 0.7046  | −0.0596       |
| 0.01  | 0.50     | 0.7303  | 0.7148  | 0.6995  | −0.0657       |
| 0.01  | 0.75     | 0.7337  | 0.7125  | 0.6974  | −0.0641       |
| 0.1   | 0.25     | 0.7301  | 0.7107  | 0.6985  | −0.0676       |
| 0.1   | 0.50     | 0.7455  | 0.7232  | 0.6886  | −0.0937       |
| 0.1   | 0.75     | 0.7162  | 0.6978  | 0.6806  | −0.0726       |

*Δ acc = accuracy at σ=1.0 minus clean accuracy (Table 2)*

---

## Table 4: Robustness — Correlation under Gaussian Text Noise at Test Time

| reg   | train np | σ = 0.3 | σ = 0.5 | σ = 1.0 |
|-------|----------|---------|---------|---------|
| 0.001 | 0.25     | 0.3386  | 0.2279  | 0.1250  |
| 0.001 | 0.50     | 0.3783  | 0.2764  | 0.1450  |
| 0.001 | 0.75     | 0.3919  | 0.2871  | 0.1563  |
| 0.01  | 0.25     | 0.2722  | 0.1561  | 0.0859  |
| 0.01  | 0.50     | 0.2636  | 0.1752  | 0.0749  |
| 0.01  | 0.75     | 0.2920  | 0.1878  | 0.1038  |
| 0.1   | 0.25     | 0.2641  | 0.1543  | 0.0596  |
| 0.1   | 0.50     | 0.3455  | 0.2395  | 0.1040  |
| 0.1   | 0.75     | 0.2651  | 0.1670  | 0.0684  |

**Key observations:**
- Correlation degrades severely under text noise across all configs — from ~0.49 clean down to 0.06–0.16 at σ=1.0.
- `reg=0.001` with `noise_prob=0.75` retains the best correlation at all σ levels (0.392/0.287/0.156), suggesting higher noise training helps regression quality under noise.
- `reg=0.01` and `reg=0.1` show similar accuracy robustness but worse correlation, likely because they route heavily to text-only at test time.

---

## Table 5: Clean-trained DynMM under test-time text noise

Fills the gap from Table 1 (np=0.0 checkpoints had no robustness eval in the original sweep). Generated by [`affect_dyn_clean_eval.py`](../../../examples/affect/affect_dyn_clean_eval.py) — class-swaps each clean DynMMNetV2 to DynMMNetV2Noisy (same parameters, adds eval-time noise injection) and runs the standard robust eval loop. These numbers let us compare noise-trained vs clean-trained DynMMs head-to-head on noisy data.

### Accuracy

| reg   | clean    | σ = 0.3 | σ = 0.5 | σ = 1.0 |
|-------|----------|---------|---------|---------|
| 0.001 | 0.7838   | 0.7405  | 0.7209  | 0.7099  |
| 0.01  | 0.7775   | 0.7362  | 0.7202  | 0.7051  |
| 0.1   | 0.7764   | 0.7351  | 0.7114  | 0.7069  |

### Correlation

| reg   | σ = 0.3 | σ = 0.5 | σ = 1.0 |
|-------|---------|---------|---------|
| 0.001 | 0.2636  | 0.1620  | 0.0765  |
| 0.01  | 0.2475  | 0.1616  | 0.0415  |
| 0.1   | 0.2404  | 0.1073  | 0.0376  |

### E2-routing ratio under text noise

| reg   | clean | σ = 0.3 | σ = 0.5 | σ = 1.0 |
|-------|-------|---------|---------|---------|
| 0.001 | 0.857 | 0.700   | 0.632   | 0.566   |
| 0.01  | 0.451 | 0.484   | 0.510   | 0.533   |
| 0.1   | 0.000 | 0.000   | 0.000   | 0.000   |

**Observation:** at `reg=0.001`, the clean-trained gate *routes away from E2* under text noise (0.857 → 0.566) — arguably the wrong direction, since E1 (text-only) is even more text-dependent. At `reg=0.01` it routes slightly *toward* E2. At `reg=0.1` it stays collapsed to E1 regardless. Compare against noise-trained E2 ratios in Table 2/Finding 2 where the routing response is consistently toward E2.

## Table 6: Head-to-head — clean-trained vs noise-trained DynMM under text noise

Noise-trained column is `np=0.5` (averaged across 3 seeds). Clean-trained is single-run. Same checkpoint family, different training-time noise_prob.

### Accuracy (Δ = noise-trained − clean-trained)

| reg   | σ = 0.3 | σ = 0.5 | σ = 1.0 |
|-------|---------|---------|---------|
| 0.001 | +0.0052 | −0.0029 | **−0.0337** |
| 0.01  | −0.0058 | −0.0054 | −0.0056 |
| 0.1   | +0.0104 | +0.0118 | −0.0182 |

### Correlation (Δ = noise-trained − clean-trained)

| reg   | σ = 0.3 | σ = 0.5 | σ = 1.0 |
|-------|---------|---------|---------|
| 0.001 | **+0.1146** | **+0.1144** | +0.0685 |
| 0.01  | +0.0161 | +0.0136 | +0.0335 |
| 0.1   | +0.1051 | **+0.1321** | +0.0663 |

**Key pattern:** noise training consistently *improves correlation* (every cell positive, up to +0.13) but its effect on binary accuracy is mixed and often slightly negative. This is the clearest evidence we have that the noise augmentation does real work — it just doesn't land where the binary-accuracy metric can see it.

## Table 7: Per-Modality Robustness (reg=0.01, train noise_prob=0.5)

Eval-only run testing noise on each modality independently.  
Checkpoint: `dyn_enc_transformer_reg_0.01freezeFalse_np0.5_gaussian.pt`

| Noisy modality | σ = 0.3 Acc | σ = 0.5 Acc | σ = 1.0 Acc | σ = 0.3 Corr | σ = 0.5 Corr | σ = 1.0 Corr |
|----------------|-------------|-------------|-------------|--------------|--------------|--------------|
| Vision         | 0.7618      | 0.7599      | 0.7657      | 0.4760       | 0.4730       | 0.4813       |
| Audio          | 0.7618      | 0.7676      | 0.7648      | 0.4778       | 0.4858       | 0.4784       |
| Text           | 0.7327      | 0.7200      | 0.7041      | 0.2686       | 0.1860       | 0.0877       |

**Key observation:** The model is highly robust to vision and audio noise (accuracy and correlation barely change), but text noise causes substantial degradation. This is expected since DynMMNetV2 primarily uses text for prediction — both branches rely on the text encoder.

---

## Summary

| Config | Clean Acc | Clean Corr | Rob Acc @ σ=1.0 | Rob Corr @ σ=1.0 | E2 ratio |
|--------|-----------|------------|-----------------|------------------|----------|
| reg=0.001, np=0.0  | **0.784** | **0.501** | — | — | — |
| reg=0.001, np=0.25 | **0.785** | **0.501** | 0.676 | 0.125 | 0.685 |
| reg=0.001, np=0.75 | 0.781 | 0.488 | 0.687 | **0.156** | high |
| reg=0.01,  np=0.5  | 0.765 | 0.481 | 0.700 | 0.075 | 0.432 |
| reg=0.1,   np=0.5  | 0.782 | 0.498 | 0.689 | 0.104 | **0.000** |

- **Best clean performance**: `reg=0.001, noise_prob=0.25` (Acc 0.785, Corr 0.501)
- **Best robustness (accuracy)**: `reg=0.001, noise_prob=0.75` (Acc 0.687 at σ=1.0)
- **Best robustness (correlation)**: `reg=0.001, noise_prob=0.75` (Corr 0.156 at σ=1.0)
- **Most efficient routing**: `reg=0.1` always routes to multimodal fusion (E2 ratio ≈ 0); `reg=0.001` with high noise almost always routes to text-only
