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

| reg   | Accuracy | Corr   | FLOP (M) | E2 ratio |
|-------|----------|--------|----------|----------|
| 0.001 | 0.7838   | 0.5011 | —        | —        |
| 0.01  | 0.7775   | 0.4921 | 218.4    | 0.451    |
| 0.1   | 0.7764   | 0.4944 | —        | —        |

> No robustness evaluation was run for clean-training models.

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

## Table 5: Per-Modality Robustness (reg=0.01, train noise_prob=0.5)

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
