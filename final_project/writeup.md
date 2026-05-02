# Dynamic Multimodal Routing with Noise-Augmented Training: A Study on CMU-MOSEI

**Course Final Project Report**

---

## Abstract

Multimodal models that dynamically route inputs to cheaper or richer processing branches offer a promising path to efficiency-robustness trade-offs. We evaluate a two-branch hard-gate transformer, DynMMNetV2, on CMU-MOSEI sentiment analysis, augmented with a noise-augmentation training strategy (DynMMNetV2Noisy) hypothesized to teach the gate to prefer the multimodal branch when text is unreliable. The main result is negative: noise-augmented routing does not improve binary sentiment accuracy under test-time text corruption. We trace this failure to a dataset-level property — MOSEI's sentiment signal is overwhelmingly text-dominated, with audio and vision jointly contributing less than one accuracy point above text-alone, and an audio-vision-only baseline achieving only 69.5% Acc-2 — precisely the performance floor of text-corrupted multimodal models. The failure is therefore structural, not algorithmic. A partial positive result emerges in fine-grained metrics: noise training consistently improves regression Pearson correlation (9/9 noisy training conditions) and Acc-7 (8/9 conditions) under test-time noise, suggesting the method does meaningful work that binary accuracy cannot reveal.

---

## 1. Introduction

Real-world multimodal systems must contend with degraded or missing modalities: a video may have corrupted audio, a live transcription may introduce errors, or sensor noise may corrupt visual features. Models trained under clean conditions are ill-suited for these scenarios, and the challenge of building robust multimodal systems has received growing attention. One appealing design principle is dynamic routing: rather than always applying the same computation to every input, a learned gate selects a processing branch suited to the sample at hand. If the gate can detect when a particular modality is unreliable, it can route to a branch that does not depend on that modality, achieving both computational efficiency and robustness.

The DynMM framework (Xue and Marculescu, 2023) instantiates this principle with a differentiable hard gate trained end-to-end alongside two branches — one cheap (text-only) and one expensive (full multimodal). The gate is regularized to prefer the cheap branch via a FLOP penalty, producing an explicit accuracy-efficiency trade-off. We extend this framework with a noise-augmentation strategy: during training, text inputs are randomly corrupted with Gaussian noise on a fraction of samples, with the hypothesis that this teaches the gate to recognize noisy text and reroute to the multimodal branch, which can draw on vision and audio to compensate.

We evaluate this hypothesis on CMU-MOSEI, a large-scale multimodal sentiment and emotion dataset that is widely used as a benchmark for multimodal learning. Our experimental design is systematic: we train a 3×4 grid over FLOP regularization strength and noise probability, evaluate under three levels of test-time text noise, and conduct targeted ablations that isolate the effect of the gate, the individual branches, and the dataset's modality structure.

The central finding is that noise-augmented routing does not reliably improve binary sentiment accuracy on MOSEI. Through a sequence of diagnostic experiments, we establish that this is not a failure of the routing mechanism per se — the gate does respond to noise in the expected direction — but a consequence of MOSEI's text-dominated sentiment signal. We demonstrate empirically that an audio-vision-only model achieves 69.5% binary accuracy and a Pearson-equivalent correlation of 0.073, exactly matching the performance of text-corrupted multimodal models. There is no usable signal in the non-text modalities for the gate to exploit.

Despite this negative result, noise-augmented training produces a consistent improvement in regression-quality metrics (Pearson correlation, MAE, Acc-7) under noise, suggesting the method captures real structure that binary accuracy obscures. This mismatch between method effect and benchmark metric is itself an important finding.

**Contributions.** (1) A systematic sweep over regularization and noise probability, with multi-seed evaluation at multiple noise levels. (2) A per-expert ablation isolating the gate's contribution from each branch's standalone performance. (3) An audio-vision-only diagnostic establishing the ceiling for text-free routing. (4) A multi-metric evaluation using proper MOSEI regression metrics that reveals the partial success of noise training invisible to binary accuracy.

---

## 2. Background

### Dynamic Routing in Multimodal Models

Dynamic neural networks adapt their computation at inference time based on the input, offering a fundamentally different efficiency-accuracy trade-off than static pruning or distillation. In the multimodal setting, DynMM (Xue and Marculescu, 2023) proposes training multiple branches of increasing capacity and a lightweight gate network that selects among them per sample. The gate is trained with a straight-through Gumbel-softmax estimator, enabling discrete routing decisions to be learned end-to-end. A FLOP regularization term penalizes use of expensive branches, pushing the gate toward the cheap branch unless accuracy suffers. DynMM demonstrated that on AV-MNIST and CMU-MOSEI, routing can match or exceed full-multimodal accuracy at a fraction of the average compute.

### CMU-MOSEI

CMU-MOSEI (Zadeh et al., 2018) is a large-scale dataset of 23,453 annotated sentence-level video clips from YouTube, labeled with sentiment intensity on a [-3, 3] scale. The three modalities — language (GloVe word vectors, 300-dim), audio (COVAREP features, 74-dim), and vision (Facet features, 35-dim) — are pre-aligned at the word level. Binary sentiment accuracy (positive/negative), Acc-7 (rounded to seven classes), MAE, and regression Pearson correlation are the standard evaluation metrics. MOSEI is widely known to be text-dominated: prior work consistently reports that unimodal text baselines are competitive with multimodal models, and modality ablations show audio and vision contribute marginal independent gains.

### Noise Augmentation for Robustness

Adding noise during training to improve test-time robustness is a well-established technique. In the unimodal setting, dropout, input perturbation, and data augmentation have all been shown to improve robustness to distribution shift. In the multimodal setting, the interaction between modalities complicates this picture: corrupting one modality may simply shift reliance to another, or it may hurt performance if the model has learned tight cross-modal correlations. Noise augmentation as a gate-training signal — where corrupted inputs during training teach the gate to route away from a corrupted-modality branch — is a natural extension, but its effectiveness depends critically on whether the alternative branch can actually compensate.

---

## 3. Method

### DynMMNetV2 Architecture

DynMMNetV2 is a two-branch transformer model with a learned hard gate. The two processing branches are:

**E1 (text-only):** A single Transformer encoder that maps the 300-dimensional text input to a 120-dimensional representation, followed by a two-layer MLP producing a scalar sentiment prediction.

**E2 (multimodal late fusion):** Three parallel Transformer encoders — vision (35→60 dim), audio (74→120 dim), and text (300→120 dim) — whose outputs are concatenated into a 300-dimensional joint representation and passed through a two-layer MLP head.

**Gate network:** A lightweight Transformer that receives the concatenation of all three raw modality inputs and produces a two-dimensional logit vector. At training time, routing is performed via straight-through Gumbel-softmax, which allows gradients to flow through the discrete routing decision. At inference time, the hard argmax is used. Each sample is routed exclusively to either E1 or E2; there is no interpolation or mixture.

**FLOP regularizer:** The training loss includes a regularization term weighted by `reg` that penalizes the proportion of samples routed to the more expensive E2 branch. This term provides the only incentive for the gate to use E1, since E2 always has access to more information. Varying `reg` across {0.001, 0.01, 0.1} produces dramatically different routing distributions — from 86% E2 usage at `reg=0.001` to 0% E2 usage at `reg=0.1` — while the accuracy impact is surprisingly small.

### Noise Augmentation: DynMMNetV2Noisy

DynMMNetV2Noisy is a subclass of DynMMNetV2 with a single modification to the forward pass: with probability `noise_prob`, the text input is replaced with text + Gaussian noise (σ=1.0) before being passed to both the gate and the E1/E2 branches. The hypothesis is that this trains the gate to associate corrupted text features with a signal favoring E2, which can partially compensate via audio and vision. No parameters change relative to the clean model; DynMMNetV2Noisy checkpoints are fully compatible with DynMMNetV2 for evaluation purposes (the clean-trained backfill diagnostic exploits this).

### Training Setup

All models are trained for 30 epochs with the AdamW optimizer, learning rate 1e-4, batch size 32, on CMU-MOSEI's standard train/val/test split. The sweep covers `reg ∈ {0.001, 0.01, 0.1}` × `noise_prob ∈ {0.0, 0.25, 0.5, 0.75}`, with 3 random seeds per noisy cell and 1 per clean. Checkpoints are selected by validation loss.

### Evaluation Setup

Test-time robustness is evaluated by injecting independent Gaussian noise (σ ∈ {0.3, 0.5, 1.0}) into each modality separately. The per-expert ablation forces the gate to route all samples to either E1 or E2 regardless of its learned preference (`infer_mode ∈ {adaptive, E1-only, E2-only}`), isolating each branch's standalone performance from the gate's routing policy. The multi-metric evaluation computes the five standard MOSEI metrics: binary Acc-2, five-class Acc-5, seven-class Acc-7, MAE, and regression Pearson correlation (continuous predictions against continuous labels). Older evaluations used phi coefficient (Pearson of binary labels vs. binary predictions) as a proxy for correlation; Table 8 uses the proper regression metric, which yields substantially higher values (≈0.674 vs. ≈0.492 clean) while preserving relative ordering.

### Audio-Vision Diagnostic

To establish the ceiling for non-text routing, we train an audio-vision-only model (AVOnlyMMDL) from scratch on MOSEI. This model is architecturally identical to E2 with the text encoder removed: vision Transformer(35→60) + audio Transformer(74→120) + Concat + MLP(180→128→1). It is trained under identical hyperparameters to provide a direct comparison.

---

## 4. Results

### Finding 0: Routing is Adjustable, Accuracy is Not

The FLOP regularizer successfully controls the routing distribution: `reg=0.001` routes 85.7% of samples to E2, `reg=0.01` routes 45.1%, and `reg=0.1` routes 0%. Despite this 2.2× difference in average FLOPs, the accuracy span is under one percentage point.

**Table 1: Clean-trained DynMMNetV2 across regularization strengths (np=0.0, proper MOSEI metrics)**

| reg | E2 ratio | Acc-2 | Acc-7 | MAE | Corr |
|------|----------|-------|-------|-----|------|
| 0.001 | 0.857 | 0.784 | 0.504 | 0.602 | 0.675 |
| 0.01 | 0.451 | 0.778 | 0.507 | 0.606 | 0.674 |
| 0.1 | 0.000 | 0.776 | 0.498 | 0.614 | 0.665 |

The 0.8 pp accuracy gap across the full routing spectrum is the first indication that MOSEI is text-dominated: routing to text-only (E1) essentially replicates multimodal performance.

### Finding 1: E2 Consistently Outperforms E1

The per-expert ablation with `reg=0.01` and `noise_prob=0.5` confirms that E2 is strictly better than E1 on clean data. The gate is the only mechanism forcing the model to accept the accuracy penalty of E1 routing.

**Table 2: Per-expert performance, reg=0.01, np=0.5 checkpoint, clean test data**

| Inference Mode | Acc-2 |
|---------------|-------|
| Adaptive (gate) | 0.761 |
| E1-only | 0.752 |
| E2-only | 0.788 |

The 3.6 pp gap between E1 and E2 represents the accuracy cost of text-only routing. Adaptive mode sits between them, reflecting the mixed routing policy at `reg=0.01`.

### Findings 2–4: The Gate Responds but Cannot Help

Under test-time text corruption (σ=1.0), the gate increases its E2 preference as expected. However, this response does not translate into accuracy recovery. All three modes degrade similarly because E2 also consumes text — the corrupted modality contaminates both branches.

**Table 3: Per-expert performance under heavy text noise (σ=1.0)**

| Inference Mode | Acc-2 | Corr |
|---------------|-------|------|
| Adaptive | 0.700 | 0.075 |
| E1-only | 0.688 | 0.064 |
| E2-only | 0.707 | 0.075 |

The 1.9 pp gap between E1 and E2 under σ=1.0 (compared to 3.6 pp clean) shows partial modality redundancy in E2, but the absolute recovery is minimal. Critically, adaptive mode nearly matches E2-only, confirming that the gate successfully identifies the better branch — there simply is no good option available.

Vision and audio noise, by contrast, have nearly no effect on any mode's accuracy, a consequence of MOSEI's text dominance rather than the gate's design.

### Finding 5: The Audio-Vision Ceiling

The audio-vision-only baseline makes the text-dominance problem quantitative. Trained from scratch on MOSEI with no text access, AVOnlyMMDL achieves 69.5% binary accuracy and a Pearson-equivalent correlation of 0.073. This model converges in 14 epochs (~100 seconds of training time). The text-corrupted multimodal model (E2, σ=1.0) achieves approximately 70.7% Acc-2 and 0.075 correlation — barely above the audio-vision ceiling. The non-text modalities in MOSEI provide almost no independent sentiment signal. Any routing policy that relies on audio-vision to compensate for text corruption is working from a 69.5% ceiling, indistinguishable within noise from the degraded performance of text-using models.

### Finding 6: Noise Training Improves Regression Quality

Despite the binary accuracy results, noise-augmented training (np=0.5) produces consistent improvements in fine-grained metrics compared to clean-trained models under test-time noise. Comparing np=0.5 to np=0.0 across all nine noisy evaluation conditions (3 reg × 3 noise levels):

- **MAE**: improved in 9/9 conditions (lower is better)
- **Regression Corr**: improved in 9/9 conditions
- **Acc-7**: improved in 8/9 conditions
- **Acc-5**: improved in 8/9 conditions
- **Acc-2**: improved in only 4/9 conditions

At `reg=0.001`, σ=0.3, the representative deltas are ΔAcc-2=+0.017, ΔAcc-7=+0.024, ΔCorr=+0.132. The +0.132 shift in regression Pearson under mild noise is a substantial improvement in continuous prediction quality. Noise training is doing real work — it is teaching the model to produce better-calibrated continuous sentiment estimates under corruption — but this benefit is largely invisible to binary accuracy because the decision boundary is already well-placed.

---

## 5. Discussion

### Why the Gate Works but Cannot Help on MOSEI

The routing mechanism functions correctly at a mechanical level. The gate learns to associate corrupted text with E2 preference, the FLOP regularizer cleanly controls routing distribution, and E2 outperforms E1 across all conditions. The problem is informational: routing to E2 is only useful if E2 can recover performance from non-text modalities, and MOSEI's audio-vision features cannot support that recovery. The audio-vision ceiling (69.5% Acc-2) is established by direct experiment, not assumed. This is a dataset-level constraint that no routing policy can circumvent within the two-branch DynMMNetV2 framework.

A second structural issue compounds the first. Even if MOSEI had rich audio-vision sentiment signal, the current E1 branch is text-only, meaning there is no branch available that avoids text corruption entirely. The gate can route away from text-dominated computation toward text-augmented computation (E2), but it cannot route to a text-free computation. A three-branch design — E1 (text-only), E2 (multimodal), E3 (audio-vision-only) — would be the minimal architecture to make text-noise routing semantically meaningful. On MOSEI this still would not help due to the informational ceiling, but the architecture would at least have a sensible target for the corrupted-text case.

### Binary Accuracy as the Wrong Metric

The Acc-2 vs. MAE/Corr divergence (4/9 vs. 9/9 conditions where noise training improves) reveals a fundamental mismatch. Binary sentiment accuracy collapses a continuous sentiment score to a single bit; a model can improve its continuous prediction quality substantially without crossing enough samples over the decision boundary to register in Acc-2. MOSEI's sentiment distribution is not uniformly distributed around zero — there is a preponderance of mildly positive samples — which means the decision boundary is in a region of relatively low density, and continuous improvements to scores on the correct side of the boundary are invisible to binary accuracy.

The multi-metric evaluation was specifically designed to detect this pattern, and it succeeded. Noise-augmented training demonstrably improves regression quality under corruption: models trained with np=0.5 produce continuous sentiment predictions that are better calibrated (lower MAE) and more correlated with ground truth (higher Pearson) than clean-trained models when text is corrupted. This is the hypothesis partially validated: noise training teaches the model something useful. The binary benchmark simply cannot see it.

### What Would Make This Hypothesis Work

The noise-augmented routing hypothesis is architecturally sound; it requires a dataset where non-text modalities carry independent, meaningful sentiment signal. AVMNIST, where visual and audio digit representations are independently informative, is a natural testbed — routing from audio-only to audiovisual when audio is corrupted would have a genuine payoff because the visual branch can recover the label. Multimodal datasets designed for robustness research, or real-world datasets with stronger acoustic or visual sentiment cues (e.g., acted emotion datasets), would also be candidate environments. The key requirement is that the audio-vision-only ceiling should be substantially above the corrupted-text floor — a condition MOSEI spectacularly fails to meet.

---

## 6. Conclusion

We evaluated noise-augmented dynamic routing on CMU-MOSEI sentiment analysis, systematically varying FLOP regularization strength and noise probability across a 3×4 sweep with multi-seed evaluation and test-time noise injection. The hypothesis — that training with corrupted text teaches the gate to improve robustness by routing to the multimodal branch — is not confirmed by binary accuracy. The gate responds to noise correctly, and routing to the multimodal branch is modestly better than text-only even under corruption, but the absolute gains are small and inconsistent because MOSEI's audio-vision features carry negligible independent sentiment signal. The audio-vision-only diagnostic (69.5% Acc-2, 0.073 correlation) establishes the ceiling that makes routing recovery impossible: it matches the floor of text-corrupted multimodal models. This is a dataset-level negative result, not an algorithmic one.

Noise-augmented training does show consistent benefit in regression quality metrics (MAE, Pearson correlation, Acc-7), confirming it captures real structure invisible to binary accuracy. The method is partially validated in a domain binary benchmarks cannot see. Future work should evaluate this approach on datasets with genuinely balanced multimodal sentiment signal, and should consider architectures with a text-free branch to give the gate a meaningful routing target under text corruption.

---

## References

Xue, B. and Marculescu, R. (2023). Dynamic Multimodal Fusion. *arXiv preprint arXiv:2204.00102*.

Zadeh, A., Liang, P. P., Poria, S., Cambria, E., and Morency, L. (2018). Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. *Proceedings of ACL 2018*.

Liang, P. P., Liu, Z., Zadeh, A., and Morency, L. (2018). Multimodal Language Analysis with Recurrent Multistage Fusion. *Proceedings of EMNLP 2018*.

Gumbel, E. J. (1954). Statistical Theory of Extreme Values and Some Practical Applications. *Applied Mathematics Series*, 33. National Bureau of Standards.

Jang, E., Gu, S., and Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. *Proceedings of ICLR 2017*.
