# Homework 2 — Multimodal Fusion and Alignment

## Overview

This assignment explores multimodal fusion techniques and contrastive alignment for digit classification on the **AV-MNIST** dataset (audio + visual MNIST), using the [MultiBench](https://github.com/pliang279/MultiBench) benchmark.

## What's Inside

### Unimodal Baselines
- Custom **AudioModel** (Conv2D-based) and **ImageModel** (Conv2D-based) trained independently on AV-MNIST with mixed-precision training
- Hyperparameter tuning (dropout, architecture depth) to maximize unimodal performance

### Fusion Techniques (implemented from scratch with einsum)
- **Early Fusion** — Outer product of modality representations via `einsum('bi,bj->bij')`
- **Late Fusion** — Independent modality encoders with decision-level combination
- **Tensor Fusion** — Full outer product interaction tensor
- **Low-Rank Tensor (LMF) Fusion** — Factorized approximation of tensor fusion for efficiency

### Fusion Analysis
- Comparative visualizations of parameter count, memory usage, and time-to-convergence across all fusion methods
- Discussion of unimodal vs. multimodal tradeoffs

### Contrastive Learning
- Implemented a general **contrastive learning model** (CLModel) with dual encoders and projection heads, trained with cross-entropy-based contrastive loss
- Applied to the SUPERB emotion recognition dataset (from HW1) for zero-shot classification
- Post-alignment visualizations showing where modality alignment succeeded and failed

### Reading
- Analysis of "Align Before Fuse" (Nagrani et al., 2021) and the Platonic Representation Hypothesis (Huh et al., 2024), with discussion of implications for audio-text emotion recognition

## Key Findings

- Image modality significantly outperformed audio on AV-MNIST; late fusion and tensor fusion both improved over unimodal baselines
- LMF achieved comparable accuracy to full tensor fusion with substantially fewer parameters and faster convergence
- Contrastive alignment worked well for high-arousal emotions (e.g., happy, angry) but struggled with subtler distinctions (e.g., sad vs. neutral), likely due to overlapping prosodic features
