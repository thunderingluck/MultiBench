# Homework 3 — Vision-Language Models: Fine-Tuning Qwen2.5-VL for Facial Action Unit Recognition

## Overview

This assignment explores Vision-Language Models (VLMs) through baseline evaluation, prompt engineering, and **LoRA fine-tuning** of [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) on a facial Action Unit (AU) heatmap dataset for emotion recognition.

## Dataset

A custom dataset of **facial AU heatmap images** paired with text annotations describing active Action Units and dominant emotions. Images visualize AU activation patterns as spatial heatmaps overlaid on face geometry. The dataset contains 912 training samples with 4 held-out test images.

## What's Inside

### Baseline Inference (Pre-trained)
- Zero-shot evaluation of Qwen2.5-VL on held-out AU heatmap images
- The model demonstrated strong **parametric knowledge** of AU terminology (e.g., knew what AU12 and AU17 mean) but poor **visual grounding** — it couldn't read activation patterns from the heatmap images

### Prompt Engineering
- Experimented with system prompt variations: few-shot examples, constrained answer formats, chain-of-thought reasoning
- Evaluated how prompt structure affects the model's ability to map visual AU patterns to emotion labels

### LoRA Fine-Tuning
- Fine-tuned Qwen2.5-VL using LoRA on the training set

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 5 |
| Learning Rate | 1e-4 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| LoRA Targets | q, k, v, o projections |

### Post-Training Evaluation
- Re-tested on the same held-out images to compare pre- vs. post-fine-tuning performance
- Analyzed which questions improved, what errors were corrected, and whether fine-tuning introduced new failure modes

### Reading
- Tsimpoukelli et al., "Multimodal Few-Shot Learning with Frozen Language Models" (2021)
- Nguyen et al., "Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP" (2022)
- Dwivedi et al., "Generative AI: Here to Stay, but for Good?" (2023)

## Key Findings

- The most striking result was the **asymmetry between parametric knowledge and visual grounding**: the base model knew AU vocabulary but couldn't interpret heatmap visualizations, confirming that VLM factual knowledge and perceptual grounding are largely independent capabilities
- LoRA fine-tuning with rank 16 (increased from default 4) was necessary to learn the heatmap-to-AU mapping — lower ranks lacked capacity for this visually novel task
- 5 epochs was sufficient; the default 100 epochs severely overfit on 912 samples
- Converting non-image data (like AU activation vectors) into image representations introduces information loss through aggregation, scale sensitivity, and spatial encoding artifacts
