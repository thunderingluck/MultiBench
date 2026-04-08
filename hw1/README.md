# Homework 1 — Dataset: Multimodal Emotion Recognition

## Overview

This assignment focuses on the full data pipeline for a multimodal emotion recognition task: collecting, preprocessing, visualizing, and evaluating a dataset that combines **audio** and **text** modalities.

## Dataset

We use the [SUPERB Emotion Recognition](https://huggingface.co/datasets/superb) benchmark, which contains speech recordings labeled with emotion categories. From this raw audio, we extract two modalities:

- **Audio features:** WAV files resampled to 16kHz, log-mel spectrograms, and MFCC vectors (13 coefficients)
- **Text transcripts:** Generated via OpenAI Whisper (`tiny` model) automatic speech recognition

All features are organized under a unified metadata CSV linking each sample's audio path, transcript, label, and extracted feature paths.

## What's Inside

- **Preprocessing pipeline** — Audio normalization, feature extraction (log-mel, MFCC), and Whisper-based ASR transcription across all dataset splits
- **Visualizations** — t-SNE embeddings of both audio (MFCC statistics) and text (TF-IDF) feature spaces, colored by emotion class; per-sample visualizations of waveforms, spectrograms, and transcripts
- **Evaluation metrics** — Implementations of accuracy, per-class precision/recall, and macro-averaged F1 score from scratch
- **Instruction tuning prompts** — Three prompt templates for zero-shot classification (sentiment analysis, emotion recognition, intent detection)
- **Bonus: Synthetic multimodal digit classification** — Built a multimodal dataset by pairing MNIST images with synthetic (sometimes noisy) text captions, then trained image-only, text-only, and early-fusion logistic regression classifiers to study how modality fusion improves robustness under noise

## Key Findings

- Audio MFCC features showed moderate emotion clustering in t-SNE space, while text TF-IDF features produced less separable clusters — suggesting prosodic cues carry stronger emotion signal than lexical content alone in this dataset
- The bonus multimodal fusion experiment confirmed that combining modalities improves robustness: the fused model outperformed both unimodal baselines, especially when the text modality was noisy
- Preprocessing engineering (resampling, handling variable-length audio, Whisper inference at scale) was more involved than expected
