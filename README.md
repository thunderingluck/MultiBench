# MMAI 2026 - Eva

Welcome to my repository for Multimodal AI (MAS.S60 / 6.S985) Spring 2026.
This is a fork of [MultiBench](https://github.com/pliang279/MultiBench) (Liang et al., NeurIPS 2021) that serves as my course homebase — homework assignments, project experiments, and explorations throughout the semester.

## Bio

Hi! I'm Eva. I'm a CS student at MIT, doing research on AI interpretability — sparse autoencoders on physics simulations at CSAIL, weight-space geometry of RLVR.

## Final Project

**Adaptive Multimodal Fusion with Per-Layer Sigmoid Gating**

Standard multimodal fusion methods apply a single global strategy across all layers, missing the fact that different modalities may be more or less informative at different levels of abstraction. This project extends [DynMM](https://arxiv.org/abs/2209.07574) (Dynamic Multimodal Fusion) with learned **per-layer sigmoid gating** on [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) for multimodal sentiment analysis. Instead of a binary keep/drop decision per modality, the model learns continuous per-layer weights that control how much each modality contributes at each stage of processing.

➡️ See [`final_project/`](final_project/) for code, configs, and detailed project README.

## Homework

| # | Topic | Link |
|---|-------|------|
| 1 | Dataset | [`hw1/`](hw1/) |
| 2 | Fusion | [`hw2/`](hw2/) |
| 3 | VLM | [`hw3/`](hw3/) |

## Repository Structure

```
.
├── README.md                # ← you are here
├── hw1/                     # Homework 1 - Dataset
├── hw2/                     # Homework 2 - Fusion
├── hw3/                     # Homework 3 - VLM
├── final_project/           # Course project code & reports
│   ├── README.md            # Detailed project description, setup, results
│   ├── midterm_report.pdf
│   └── ...                  # Training scripts, configs, analysis notebooks
└── .gitignore
```

> This repo is forked from [pliang279/MultiBench](https://github.com/pliang279/MultiBench). The original MultiBench codebase provides the multimodal benchmarking infrastructure; my additions live in `hw1/`–`hw3/` and `final_project/`.

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
  <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
</a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
