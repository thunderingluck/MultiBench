#!/bin/bash
#SBATCH --job-name=affect_mm_av
#SBATCH --output=./log/mosei/affect_mm_av_%j.out
#SBATCH --error=./log/mosei/affect_mm_av_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Text-free late-fusion baseline on MOSEI (audio + vision only).
# Viability threshold: clean accuracy must clear ~0.71 (E2 under text noise
# at sigma=1.0) to justify building a three-branch DynMM variant with an
# audio+vision expert.

source activate multibench

cd /orcd/home/002/evag/code/MultiBench/final_project

python examples/affect/affect_mm_av.py --n-runs 1 --n-epochs 1000
