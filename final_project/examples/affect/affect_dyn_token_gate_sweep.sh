#!/bin/bash
#SBATCH --job-name=dyn_token
#SBATCH --output=./log/mosei/dyn_token_%A_%a.out
#SBATCH --error=./log/mosei/dyn_token_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1

# Per-timestep (token-level) gate conditioned on per-modality reliability.
#   array=0 -> np=0.0 (architectural test: per-token gate alone)
#   array=1 -> np=0.5 (per-token gate + noise augmentation)
# reg fixed at 0.01 to match the confidence-aware gate sweep.

source activate multibench

cd /orcd/home/002/evag/code/MultiBench/final_project

NPS=(0.0 0.5)
NP=${NPS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID: token gate, reg=0.01, np=$NP"
echo "============================================================"

python examples/affect/affect_dyn_token_gate.py --hard-gate --reg 0.01 \
    --text-noise-prob $NP --text-noise-mode gaussian \
    --eval-noise-sigmas 0.3,0.5,1.0 \
    --eval-noise-modalities text
