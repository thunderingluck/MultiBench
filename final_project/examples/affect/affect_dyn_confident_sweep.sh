#!/bin/bash
#SBATCH --job-name=dyn_conf
#SBATCH --output=./log/mosei/dyn_conf_%A_%a.out
#SBATCH --error=./log/mosei/dyn_conf_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1

# Confidence-aware DynMM gate. Tests two configs:
#   array=0 -> np=0.0 (falsifiable test: does the architecture alone fix routing?)
#   array=1 -> np=0.5 (combine architectural + noise-augmentation fixes)
# reg fixed at 0.01 (mid-routing, most interpretable).

source activate multibench

cd /orcd/home/002/evag/code/MultiBench/final_project

NPS=(0.0 0.5)
NP=${NPS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID: confidence gate, reg=0.01, np=$NP"
echo "============================================================"

python examples/affect/affect_dyn_confident.py --hard-gate --reg 0.01 \
    --text-noise-prob $NP --text-noise-mode gaussian \
    --eval-noise-sigmas 0.3,0.5,1.0 \
    --eval-noise-modalities text
