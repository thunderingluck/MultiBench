#!/bin/bash
#SBATCH --job-name=dyn_robust_eval
#SBATCH --output=./log/mosei/robust_eval_%A_%a.out
#SBATCH --error=./log/mosei/robust_eval_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-11

# Eval-only sweep: tests existing noise-augmented checkpoints under
# per-modality Gaussian noise (vision / audio / text) at sigma in {0.3, 0.5, 1.0}.
# Purpose: determine whether the gate has a real job under audio/vision corruption
# (regimes where E1 ignores the noisy modality and E2 doesn't).

source activate multibench

cd /orcd/home/002/evag/code/MultiBench/final_project

# Grid: 4 noise_probs x 3 infer_modes = 12 tasks. reg fixed at 0.01 (best cell
# from prior sweep). np=0.0 skipped since it has no noisy checkpoint.
NOISE_PROBS=(0.25 0.5 0.75)
INFER_MODES=(0 1 2)     # 0=adaptive, 1=E1-only, 2=E2-only
REG=0.01
EVAL_SIGMAS="0.3,0.5,1.0"
EVAL_MODS="vision,audio,text"

N_NP=${#NOISE_PROBS[@]}
N_IM=${#INFER_MODES[@]}
NP=${NOISE_PROBS[$(($SLURM_ARRAY_TASK_ID % $N_NP))]}
IM=${INFER_MODES[$(($SLURM_ARRAY_TASK_ID / $N_NP))]}

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID: reg=$REG, np=$NP, infer-mode=$IM"
echo "============================================================"

python examples/affect/affect_dyn.py --hard-gate --reg $REG \
    --text-noise-prob $NP --text-noise-mode gaussian \
    --eval-only \
    --eval-noise-sigmas $EVAL_SIGMAS \
    --eval-noise-modalities $EVAL_MODS \
    --infer-mode $IM

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID done"
echo "============================================================"
