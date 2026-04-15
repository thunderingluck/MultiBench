#!/bin/bash
#SBATCH --job-name=affect_dyn
#SBATCH --output=./log/mosei/affect_dyn_%A_%a.out
#SBATCH --error=./log/mosei/affect_dyn_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-11

source activate multibench

cd /orcd/home/002/evag/code/MultiBench/final_project

REGS=(0.001 0.01 0.1)
NOISE_PROBS=(0.0 0.25 0.5 0.75)
EVAL_SIGMAS="0.3,0.5,1.0"

N_REG=${#REGS[@]}
REG=${REGS[$(($SLURM_ARRAY_TASK_ID % $N_REG))]}
NP=${NOISE_PROBS[$(($SLURM_ARRAY_TASK_ID / $N_REG))]}

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID: reg=$REG, text-noise-prob=$NP"
echo "============================================================"

# Stage 1: train (adaptive gate) and evaluate clean + robustness sweep
echo ">>> Stage 1: train + eval adaptive gate (infer-mode 0)"
python examples/affect/affect_dyn.py --hard-gate --reg $REG \
    --text-noise-prob $NP --text-noise-mode gaussian \
    --eval-noise-sigmas $EVAL_SIGMAS

# Stage 2: re-eval same checkpoint forcing E1 only (text path)
# Only meaningful when NP > 0 (otherwise no noisy checkpoint was produced with the tag)
if [ "$NP" != "0.0" ]; then
    echo ">>> Stage 2: eval E1-only (infer-mode 1)"
    python examples/affect/affect_dyn.py --hard-gate --reg $REG \
        --text-noise-prob $NP --text-noise-mode gaussian \
        --eval-only --eval-noise-sigmas $EVAL_SIGMAS \
        --infer-mode 1

    echo ">>> Stage 3: eval E2-only (infer-mode 2)"
    python examples/affect/affect_dyn.py --hard-gate --reg $REG \
        --text-noise-prob $NP --text-noise-mode gaussian \
        --eval-only --eval-noise-sigmas $EVAL_SIGMAS \
        --infer-mode 2
fi

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID done"
echo "============================================================"
