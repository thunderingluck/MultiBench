#!/bin/bash
#SBATCH --job-name=affect_dyn
#SBATCH --output=./log/mosei/affect_dyn_%j.out
#SBATCH --error=./log/mosei/affect_dyn_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-11

source activate multibench

cd /orcd/home/002/evag/code/MultiBench/final_project

REGS=(0.001 0.01 0.1)
NOISE_PROBS=(0.0 0.25 0.5 0.75)

N_REG=${#REGS[@]}
REG=${REGS[$(($SLURM_ARRAY_TASK_ID % $N_REG))]}
NP=${NOISE_PROBS[$(($SLURM_ARRAY_TASK_ID / $N_REG))]}

python examples/affect/affect_dyn.py --hard-gate --reg $REG \
    --text-noise-prob $NP --text-noise-mode gaussian
