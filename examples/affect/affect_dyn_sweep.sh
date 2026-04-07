#!/bin/bash
#SBATCH --job-name=affect_dyn
#SBATCH --output=./log/mosei/affect_dyn_%j.out
#SBATCH --error=./log/mosei/affect_dyn_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-2

source activate multibench

cd /orcd/home/002/evag/code/MultiBench

REGS=(0.001 0.01 0.1)

REG=${REGS[$SLURM_ARRAY_TASK_ID]}

python examples/affect/affect_dyn.py --hard-gate --reg $REG
