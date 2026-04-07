#!/bin/bash
#SBATCH --job-name=affect_mm
#SBATCH --output=./log/mosei/affect_mm_%j.out
#SBATCH --error=./log/mosei/affect_mm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00

source activate multibench

cd /orcd/home/002/evag/code/MultiBench/final_project

python examples/affect/affect_mm.py --fusion 3 "$@"
