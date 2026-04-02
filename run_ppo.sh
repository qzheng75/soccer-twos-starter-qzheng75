#!/bin/bash
#SBATCH --job-name=ppo_soccer
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qzheng75@gatech.edu

mkdir -p logs

source /storage/ice1/0/9/qzheng75/miniconda/etc/profile.d/conda.sh
conda activate soccertwos

cd /storage/ice1/0/9/qzheng75/soccer-twos-starter-qzheng75

python example_ray_ppo_sp_still.py

