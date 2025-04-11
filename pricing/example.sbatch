#!/bin/bash
#SBATCH --job-name=example2_sbatch
#SBATCH --output=example_sbatch2.out
#SBATCH --error=example_sbatch2.err
#SBATCH --account=pi-haifengxu
#SBATCH --time=24:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000

module load python
module load gurobi
source activate py39
python pricing_experiments.py --n_problems 5 --dataname school_trajectories --n_samples 100 --Q 20 --n_types 20
