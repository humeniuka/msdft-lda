#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=chemie
#SBATCH --output=xms-caspt2.err

# Activate virtual python environment
source ~/msdft-venv/bin/activate

# Load environment variables for BAGEL
source ~/software/bagel/bagel_environment.sh

# Specify number of CPUs for BAGEL
export BAGEL_NUM_THREADS=${SLURM_CPUS_PER_TASK:=8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:=8}

# Run XMS-CASPT2 through python wrapper script
python calculate_ethylene_torsion_curve_xms-caspt2.py
