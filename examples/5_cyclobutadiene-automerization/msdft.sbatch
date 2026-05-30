#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=msdft.err
#SBATCH --time=52:00:00

# Activate virtual python environment
source ~/msdft-venv/bin/activate

# Specify number of CPUs for pyscf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Cached results are stored in hidden subfolder
export PREFECT_LOCAL_STORAGE_PATH=$(pwd)/.prefect/storage

# Flush stdout immediately
export PYTHONUNBUFFERED=1

# Folder for output needed for debugging (orbitals, CI coefficients)
mkdir -p tmp

# Run MSDFT
python calculate_cyclobutadiene_automerization_msdft.py
