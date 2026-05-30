#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=h2_fci_msdft.err
#SBATCH --time=10:00:00

# Activate virtual python environment
source ~/msdft-venv/bin/activate

# Specify number of CPUs for pyscf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Cached results are stored in hidden subfolder
export PREFECT_LOCAL_STORAGE_PATH=$(pwd)/.prefect/storage

# Flush stdout immediately
export PYTHONUNBUFFERED=1

# Run MSDFT
python calculate_h2_dissociation_fci_vs_msdft.py
