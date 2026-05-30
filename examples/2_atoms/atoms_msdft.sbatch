#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=193400M
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=atoms_msdft.err

# Activate virtual python environment
source ~/msdft-venv/bin/activate

# Specify number of CPUs for pyscf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Show which type of GPU is available
nvidia-smi

# Cached results are stored in hidden subfolder
export PREFECT_LOCAL_STORAGE_PATH=$(pwd)/.prefect/storage

# Flush print statements immediately
export PYTHONUNBUFFERED=1

# Run MSDFT
python atoms_table.py
