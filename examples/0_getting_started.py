#!/usr/bin/env python
import pyscf.gto
import torch

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.pure import LDA
from mlmsdft.dft.spin import SpinType

# H2 as a pyscf molecule
mol = pyscf.gto.M(
    atom = "H 0 0 -0.35; H 0 0 0.35",
    basis = "cc-pvdz",
    charge = 0,
    spin = 0)

# MSDFT Hamiltonian with LMDA functional
xc_functional = LDA(mol)
hamiltonian = HamiltonianSemilocal(
    mol,
    exchange_functional = xc_functional.exchange,
    correlation_functional = xc_functional.correlation,
    spin_type = SpinType.INVARIANT_MIX,
    # Increase number of chunks, if you run out of memory.
    grid_chunks = 1
)

# Minimize subspace energy of (2,2) CAS space without spin symmetry.
msmd = MultistateMatrixDensityCAS.from_guess(
    mol, norb=2, nelec=2,
    guess="hcore",
    spin_symmetry=False, spin_type=SpinType.INVARIANT_MIX
)

# Code runs much faster on a GPU.
if torch.cuda.is_available():
    msmd.to(device='cuda')

# Minimize the state-averaged energy and diagonalize Hamiltonian.
energies, msmd = minimize_subspace_energy(
    hamiltonian, msmd, ftol=5.0e-8, gtol=1.0e-5, debug=1)

# Optimized matrix density and CI coefficients of states.
print(msmd)
print(f"Energies (in Hartree): {energies}")
