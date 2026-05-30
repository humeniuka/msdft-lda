Multistate Density Functional Theory with the Local Density Approximation
-------------------------------------------------------------------------
The python package ``mlmsdft`` implements the local density matrix approximation (LDMA) of Ref [1]_ for calculating excited states variationally. Rigorous multistate density functional theory is introduced in Ref [2]_. The fundamental quantity in this theory is the matrix density D(r), which contains the state densities on the diagonal and the transition densities on the off-diagonal. In LDMA the local exchange-correlation functional is converted into an analytic matrix functional of D(r). [3]_

The code for reproducing the figures and tables of Ref. [1]_ can be found in the subfolder ``examples/``.


Installation
------------
To install the pacakge in a separate virtual environment run

.. code-block:: bash

   $ python -m venv ~/msdft-venv
   $ source ~/msdft-venv/bin/activate

followed by

.. code-block:: bash

   $ pip install -e .

To verify the proper functioning of the code a set of tests should be run with

.. code-block:: bash

   $ cd tests
   $ python3.11 -m unittest

Getting Started
---------------

Variationally optimize the lowest few singlet and triplet states of H2 molecule for (2e,2o) active space:

.. code-block:: python

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


----------
References
----------
.. [1] Alexander Humeniuk, Yangyi Lu, Jiali Gao "Covariant Local Matrix Density Approximation through Spectral Reconstruction in Multistate Density Functional Theory",
   submitted to JCTC (2026)
   https://doi.org/TO-BE-ADDED
.. [2] Yangyi Lu, Jiali Gao, "Multistate Density Functional Theory for Excited States",
   J. Phys. Chem. Lett. 2022, 13, 7762-7769,
   https://doi.org/10.1021/acs.jpclett.2c02088
.. [3] Alexander Humeniuk, "Approximate Functionals for Multistate Density Functional Theory",
   J. Chem. Theory. Comput. 2024, 20, 13, 5497-5509,
   https://doi.org/10.1021/acs.jctc.4c00330
