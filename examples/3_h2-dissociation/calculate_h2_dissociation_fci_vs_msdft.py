#!/usr/bin/env python
"""
Compute dissociation curves for lowest few states of H2 with different multistate functionals.
"""
# coding: utf-8
import numpy
import pandas
from pyscf.data.nist import HARTREE2EV
import pyscf.gto
import pyscf.dft
import pyscf.fci
import pyscf.scf
import torch
from tqdm import tqdm

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.pure import LDA
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import merge_multiplet_energies
from mlmsdft.dft.spin import index_within_multiplicity


if __name__ == "__main__":
    # number of bond lengths
    ncoord = 90

    basis = 'cc-pvdz'

    bond_lengths = numpy.linspace(0.2, 5.0, ncoord)
    dataframes = []
    for i, r in enumerate(tqdm(bond_lengths)):
        # H2
        mol = pyscf.gto.M(
            atom = f"H 0 0 {-r/2}; H 0 0 {r/2}",
            basis = basis,
            charge = 0,
            spin = 0)
        # --- FCI ---
        print("FCI")
        # initial guess for orbitals
        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        rohf.analyze()

        # full CI for lowest three singlet states, ¹S
        norb = rohf.mo_energy.size
        # closed-shell
        nelec = (1,1)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff)
        fci = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, rohf.mo_coeff), shift=.5)
        fci.nroots = 3
        e, c = fci.kernel(nelec=nelec)
        e0 = e[0]
        for i, x in enumerate(c):
            print('state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e0)*HARTREE2EV, fci.spin_square(x, norb, nelec)[1]))
        energies_singlet = e

        # full CI for triplet excited state ³S
        mol.spin = 2
        norb = rohf.mo_energy.size
        nelec = (2,0)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff)
        fci = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, rohf.mo_coeff), shift=.5)
        fci.nroots = 1+1
        e, c = fci.kernel(nelec=nelec)
        for i, x in enumerate(c):
            print('state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e0)*HARTREE2EV, fci.spin_square(x, norb, nelec)[1]))
        energies_triplet = e[:1]

        # Save results of calculation to table.
        energies = numpy.hstack([energies_singlet, energies_triplet])
        spin_multiplicities = numpy.array([1, 1, 1] + [3])
        state_indices = index_within_multiplicity(spin_multiplicities)

        nrows = len(energies)
        data = {
            "basis": [mol.basis] * nrows,
            "bond length (Angstrom)": r,
            "method": ["FCI"] * nrows,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "energy (Hartree)": energies,
        }
        df_fci = pandas.DataFrame.from_dict(data)
        print(df_fci)
        dataframes.append(df_fci)

        for spin_type in SpinType:
            for xc_name, xc_functional_class in [
                # (xc_name, xc_functional_class)
                # pure xc-functionals
                ('LDA', LDA),
            ]:
                # --- MSDFT ---
                print(xc_name)
                xc_functional = xc_functional_class(mol)
                hamiltonian = HamiltonianSemilocal(
                    mol,
                    # compute kinetic energy from wavefunctions
                    kinetic_functional = None,
                    exchange_functional = xc_functional.exchange,
                    correlation_functional = xc_functional.correlation,
                    exact_exchange_functional = xc_functional.exact_exchange,
                    spin_type = spin_type,
                    # Increase number of chunks, if you run out of memory.
                    grid_chunks = 1
                )

                # Minimize subspace energy of (2,2) CAS space without spin symmetry.
                norb = 2
                nelec = 2
                msmd = MultistateMatrixDensityCAS.from_guess(
                    mol, norb, nelec,
                    guess="hcore",
                    spin_symmetry=False, spin_type=spin_type,
                )

                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'
                msmd.to(device=device)

                # convergence tolerance for function values |f(i+1)-f(i)|
                # and gradient |df/dx| depend on functional
                if "BR89" in xc_name:
                    ftol = 5.0e-6
                    gtol = 1.0e-3
                else:
                    ftol = 5.0e-8
                    gtol = 1.0e-5
                # Minimize the state-averaged energy and diagonalize Hamiltonian.
                energies, msmd = minimize_subspace_energy(
                    hamiltonian, msmd, ftol=ftol, gtol=gtol, debug=1)

                # convert torch tensor -> numpy array
                energies = energies.cpu().detach().numpy()

                # Classify states by spin
                spin_multiplicities = msmd.spin_multiplicity().cpu().detach().numpy()
                if spin_type == SpinType.INVARIANT:
                    # Combine energies from degenerate multiplets.
                    # e.g. E(S=1,Sz=-1),E(S=1,Sz=0),E(S=1,Sz=+1) ---> E(S=1)
                    energies, spin_multiplicities = merge_multiplet_energies(
                        energies, spin_multiplicities)
                state_indices = index_within_multiplicity(spin_multiplicities)

                # Save results of calculation to table.
                nrows = len(energies)
                # Add a space to the name of the xc-functional, so that its columns
                # come first when sorted alphabetically.
                xc_name = " "+xc_name

                data = {
                    "basis": [mol.basis] * nrows,
                    "bond length (Angstrom)": r,
                    "method": [xc_name] * nrows,
                    "state index": state_indices,
                    "spin multiplicity": spin_multiplicities,
                    "spin type": spin_type.name,
                    "energy (Hartree)": energies,
                }
                df_msdft = pandas.DataFrame.from_dict(data)
                print(df_msdft)

                dataframes.append(df_msdft)

    df = pandas.concat(dataframes)
    df.to_csv("h2_dissociation.csv")
    print(df)
