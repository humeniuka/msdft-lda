#!/usr/bin/env python
"""
Compute the potential energy curves of the lowest three singlet states
of twisted ethylene with multistate LDA functional.
"""
# coding: utf-8
import numpy
import pandas
from prefect import task
import pyscf.gto
import pyscf.scf
import pyscf.tools
import torch
import torch.linalg
from tqdm import tqdm

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.pure import LDA
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import merge_multiplet_energies
from mlmsdft.dft.spin import index_within_multiplicity
from mlmsdft.utils.workflow import cache_key_function


@task(cache_key_fn=cache_key_function)
# Results are cached in .prefect/storage/
def ethylene_msdft_single_point(
    torsion_angle = 0.0,
    basis = 'cc-pvdz',
    # 2 electrons in 2 orbitals
    nelec = 2,
    norb = 2,
    # Only states with the same total spin are included in the subspace
    spin_symmetry = True,
    # Only singlet states
    spin = 0,
    spin_type = SpinType.INVARIANT,
    max_level = numpy.inf,
    xc_name = 'LDA',
    xc_functional_class = LDA,
) -> pandas.DataFrame:
    """
    Perform an MSDFT calculation on twisted ethylene.

    :param torsion_angle: torsion angle around C=C bond (in degrees)
    :param basis: basis set
    :paran nelec: number of electrons in active space
    :param norb: number of orbitals in active space
    :param spin_symmetry: Whether to include only states with the same spin
    :param spin: 2*S, spin=0 for singlets
    :param spin_type: determines whether to use the charge density (SpinType.UNPOLARIZED),
        separate spin densities (SpinType.POLARIZED) or the supermatrix of
        same- and mixed-spin densities (SpinType.INVARIANT)
    :param max_level:  Determinants are only included if they differ by at most `max_level`
        from the HF determinant (0 - HF, 1 - HF+singles, 2 - HF+singles+doubles, etc.)
    :param xc_name: Name of xc-functional used in table
    :param xc_functional_class: instance of `:~class:PureXCFunctional

    :return df: pandas dataframe with excitation energies
    """
    # The geometry of ethylene is taken from
    # A. I. Krylov, Chemical Physics Letters 338, 375 (2001).
    rCC = 1.330
    rCH = 1.076
    # H-C-H angle in radians
    angleHCH = numpy.deg2rad(116.6)
    #
    xH = numpy.cos(angleHCH/2.0) * rCH
    yH = numpy.sin(angleHCH/2.0) * rCH
    # torsion angle
    alpha = numpy.deg2rad(torsion_angle)
    c = numpy.cos(alpha)
    s = numpy.sin(alpha)

    # ethylene
    mol = pyscf.gto.M(
        atom = f"""
        C {-rCC/2.0}    0.0        0.0;
        C { rCC/2.0}    0.0        0.0;
        H {-rCC/2.0-xH} {-yH}      0.0;
        H {-rCC/2.0-xH} { yH}      0.0;
        H { rCC/2.0+xH} { c*yH}    { s*yH};
        H { rCC/2.0+xH} {-c*yH}    {-s*yH};
        """,
        basis = basis,
        charge = 0,
        spin = spin
    )

    print("Initial guess for molecular orbitals is taken from ROHF calculation.")
    rohf = pyscf.scf.ROHF(mol)
    rohf.verbose = 4
    rohf.kernel()
    pyscf.tools.molden.from_scf(rohf, "/tmp/orbitals.rohf.molden")
    mo_coeff = rohf.mo_coeff

    msmd = MultistateMatrixDensityCAS.from_guess(
        mol, norb, nelec,
        spin_symmetry=spin_symmetry,
        spin_type=spin_type,
        max_level=max_level,
        guess=mo_coeff)
    print(f"number of states in subspace: {msmd.number_of_states}")

    # Where to run the calculation, 'cuda' or 'cpu'?
    use_cuda = True
    if torch.cuda.is_available() and use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    msmd.to(device=device)

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

    # Minimize the state-averaged energy and diagonalize Hamiltonian.
    energies, msmd = minimize_subspace_energy(
        hamiltonian, msmd, ftol=5.0e-8, gtol=1.0e-5, debug=1)

    # Classify states by spin
    energies = energies.cpu().detach().numpy()
    spin_multiplicities = msmd.spin_multiplicity().cpu().detach().numpy()
    if spin_type == SpinType.INVARIANT:
        # Combine energies from degenerate multiplets.
        # e.g. E(S=1,Sz=-1),E(S=1,Sz=0),E(S=1,Sz=+1) ---> E(S=1)
        energies, spin_multiplicities = merge_multiplet_energies(
            energies, spin_multiplicities)
    state_indices = index_within_multiplicity(spin_multiplicities)

    # Save results of calculation to table.
    nrows = len(energies)

    data = {
        "basis": [mol.basis] * nrows,
        "torsion angle (deg)": torsion_angle,
        "method": [xc_name] * nrows,
        "state index": state_indices,
        "spin multiplicity": spin_multiplicities,
        "spin type": spin_type.name,
        "energy (Hartree)": energies,
    }
    df_msdft = pandas.DataFrame.from_dict(data)

    return df_msdft


if __name__ == "__main__":
    # number of torsion angles
    ncoord = 91

    torsion_angles = numpy.linspace(0.0, 180.0, ncoord)
    # Since the subspace energy is minimized by gradient descent, we have
    # to slightly perturb the planar ethylene molecule (at 0 and 180 deg)
    # to get out of a local minimum.
    # 0 + 0.01
    torsion_angles[0] += 0.01
    # 180 - 0.01
    torsion_angles[-1] -= 0.01

    dataframes = []
    # Results for different spin types are virtually indistinguishable,
    # so we limit ourselves to the INVARIANT calculation.
    for spin_type in [SpinType.INVARIANT]:
        # Compute potential energy curves
        for i, torsion_angle in enumerate(tqdm(torsion_angles)):
            df_msdft = ethylene_msdft_single_point(
                torsion_angle=torsion_angle,
                spin_type=spin_type
            )

            dataframes.append(df_msdft)
            print(df_msdft)

            # Save intermediates after each step.
            df = pandas.concat(dataframes)
            df.to_csv("ethylene_torsion.msdft.csv", index=False)

    # Show complete table
    print(df)
