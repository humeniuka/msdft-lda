#!/usr/bin/env python
"""
Compute MSDFT energies
"""
import argparse
import json
import numpy
import os
from os.path import exists, join
import pandas
import pathlib
from pyscf.data.nist import HARTREE2EV
import pyscf.gto
import pyscf.scf
import pyscf.tools.molden
import torch
import torch.linalg
import textwrap

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.density import reorder_active_orbitals
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.xc import lda_c_chachiyo
from mlmsdft.dft.xc import lda_x_dirac


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Compute MSDFT energies for compound in .xyz file

        NOTE: If you run out of memory, increase the 'grid_chunks' parameter (default 1) in the .json file
            in order to divide the coordinate grid into chunks, which are processed sequentially.

        """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'xyz_file',
        help=(
            ".xyz file with molecular geometry (in Angstrom)."
        )
    )
    parser.add_argument(
        '-p', '--parameters', type=str,
        help=(
            "The parameters for the calculation (basis set, active orbitals etc.) are read "
            "from a .json file. If left empty, parameters are read from a .json file with "
            "a similar name as the .xyz file (e.g. molecule.json for molecule.xyz)"
        ),
        default=None
    )
    parser.add_argument(
        '--use-cuda', action='store_true',
        help="Use GPU if available. For large system, the GPU memory might not be enough.",
        default=False
    )
    parser.add_argument(
        '--results-dir',
        help=(
            ".csv table with state energies and .molden files with initial and optimized orbitals "
            "are written to this folder."
        ),
        default=None
    )
    args = parser.parse_args()

    # remove suffix, molecule.xyz -> molecule
    path = pathlib.Path(args.xyz_file)
    compound = path.stem

    # find .json file with parameters controlling the calculation.
    if args.parameters is None:
        # Read .json file with similar name,
        # e.g. /some/path/molecule.xyz -> /some/path/molecule.json
        json_file = path.parent/(path.stem+".json")
    else:
        json_file = args.parameters

    # Results of calculations and orbitals are stored in separate folder.
    if args.results_dir is None:
        results_dir = path.parent/"results"
    else:
        results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    compute_msdft_energies(
        compound,
        args.xyz_file,
        json_file,
        use_cuda=args.use_cuda,
        results_dir=results_dir
    )


def compute_msdft_energies(
    compound: str,
    xyz_file: str,
    json_file: str,
    use_cuda=False,
    results_dir="results"
):
    """
    compute MSDFT energy for given compound.

    The molecular geometry is read from {compound}.xyz and the parameters for the
    calculation (basis, charge, active space, etc.) are read from {compound}.json
    if the file exists.
    """
    if torch.cuda.is_available() and use_cuda:
        device = 'cuda'
    else:
        # Run calculations on CPU, laptop GPU does not have enough memory.
        device = 'cpu'

    # empty dataframe, calculations for different geometries are appended.
    df = pandas.DataFrame()

    # Load parameters of calculation from .json file.
    if exists(json_file):
        with open(json_file, 'r') as fp:
            print(f"Parameters for calculation are read from {json_file} .")
            parameters = json.load(fp)
    else:
        # Use default parameters
        parameters = {}

    # Geometry is read from .xyz file.
    print(f"Molecular geometry is read from {xyz_file} .")

    mol = pyscf.gto.M(
        atom = xyz_file,
        unit = parameters.get('unit', 'Angstrom'),
        basis = parameters.get('basis', 'cc-pvdz'),
        charge = parameters.get('charge', 0),
        spin = parameters.get('spin', 0)
    )

    # It is essential to select the correct active orbitals.
    # All frontier orbitals that are close in energy should be included.
    # So it is always a good idea to look at the initial guess and make
    # sure that the selected active space is the intended one.
    print("Initial guess for molecular orbitals is taken from ROHF calculation.")
    rohf = pyscf.scf.ROHF(mol)
    rohf.verbose = 4
    rohf.kernel()
    pyscf.tools.molden.from_scf(rohf,
        join(results_dir, f"{compound}.rohf.molden"),
    )
    mo_coeff = rohf.mo_coeff

    # `nelec` electrons in `norb` orbitals
    nelec = parameters.get('nelec', 2)
    norb = parameters.get('norb', 2)

    # Sometimes it is necessary to reorder the orbitals.
    active_orbitals = parameters.get('active_orbitals', [])
    if active_orbitals:
        assert len(active_orbitals) == norb
        mo_coeff = reorder_active_orbitals(mol, mo_coeff, active_orbitals, nelec)

    # Whether to include only states with the same spin
    spin_symmetry = parameters.get('spin_symmetry', True)

    # Determinants are only included if they differ by at most `max_excitation_level`
    # from the HF determinant (0 - HF, 1 - HF+singles, 2 - HF+singles+doubles, etc.)
    max_level = parameters.get('excitation_level', numpy.inf)

    # Whether to use the charge density, separate spin densities
    # or the supermatrix of same- and mixed-spin densities.
    spin_type = SpinType(parameters.get('spin_type', 'spin_unpolarized'))

    msmd = MultistateMatrixDensityCAS.from_guess(
        mol, norb, nelec,
        spin_symmetry=spin_symmetry, spin_type=spin_type, max_level=max_level, guess=mo_coeff)
    msmd.to(device=device)

    print(msmd)
    # Export the initial guess for the molecular orbitals.
    pyscf.tools.molden.from_mo(
        mol,
        join(results_dir, f"{compound}.guess_orbitals.molden"),
        msmd.orbital_coefficients()
    )

    # xc-functional for MSDFT
    xc_functional = parameters.get('xc_functional', 'lda').lower()
    assert xc_functional in ['lda']
    # LDA
    exchange_functional = lda_x_dirac
    correlation_functional = lda_c_chachiyo
    print(f"Exchange functional    : {exchange_functional.__name__}")
    print(f"Correlation functional : {correlation_functional.__name__}")

    hamiltonian = HamiltonianSemilocal(
        mol,
        # compute kinetic energy from wavefunctions
        kinetic_functional = None,
        exchange_functional = exchange_functional,
        correlation_functional = correlation_functional,
        spin_type = spin_type,
        # Reduce the grid level, if you run out of memory.
        grid_level = parameters.get('grid_level', 3),
        # Increase number of chunks, if you run out of memory.
        grid_chunks = parameters.get('grid_chunks', 1)
    )

    optimize_orbitals = parameters.get('optimize_orbitals', True)
    print(f"Orbital optimization: {optimize_orbitals}")
    if optimize_orbitals:
        # State-averaged energy is calculated as the sum of the lowest
        # `state_average` eigenvalues of the Hamiltonian. None means
        # that all states are averaged over.
        state_average = parameters.get('state_average', None)
        print(f"Number of states included in state average (None=all): {state_average}")

        # Minimize the state-averaged energy.
        energies, msmd = minimize_subspace_energy(
            hamiltonian, msmd,
            state_average=state_average,
            debug=1, gtol=1.0e-5
        )

        # Export final optimized molecular orbitals.
        pyscf.tools.molden.from_mo(
            mol,
            join(results_dir, f"{compound}.optimized_orbitals.molden"),
            msmd.orbital_coefficients()
        )
    else:
        with torch.no_grad():
            # Diagonalize Hamiltonian using the initial guess molecular orbitals
            H = hamiltonian(msmd)
            energies, eigvecs = torch.linalg.eigh(H)

        # Transform matrix density to the basis of the eigenstates
        # D(r) -> Uᵀ D(r) U
        msmd.basis_transformation(eigvecs.T)

    # Print final CI coefficients
    print(msmd)
    # Save CI coefficients also to file
    with open(join(results_dir, f"{compound}.msdft.{mol.basis}.dets"), "w") as fp:
        fp.write(str(msmd))

    # Save results of calculation to table.
    energies = energies.detach().cpu().numpy()
    nrows = len(energies)
    data = {
        "compound": [compound] * nrows,
        "basis": [mol.basis] * nrows,
        # active space
        "electrons": [nelec] * nrows,
        "orbitals": [norb] * nrows,
        "method": [f"MSDFT({nelec}e/{norb}o)"] * nrows,
        "spin symmetry": spin_symmetry,
        "exchange": [exchange_functional.__name__] * nrows,
        "correlation": [correlation_functional.__name__] * nrows,
        "<S²>": numpy.round(msmd.spin_s2_expectation().detach().cpu().numpy(), 2),
        "energy (Hartree)": energies,
        r"vertical ΔE (eV)": (energies - energies[0]) * HARTREE2EV
    }
    df = pandas.concat([df, pandas.DataFrame.from_dict(data)])
    df.to_csv(join(results_dir, f"{compound}.msdft.{mol.basis}.csv"))
    print(df)

if __name__ == "__main__":
    main()
