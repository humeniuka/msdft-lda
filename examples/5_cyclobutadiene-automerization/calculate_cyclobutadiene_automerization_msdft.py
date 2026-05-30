#!/usr/bin/env python
"""
Compute the potential energy curves of the lowest few electronic states
for the automerization of cyclobutadiene

    rectangle <-> square <-> rectangle

Reference
---------
[Eckert-Maksic2006] M. Eckert-Maksic et al.
    "Automerization reaction of cyclobutadiene and its barrier
    height: An ab initio benchmark multireference average-
    quadratic coupled cluster study"
    J. Chem. Phys. 125, 064310 (2006)
    https://doi.org/10.1063/1.2222366
"""
# coding: utf-8
import numpy
import numpy.linalg as la
import pandas
from prefect import task
import pyscf.gto
import pyscf.scf
import pyscf.tools
import torch
import torch.linalg
from tqdm import tqdm

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.density import reorder_active_orbitals
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.pure import LDA
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import merge_multiplet_energies
from mlmsdft.dft.spin import index_within_multiplicity
from mlmsdft.utils.workflow import cache_key_function


def cyclobutadiene(
    rCCx = 1.562,
    rCCy = 1.349,
    rCH = 1.077,
    angleCHx = 134.9,
    basis = 'cc-pvdz',
) -> pyscf.gto.Mole:
    """
    Geometry of D2h or D4h cyclobutadiene

    rCCx: length of horizontal CC bonds (in Ang)
    rCCy: length of vertical CC bonds (in Ang)
    rCH: length of CH bond (in Ang)
    angleCHx: angle between CH bond and x-axis (in degrees)
    """
    # convert angle to radians
    angleCHx = numpy.deg2rad(angleCHx)
    # position of hydrogen in the top right corner
    xH = rCCx/2.0 + numpy.cos(numpy.pi-angleCHx) * rCH
    yH = rCCy/2.0 + numpy.sin(numpy.pi-angleCHx) * rCH

    # cyclobutadiene
    mol = pyscf.gto.M(
        atom = f"""
        C { rCCx/2.0}    { rCCy/2.0}        0.0;
        C { rCCx/2.0}    {-rCCy/2.0}        0.0;
        C {-rCCx/2.0}    {-rCCy/2.0}        0.0;
        C {-rCCx/2.0}    { rCCy/2.0}        0.0;
        H { xH}          { yH}              0.0;
        H { xH}          {-yH}              0.0;
        H {-xH}          {-yH}              0.0;
        H {-xH}          { yH}              0.0;
        """,
        basis = basis,
        spin = 0
    )
    return mol


def find_active_pi_orbitals(mol: pyscf.gto.Mole, mo_coeff, nelec_pi, norb_pi, verbose=True):
    """
    Find the indices of `norb` pi-orbitals among the MO coefficients
    from the HF calculations.

    :param mol: molecule with the number of electrons
    :type mol: pyscf.gto.Mole

    :param mo_coeff: molecular orbitals coefficients
    :type mo_coeff: numpy.ndarray of shape (nao,nao)

    :param nelec_pi: number of active pi electrons
    :type nelec_pi: int > 0

    :param norb_pi: number of active pi orbitals
    :type norb_pi: int > 0

    :param verbose: print norms of projections of MO coefficients
        that are used to assign pi-orbital character
    :type param: bool


    :return active_orbitals: 0-based indices of active orbitals
    :rtype active_orbitals: list of int
    """
    # total number of electrons
    nelec = mol.tot_electrons()
    # Indices of HOMO and LUMO
    # For odd number of electrons, the HOMO is actually the SOMO (singly occupied MO).
    HOMO = (nelec+1)//2-1

    # Find the indices of all atomic pz-orbitals on carbon atoms
    pz_ao_indices = []
    for i,(atom_id, symbol_str, nl_str, real_sph_str) in enumerate(
            pyscf.gto.sph_labels(mol, fmt=False)):
        if symbol_str == 'C' and real_sph_str == 'z':
            pz_ao_indices.append(i)
    pz_ao_indices = numpy.array(pz_ao_indices)
    # If the molecule lies in the xy plane, the pz-orbitals of the pi-system
    # should be separated from the sigma-orbitals.
    # Projecting the MO coefficients onto pz-orbitals, annihilates all orbitals
    # that are not pi-orbitals.
    pz_norms = la.norm(mo_coeff[pz_ao_indices,:], axis=0)
    # Indices of pi-orbitals.
    pi_mo_indices = numpy.where(pz_norms > 0.5)[0]
    # Indices of occupied and virtual indices pi-orbitals. It is assumed that the MO
    # coefficients are sorted by energy.
    pi_occ_indices = pi_mo_indices[pi_mo_indices <= HOMO]
    pi_virt_indices = pi_mo_indices[pi_mo_indices > HOMO]

    # Select highest occupied pi orbitals to accomodate `nelec_pi` pi electrons
    active_orbitals = pi_occ_indices[-(nelec_pi+1)//2:].tolist()
    # Add enough virtual pi orbitals until the list contains `norb_pi` active orbitals.
    active_orbitals += pi_virt_indices[:(norb_pi-len(active_orbitals))].tolist()

    # Show which MOs are pi-orbitals and belong to the active space.
    if verbose:
        print(" MO  |<pz,MO>|²  pi-orbital  active")
        for mo, pz_norm in enumerate(pz_norms):
            if mo in pi_mo_indices:
                is_pi_str = "yes"
            else:
                is_pi_str = "   "

            if mo in active_orbitals:
                is_active_str = "yes"
            else:
                is_active_str = "   "
            print(f"{mo+1:5}     {pz_norm:5.3f}     {is_pi_str}     {is_active_str}")

    print(f"pi orbitals        : {pi_mo_indices}")
    print(f"active pi orbitals : {active_orbitals}")

    return active_orbitals


@task(cache_key_fn=cache_key_function)
# Results are cached in .prefect/storage/
def cyclobutadiene_msdft_single_point(
    scan_coordinate = 0.0,
    basis = 'cc-pvdz',
    # 4 electrons in 4 orbitals
    nelec = 4,
    norb = 4,
    optimize_orbitals = False,
    state_average = None,
    # All spin states are allowed
    spin_symmetry = False,
    spin_type = SpinType.POLARIZED,
    max_level = numpy.inf,
    xc_name = 'LDA',
    xc_functional_class = LDA,
) -> pandas.DataFrame:
    """
    Perform an MSDFT calculation along the automerization coordinate of cyclobutadiene.

    :param scan_coordinate: parameter s that interpolates linearly
        between the rectangular (s=-1), square (s=0) and the other rectangular (s=+1)
        geometries of cyclobutadiene
    :param basis: basis set
    :paran nelec: number of electrons in active space
    :param norb: number of orbitals in active space
    :param optimize_orbitals: If orbital optimization is turned off, the Hamiltonian
        is constructed for the HF orbitals and diagonalized once to get the CI
        coefficients of the CAS wavefunctions.

    :param state_average: number of states M to include in the state average
        The subspace energy is calculated as the sum over the lowest M eigenvalues
        of the Hamiltonian. None means to include all states.
        `state_average` only matters if the orbitals are optimized.
    :type state_average: int or None

    :param spin_symmetry: Whether to include only states with the same spin
    :param spin_type: determines whether to use the charge density (SpinType.UNPOLARIZED),
        separate spin densities (SpinType.POLARIZED) or the supermatrix of
        same- and mixed-spin densities (SpinType.INVARIANT)
    :param max_level:  Determinants are only included if they differ by at most `max_level`
        from the HF determinant (0 - HF, 1 - HF+singles, 2 - HF+singles+doubles, etc.)
    :param xc_name: Name of xc-functional used in table
    :param xc_functional_class: instance of `:~class:PureXCFunctional

    :return df: pandas dataframe with excitation energies
    """
    print(f"* Scan coordinate s= {scan_coordinate}")
    # The geometric parameters come from table I row "cc-pVDZ" of [Eckert-Maksic].
    # - D2h square: C=C bonds parallel to y-axis
    #   rCCx=1.573, rCCy=1.367, rCH=1.093, angleCHx=134.9°
    # - D2h square C=C bonds parallel to x-axis
    #   rCCx=1.367, rCCy=1.573, rCH=1.093, angleCHx=270.0°-134.9°
    # - square transition state
    #   rCCx=1.461, rCCy=1.461, rCH=1.092, angleCHx=135.0°

    assert -1.0 <= scan_coordinate <= 1.0
    s = abs(scan_coordinate)
    if scan_coordinate <= 0.0:
        # Interpolate linearly in internal coordinates between
        # the left rectangle minimum and the square transition state.
        mol = cyclobutadiene(
            rCCx=s*1.573+(1-s)*1.461,
            rCCy=s*1.367+(1-s)*1.461,
            rCH=s*1.093+(1-s)*1.092,
            angleCHx=s*134.9+(1-s)*135.0,
            basis=basis
        )
    else:
        # Interpolate linearly between the square transition state
        # and the right rectangular minimum.
        mol = cyclobutadiene(
            rCCx=s*1.367+(1-s)*1.461,
            rCCy=s*1.573+(1-s)*1.461,
            rCH=s*1.093+(1-s)*1.092,
            angleCHx=s*(270.0-134.9)+(1-s)*135.0,
            basis=basis
        )

    # mol.tofile(f"tmp/cyclobutadiene_{s}.xyz", format="xyz")

    print("Initial guess for molecular orbitals is taken from ROHF calculation.")
    rohf = pyscf.scf.ROHF(mol)
    rohf.verbose = 4
    rohf.kernel()
    pyscf.tools.molden.from_scf(rohf, f"tmp/orbitals_{scan_coordinate}.rohf.molden")
    mo_coeff = rohf.mo_coeff

    # Find the pi-orbitals that should go into the active space
    # and reorder the molecular orbitals accordingly.
    active_orbitals = find_active_pi_orbitals(mol, mo_coeff, nelec, norb)
    mo_coeff = reorder_active_orbitals(mol, mo_coeff, active_orbitals, nelec)

    msmd = MultistateMatrixDensityCAS.from_guess(
        mol, norb, nelec,
        spin_symmetry=spin_symmetry,
        spin_type=spin_type,
        max_level=max_level,
        guess=mo_coeff)
    # Export the initial guess for the molecular orbitals.
    pyscf.tools.molden.from_mo(
        mol,
        f"tmp/orbitals_{scan_coordinate}.guess.molden",
        msmd.orbital_coefficients()
    )

    print(f"number of states in subspace: {msmd.number_of_states}")

    # Where to run the calculation, 'cuda' or 'cpu'?
    use_cuda = True
    if torch.cuda.is_available() and use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    msmd.to(device=device)

    xc_functional = xc_functional_class(mol)

    if spin_type in [SpinType.INVARIANT, SpinType.INVARIANT_MIX]:
        # With INVARIANT and INVARIANT_MIX spin treatments the matrices
        # that have to be diagonalized are larger.
        grid_chunks = 50*16
    else:
        grid_chunks = 50

    hamiltonian = HamiltonianSemilocal(
        mol,
        # compute kinetic energy from wavefunctions
        kinetic_functional = None,
        exchange_functional = xc_functional.exchange,
        correlation_functional = xc_functional.correlation,
        exact_exchange_functional = xc_functional.exact_exchange,
        spin_type = spin_type,
        # Increase number of chunks, if you run out of memory.
        grid_chunks = grid_chunks
    )

    print(f"Orbital optimization: {optimize_orbitals}")
    if optimize_orbitals:
        print(f"Number of states included in state average (None=all): {state_average}")
        # Minimize the state-averaged energy and diagonalize Hamiltonian.
        energies, msmd = minimize_subspace_energy(
            hamiltonian, msmd, state_average=state_average,
            ftol=5.0e-8, gtol=1.0e-5, debug=1
        )

        # Export final optimized molecular orbitals.
        pyscf.tools.molden.from_mo(
            mol,
            f"tmp/orbitals_{scan_coordinate}.optimized.molden",
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
    with open(f"tmp/msdft_{scan_coordinate}.{mol.basis}.dets", "w") as fp:
        fp.write(str(msmd))

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
        "scan coordinate": [scan_coordinate] * nrows,
        "method": [xc_name] * nrows,
        "state index": state_indices,
        "spin multiplicity": spin_multiplicities,
        "spin type": spin_type.name,
        "energy (Hartree)": energies,
    }
    df_msdft = pandas.DataFrame.from_dict(data)

    return df_msdft


if __name__ == "__main__":
    # number of scan geometries
    ncoord = 21

    scan_coordinates = numpy.linspace(-1.0, 1.0, ncoord)
    # Since the subspace energy is minimized by gradient descent, we have
    # to slightly perturb the D4h symmetric structure (s=0.0)
    # to get out of a local minimum.
    scan_coordinates[scan_coordinates == 0.0] += 0.001

    dataframes = []
    for spin_type in SpinType:
        for (xc_name, xc_functional_class) in [
                ("LDA", LDA),
        ]:
            # Compute potential energy curves
            for i, scan_coordinate in enumerate(tqdm(scan_coordinates)):
                df_msdft = cyclobutadiene_msdft_single_point(
                    scan_coordinate=scan_coordinate,
                    spin_type=spin_type,
                    xc_name=xc_name,
                    xc_functional_class=xc_functional_class
                )

                dataframes.append(df_msdft)
                print(df_msdft)

                # Save intermediates after each step.
                df = pandas.concat(dataframes)
                df.to_csv("cyclobutadiene_automerization.msdft.csv", index=False)

    # Show complete table
    print(df)
