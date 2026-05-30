# -*- coding: utf-8 -*-
"""
FCI and MSDFT calculations on a few atoms.
"""
from abc import ABC, abstractmethod
import numpy
import pandas
import torch

from prefect import task, flow

import pyscf.data
from pyscf.data.nist import HARTREE2EV
import pyscf.fci
import pyscf.gto
import pyscf.scf

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.xc import lda_x_dirac, lda_c_chachiyo
from mlmsdft.dft.pure import LDA
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import merge_multiplet_energies
from mlmsdft.utils.workflow import cache_key_function


__all__ = ["available_atoms"]


def unique_counts(array, difference_threshold=1.0e-5):
    """
    Count the number of repeated elements in `array` and return
    only the unique elements and how often each of them occurs.

    :param array: list of floats with repetitions

    :param difference_threshold: Two elements in `array` are considered
        the same if they differ by less than `difference_threshold`

    :return (unique, count):
        sorted list of the unique elements and their counts.
        Element `unique[i]` was found `count[i]` times.
    """
    array = numpy.sort(array)
    unique = [array[0]]
    counts = [1]
    for element in array[1:]:
        if abs(element - unique[-1]) < difference_threshold:
            counts[-1] += 1
        else:
            unique.append(element)
            counts.append(1)
    return unique, counts


def group_atomic_energy_levels(energies, spin_multiplicities):
    """
    Atomic energy levels (such as 1³P for the carbon ground state) are classified by their
    spin multiplicy, orbital angular momentum and an 1-based index that distinguishes states
    with the same spin and spatial symmetry (1¹S vs. 2¹S)

    Since states with the same total angular momentum L (e.g. L=1) but different projections
    onto the quantization axis (Lz=-1,0,1) have the same energy, we can determine L by counting
    the number of degenerate levels (having the same spin multiplicity)

    Each energy level is desribed a tuple (spin multiplicity 2*S+1, spatial multiplicity 2*L+1, state index N)
    that can be converted into a term symbol N²ˢ⁺¹(2L+1) as shown below for a few examples:

    spin            spatial        state        term
    multiplicity    multiplicity   index        symbol
    2*S+1           2*L+1                       N
       1               1             1          1¹S
       1               5             2          2¹D
       3               3             1          1³P

    :param energies: eigenenergies of atomic Hamiltonian with Sz=0
    :type energies: numpy.array of shape (m,)

    :param spin_multiplicities: spin multiplicity 2*S+1 for each eigenenergy
    :type spin_multiplicities: numpy.array of int's of shape (m,)

    :return: list of tuples (E, term-symbol, 2*S+1, 2*L+1, N) for each energy level
        where E is the total energy and 2*L+1 is the spatial degeneracy of that level
    """
    assert len(energies) == len(spin_multiplicities)
    level_energies = []
    term_symbols = []
    # 2*S+1 level
    level_spin_multiplicities = []
    # 2*L+1 for each level
    level_spatial_multiplicities = []
    # N for each level
    state_indices = []

    # superscript 2*S+1 in term symbol
    spin_superscript = {
        1: r"¹", 2: r"²", 3: r"³", 4: r"⁴", 5: r"⁵", 6: r"⁶", 7: r"⁷", 8: r"⁸", 9: r"⁹"
    }
    # spatial multiplicity 2*L+1 in spectroscopic notation
    spectroscopic_notation = {
        1: "S", 3: "P", 5: "D", 7: "F", 9: "G", 11: "H"
    }

    for spin_multiplicity in numpy.unique(spin_multiplicities):
        unique, counts = unique_counts(energies[spin_multiplicities == spin_multiplicity])
        for energy, spatial_multiplicity in zip(unique, counts):
            # state index that distinguishes between states with the same
            # spatial and spin symmetry.
            state_index = numpy.count_nonzero(
                (numpy.array(level_spin_multiplicities) == spin_multiplicity) &
                (numpy.array(level_spatial_multiplicities) == spatial_multiplicity)
            )
            level_energies.append(energy)
            # term symbol, e.g. 1³P
            term_symbol = "%d%s%s" % (
                state_index+1,
                spin_superscript[spin_multiplicity],
                spectroscopic_notation.get(spatial_multiplicity, "?")
            )
            term_symbols.append(term_symbol)

            level_spin_multiplicities.append(spin_multiplicity)
            level_spatial_multiplicities.append(spatial_multiplicity)
            state_indices.append(state_index+1)
    # sort by energy
    sort_idx = numpy.argsort(level_energies)
    return (
        numpy.array(level_energies)[sort_idx],
        numpy.array(term_symbols)[sort_idx],
        numpy.array(level_spin_multiplicities)[sort_idx],
        numpy.array(level_spatial_multiplicities)[sort_idx],
        numpy.array(state_indices)[sort_idx],
    )


class AtomicMultipletCalculation(ABC):

    @abstractmethod
    def full_configuration_interaction(self) -> pandas.DataFrame:
        """
        perform a FCI calculation
        """
        pass

    @abstractmethod
    def experiment(self) -> pandas.DataFrame:
        """
        experimental data from the NIST Atomic Spectra Database.
        """
        pass

    @abstractmethod
    def initial_matrix_density(self, spin_type: SpinType) -> MultistateMatrixDensityCAS:
        """
        set up an initial matrix density of an atom for a MSDFT calculation

        :param spin_type: Determines which spin projections Sz are included in the
            active subspace. The options are:
            POLARIZED, UNPOLARIZED: Only a single Sz=(neleca-nelecb)/2 component
                is included. For example, for triplet states only one of the three components
                would be present.
            INVARIANT, INVARIANT_MIX: All possible Sz components are included. For example,
                for triplet states all three components with Sz=-1,0,+1 would be present.
        :type spin_type: SpinType
        """
        pass

    @property
    def mol(self):
        # _mol has to be initialized inside constructor __init__()
        return self._mol

    @property
    def rohf(self):
        # _rohf has to be initialized inside constructor __init__()
        return self._rohf

    @task(cache_key_fn=cache_key_function)
    # Results are cached in ~/.prefect/storage/
    def multistate_dft(
        self,
        xc_name="LDA",
        exchange=lda_x_dirac,
        correlation=lda_c_chachiyo,
        exact_exchange=None,
        spin_type=SpinType.UNPOLARIZED,
        use_cuda=False,
    ) -> pandas.DataFrame:
        """
        Optimize the initial matrix density so that the MSDFT subspace energy attains
        a minimum. The individual energies of the eigenstates are obtained by diagonalizing
        the subspace Hamiltonian at the minimizing matrix density.
        The eigenstates are classified by their spin and spatial multiplicities.

        :param xc_name: name of exchange-correlation functional
        :type xc_name: str

        :param exchange: exchange functional, e.g. lda_x_dirac
        :type exchange: Callable

        :param correlation: correlation functional, e.g. lda_c_chachiyo
        :type correlation: Callable

        :param exact_exchange: Functional for exact (Hartree-Fock) exchange, e.g. HF(mol).exact_exchange
        :type exact_exchange: Callable or None

        :param spin_type: Determines how the electronic spin degrees of the matrix
            density are treated when constructing the Hamiltonian. The options are:
            UNPOLARIZED: The Hamiltonian is constructed from the spin-traced matrix density,
                Dᵢⱼ=Dᵅᵢⱼ(r)+Dᵝᵢⱼ(r). The (exact) exchange is calculated as 2*X[D/2]ᵢⱼ.
            POLARIZED: The (exact) exchange part of the Hamiltonian is computed separately for the
                spin-up and spin-down matrix densities, Xᵢⱼ = X[Dᵅᵢⱼ] + X[Dᵝᵢⱼ].
            INVARIANT:
                The (exact) exchange part of the Hamiltonian is computed from the (2N)x(2N)
                supermatrix containing the NxN same-spin blocks Dᵅᵅ and Dᵝᵝ on the diagonal and
                the mixed-spin blocks Dᵅᵝ and Dᵝᵅ on the off-diagonal:
                Xᵢⱼ=spin_trace(X[(Dᵅᵅ Dᵅᵝ \\ Dᵝᵅ Dᵝ)])
                The resulting exchange energy matrix is invariant under rotations of the
                electronic spins, provided that all components of a spin multiplet (Sz=-S,...,S)
                are included in the subspace.
            INVARIANT_MIX:
                The (exact) exchange part of the Hamiltonian is the average of 50% of the
                unpolarized and 50% of the invariant exchange parts,
                Xᵢⱼ=1/2 { 2 X[D/2]ᵢⱼ +  spin_trace(X[(Dᵅᵅ Dᵅᵝ \\ Dᵝᵅ Dᵝ)])ᵢⱼ }
        :type spin_type: SpinType

        :param use_cuda: Run the calculation on the GPU is available.
        :type use_cuda: bool
        """
        if torch.cuda.is_available() and use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        mol = self.mol

        hamiltonian = HamiltonianSemilocal(
            mol,
            # compute kinetic energy from wavefunctions
            kinetic_functional = None,
            exchange_functional = exchange,
            correlation_functional = correlation,
            exact_exchange_functional = exact_exchange,
            spin_type = spin_type,
            # Increase number of chunks, if you run out of memory.
            grid_chunks = 10
        )

        # Initial guess for matrix densities (from ROHF calculation)
        # If spin type is POLARIZED or UNPOLARIZED, only one Sz component is included
        # in the subspace per spin multiplet. For INVARIANT, all Sz=-S,...,S components
        # are included, leading to much larger matrices.
        msmd = self.initial_matrix_density(spin_type)
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

        # Classify states by spin and spatial multiplicities
        spin_multiplicities = msmd.spin_multiplicity().cpu().detach().numpy()
        if spin_type in [SpinType.INVARIANT, SpinType.INVARIANT_MIX]:
            # Combine energies from degenerate multiplets.
            # e.g. E(S=1,Sz=-1),E(S=1,Sz=0),E(S=1,Sz=+1) ---> E(S=1)
            energies, spin_multiplicities = merge_multiplet_energies(
                energies, spin_multiplicities)
        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        # Save results of calculation to table.
        nrows = len(energies)
        # Add a space to the name of the xc-functional, so that its columns
        # come first when sorted alphabetically.
        xc_name = " "+xc_name

        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": [xc_name] * nrows,
            "spin type": [spin_type.name] * nrows,
            "active space": [(msmd.active_space.nelec, msmd.active_space.norb)] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies[0]) * HARTREE2EV
        }
        df_msdft = pandas.DataFrame.from_dict(data)
        return df_msdft


class Hydrogen(AtomicMultipletCalculation):
    """ Compute the energies of the lowest ²S, ²P, ²S states of the hydrogen atom. """
    def __init__(self, basis='aug-cc-pvdz'):
        # Hydrogen atom
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'H'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The hydrogen atom has only a single electron in the configuration 1s,
        # which is spherically symmetric.
        mol.spin = 1
        mol.charge = 0
        mol.build()
        self._mol = mol

        # Initial guess orbitals are taken from ROHF
        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        rohf.analyze()
        self._rohf = rohf

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for 1²S, 2²S and ²P
        norb = rohf.mo_energy.size
        nelec = (1,0)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
        # Multiplicity of 1²S is 1 (2*L+1=1),
        # multiplicity of 2²S is 2 (2*L+1=1)
        # multiplicity of ²P is 3 (2*L+1=3),
        fci.nroots = 1+1+3
        fci_energies, fcivecs = fci.kernel(nelec=nelec)
        energies = fci_energies
        # spin multiplicities 2*S+1
        spin_multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
        spin_multiplicities = numpy.round(spin_multiplicities, decimals=2)

        # NOTE: The 2¹S and ²P states are not exactly degenerate, because the Gaussian basis
        # set is not complete and the radial parts of s- and p-functions are not identical.
        # Therefore the molecular orbitals 2s and 2p do not have exactly the same energy.
        e = energies
        for i in range(0, len(fcivecs)):
            print(
                'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e[0])*HARTREE2EV, spin_multiplicities[i])
            )
        # Check that we got the expected states.
        numpy.testing.assert_allclose(spin_multiplicities, 1*[2.0] + 1*[2.0] + 3*[2.0])

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 3,
            "method": ["experiment"] * 3,
            "configuration": ["1s", "2s", "2p"],
            "term": [r"1²S", r"2²S", r"1²P"],
            "state index": [1,2,1],
            "spin multiplicity": [2,2,2],
            "spatial multiplicity": [1,1,3],
            r"ΔE (eV)": [0.0, 10.20, 10.20]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # CAS: 1 electrons in 5 orbitals
        nelec = 1
        norb = 5
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False, spin_type=spin_type,
        )
        return msmd


class Helium(AtomicMultipletCalculation):
    def __init__(self, basis='aug-cc-pvdz'):
        # Helium atom
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'He'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # spin = Sz
        mol.spin = 0
        mol.charge = 0
        mol.build()
        self._mol = mol

        # initial guess for orbitals
        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        rohf.analyze()

        self._rohf = rohf

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for lowest two singlet states, ¹S
        norb = rohf.mo_energy.size
        # closed-shell
        nelec = (1,1)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff)
        fci = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, rohf.mo_coeff), shift=.5)
        fci.nroots = 2
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
        spin_multiplicities = numpy.array([1, 1] + [3])

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 3,
            "method": ["experiment"] * 3,
            "configuration": ["1s²", "1s2s", "1s2s"],
            "term": [r"1¹S", r"1³S", r"2¹S"],
            "state index": [1, 1, 2],
            "spin multiplicity": [1, 3, 1],
            "spatial multiplicity": [1, 1, 1],
            r"ΔE (eV)": [0.0] + [19.819614525, 20.615774823]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # CAS: 2 electrons in 2 orbitals (1s,2s)
        nelec = 2
        norb = 2
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False, spin_type=spin_type,
        )
        return msmd

class Lithium(AtomicMultipletCalculation):
    """
    Compute the energies of the lowest doublet ²S and ²P states of the Lithium atom.
    """
    def __init__(self, basis="cc-pvdz"):
        #
        # taken from https://github.com/pyscf/pyscf/blob/master/examples/scf/31-v_atom_rohf.py
        #
        # Spherical symmetry needs to be carefully treated in the atomic calculation.
        # The default initial guess may break the spherical symmetry.  To preserve the
        # spherical symmetry in the atomic calculation, it is often needed to tune the
        # initial guess and SCF model.
        #
        # Construct the atomic initial guess from cation.
        #
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'Li'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The electronic configuration of neutral lithium is 1s²2s
        # The cation Li^{+} has the closed-shell configuration 1s², which is spherically symmetric.
        mol.spin = 0
        mol.charge = 1
        mol.build()

        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        self._rohf = rohf

        # Restore the neutral atom, the ground state is a ²S
        # spin = 2*Sz
        mol.spin = 1
        mol.charge = 0
        mol.build()
        self._mol = mol

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for ²S, ²P
        norb = rohf.mo_energy.size
        nelec = (2,1)
        mol.nelec = nelec
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
        # Multiplicity of ²S is 1 (2*L+1=1),
        # multiplicity of ²P is 3 (2*L+1=3)
        fci.nroots = 1+3
        fci_energies, fcivecs = fci.kernel(nelec=nelec)
        energies = fci_energies
        # spin multiplicities 2*S+1
        spin_multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
        spin_multiplicities = numpy.round(spin_multiplicities, decimals=2)
        e = fci_energies
        for i in range(0, len(fcivecs)):
            print(
                'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e[0])*HARTREE2EV, spin_multiplicities[i])
            )
        # Check that we got the expected states.
        numpy.testing.assert_allclose(spin_multiplicities, 1*[2.0] + 3*[2.0])

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 2,
            "method": ["experiment"] * 2,
            "configuration": ["1s²2s", "1s²2p"],
            "term": [r"1²S", r"1²P"],
            "state index": [1, 1],
            "spin multiplicity": [2,2],
            "spatial multiplicity": [1, 3],
            r"ΔE (eV)": [0.0, 1.85]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # CAS: 1 electrons in 4 orbitals (2s, 2px,2py,2pz)
        nelec = 1
        norb = 4
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False, spin_type=spin_type,
        )
        return msmd


class Beryllium(AtomicMultipletCalculation):
    """
    Compute the energies of the ¹S ground state and the lowest ³P and ¹P excited states
    of the Beryllium atom.
    """
    def __init__(self, basis='cc-pvdz'):
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'Be'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The electronic configuration of neutral beryllium is 1s²2s², which is spherically symmetric.
        mol.spin = 0
        mol.charge = 0
        mol.build()
        self._mol = mol

        # Initial guess from ROHF
        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        self._rohf = rohf

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for ¹S, ³P and ¹P
        norb = rohf.mo_energy.size
        nelec = (2,2)
        mol.nelec = nelec
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
        # Multiplicity of ¹S is 1 (2*L+1=2*0+1=1),
        # multiplicity of ³P is 3 (2*L+1=2*1+1=3),
        # multiplicity of ¹P is 1 (2*L+1=2*1+1=3)
        fci.nroots = 1+3+3
        fci_energies, fcivecs = fci.kernel(nelec=nelec)
        energies = fci_energies
        # spin multiplicities 2*S+1
        spin_multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
        spin_multiplicities = numpy.round(spin_multiplicities, decimals=2)
        e = fci_energies
        for i in range(0, len(fcivecs)):
            print(
                'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e[0])*HARTREE2EV, spin_multiplicities[i])
            )
        # Check that we got the expected states.
        numpy.testing.assert_allclose(spin_multiplicities, [1.0] + 3*[3.0] + 3*[1.0])

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 3,
            "method": ["experiment"] * 3,
            "configuration": ["1s²2s²", "1s²2s2p", "1s²2s2p"],
            "term": [r"1¹S", r"1³P", r"1¹P"],
            "state index": [1,1,1],
            "spin multiplicity": [1,3,3],
            "spatial multiplicity": [1,3,3],
            r"ΔE (eV)": [0.0, 2.73, 5.28]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # CAS: 2 electrons in 4 orbitals (2s,2px,2py,2pz)
        nelec = 2
        norb = 4
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False, spin_type=spin_type,
        )
        return msmd


class Boron(AtomicMultipletCalculation):
    """
    Compute the energies of the lowest doublet (²P) and quadruplet (⁴P) states
    of the boron atom.
    """
    def __init__(self, basis='cc-pvdz'):
        # Construct the atomic initial guess from cation.
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'B'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The electronic configuration of neutral boron is 1s²2s²2p
        # The cation B^{+} has the closed-shell configuration 1s²2s², which is spherically symmetric.
        mol.spin = 0
        mol.charge = 1
        mol.build()

        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        self._rohf = rohf

        # Restore the neutral atom, the ground state is a ²P
        # spin = 2*Sz
        mol.spin = 1
        mol.charge = 0
        mol.build()
        self._mol = mol

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for ²P, ⁴P
        norb = rohf.mo_energy.size
        nelec = (3,2)
        mol.nelec = nelec
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
        # Multiplicity of ²P is 3 (2*L+1=3),
        # multiplicity of ⁴P is 3 (2*L+1=3)
        fci.nroots = 3+3
        fci_energies, fcivecs = fci.kernel(nelec=nelec)
        energies = fci_energies
        # spin multiplicities 2*S+1
        spin_multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
        spin_multiplicities = numpy.round(spin_multiplicities, decimals=2)
        e = fci_energies
        for i in range(0, len(fcivecs)):
            print(
                'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e[0])*HARTREE2EV, spin_multiplicities[i])
            )
        # Check that we got the expected states.
        numpy.testing.assert_allclose(spin_multiplicities, 3*[2.0] + 3*[4.0])

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 2,
            "method": ["experiment"] * 2,
            "configuration": [r"2s²2p", r"2s2p²"],
            "term": [r"1²P", r"1⁴P"],
            "state index": [1,1],
            "spin multiplicity": [2, 4],
            "spatial multiplicity": [3, 3],
            r"ΔE (eV)": [0.0, 3.55]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # CAS: 3 electrons in 4 orbitals (2s,2px,2py,2pz)
        nelec = 3
        norb = 4
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False, spin_type=spin_type,
        )
        return msmd


class Carbon(AtomicMultipletCalculation):
    def __init__(self, basis='cc-pvdz'):
        #
        # taken from https://github.com/pyscf/pyscf/blob/master/examples/scf/31-v_atom_rohf.py
        #
        # Spherical symmetry needs to be carefully treated in the atomic calculation.
        # The default initial guess may break the spherical symmetry.  To preserve the
        # spherical symmetry in the atomic calculation, it is often needed to tune the
        # initial guess and SCF model.
        #
        # Construct the atomic initial guess from cation.
        #
        mol = pyscf.gto.Mole()
        mol.verbose = 4
        mol.atom = 'C'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The electronic configuration of neutral carbon is 1s²2s²2p²
        # The dication C^{2+} has the closed-shell configuration 1s²2s², which is spherically symmetric.
        mol.spin = 0
        mol.charge = 2
        mol.build()

        # initial guess for orbitals from cation
        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        self._rohf = rohf

        # Restore the neutral atom, the ground state is a ³P
        # spin = 2*Sz
        mol.spin = 2
        mol.charge = 0
        self._mol = mol

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for triplet ground state, ³P
        norb = rohf.mo_energy.size
        # high-spin
        nelec = (4,2)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff)
        fci = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, rohf.mo_coeff), shift=.5)
        # Multiplicity of ³P is (2*L+1)=3.
        fci.nroots = 3
        e, c = fci.kernel(nelec=nelec)
        e0 = e[0]
        for i, x in enumerate(c):
            print('state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e0)*HARTREE2EV, fci.spin_square(x, norb, nelec)[1]))
        energies_triplet = e

        # full CI for singlet excited states ¹D and ¹S
        norb = rohf.mo_energy.size
        nelec = (3,3)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=True)
        fci = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, rohf.mo_coeff), shift=.5)
        # Multiplicity of ¹D is (2*L+1)=5 and multiplicity of ¹S is 1.
        fci.nroots = 5+1
        e, c = fci.kernel(nelec=nelec)
        for i, x in enumerate(c):
            print('state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e0)*HARTREE2EV, fci.spin_square(x, norb, nelec)[1]))
        energies_singlet = e

        energies = numpy.hstack([energies_triplet, energies_singlet])
        spin_multiplicities = numpy.array([3]*3 + [1]*5 + [1]*1)

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 3,
            "method": ["experiment"] * 3,
            "configuration": [r"2p²"] * 3,
            "term": [r"1³P", r"1¹D", r"1¹S"],
            "state index": [1, 1, 1],
            "spin multiplicity": [3, 1, 1],
            "spatial multiplicity": [3, 5, 1],
            r"ΔE (eV)": [0.0, 1.26, 2.68]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # MSDFT for triplet ground state ³P and singlet excited states, ¹D and ¹S
        # CAS: 2 electrons in 3 orbitals (px,py,pz)
        nelec = 2
        norb = 3
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False, spin_type=spin_type,
        )
        return msmd


class Nitrogen(AtomicMultipletCalculation):
    """
    Compute the energies of the lowest quadruplet ⁴S and the doublet states ²D and ²P
    of the nitrogen atom.
    """
    def __init__(self, basis='cc-pvdz'):
        # Construct the atomic initial guess from cation.
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'N'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The electronic configuration of neutral nitrogen is 1s²2s²2p³
        # The trication N^{3+} has the closed-shell configuration 1s²2s², which is spherically symmetric.
        mol.spin = 0
        mol.charge = 3
        mol.build()

        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        self._rohf = rohf

        # Restore the neutral atom, the ground state is a ⁴S
        # spin = 2*Sz
        mol.spin = 3
        mol.charge = 0
        mol.build()
        self._mol = mol

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for quadruplet ground state, ⁴S
        norb = rohf.mo_energy.size
        # high-spin
        nelec = (5,2)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff)
        fci = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, rohf.mo_coeff), shift=.5)
        e, c = fci.kernel(nelec=nelec)
        print('E = %.12f  2S+1 = %.7f' %
            (e, fci.spin_square(c, norb, nelec)[1]))
        e0 = e

        # full CI for doublet excited states ²D and ²P
        norb = rohf.mo_energy.size
        nelec = (4,3)
        mol.nelec = nelec
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=True)
        fci = pyscf.fci.addons.fix_spin_(pyscf.fci.FCI(mol, rohf.mo_coeff), shift=.5)
        # Multiplicity of ²D is 5 and multiplicity of ²P is 3.
        fci.nroots = 10
        e, c = fci.kernel(nelec=nelec)
        for i, x in enumerate(c):
            print('state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e0)*HARTREE2EV, fci.spin_square(x, norb, nelec)[1]))

        # Save results of calculation to table.
        energies = numpy.array([e0] + e[0:(5+3)].tolist())
        spin_multiplicities = numpy.array([4]*1 + [2]*5 + [2]*3)

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 3,
            "method": ["experiment"] * 3,
            "configuration": [r"2p³"] * 3,
            "term": [r"1⁴S", r"1²D", r"1²P"],
            "state index": [1, 1, 1],
            "spin multiplicity": [4, 2, 2],
            "spatial multiplicity": [1, 5, 3],
            r"ΔE (eV)": [0.0, 2.38, 3.58]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # CAS: 3 electrons in 3 orbitals (px,py,pz)
        nelec = 3
        norb = 3
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False,
            spin_type=spin_type,
        )
        return msmd


class Oxygen(AtomicMultipletCalculation):
    """
    Compute the energies of the lowest triplet ³P and singlet ¹D and ¹S states
    of the oxygen atom
    """
    def __init__(self, basis='cc-pvdz'):
        # Construct the atomic initial guess from cation.
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'O'
        mol.basis = basis
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The electronic configuration of neutral oxygen is 1s²2s²2p⁴
        # The trication O^{4+} has the closed-shell configuration 1s²2s², which is spherically symmetric.
        mol.spin = 0
        mol.charge = 4
        mol.build()

        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()
        self._rohf = rohf

        # Restore the neutral atom, the ground state is a ³P
        # spin = 2*Sz
        mol.spin = 2
        mol.charge = 0
        mol.build()
        self._mol = mol

    @task(cache_key_fn=cache_key_function)
    def full_configuration_interaction(self):
        mol, rohf = self.mol, self.rohf
        # full CI for ³P, ¹D and ¹S
        norb = rohf.mo_energy.size
        nelec = (4,4)
        mol.nelec = nelec
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
        # Multiplicity of ³P is 3 (2*L+1=3),
        # multiplicity of ¹D is (2*L+1)=5 and multiplicity of ¹S is 1.
        fci.nroots = 3+5+1
        fci_energies, fcivecs = fci.kernel(nelec=nelec)
        energies = fci_energies
        # spin multiplicities 2*S+1
        spin_multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
        spin_multiplicities = numpy.round(spin_multiplicities, decimals=2)
        e = fci_energies
        for i in range(0, len(fcivecs)):
            print(
                'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
                (i, e[i], (e[i]-e[0])*HARTREE2EV, spin_multiplicities[i])
            )
        # Check that we got the expected states.
        numpy.testing.assert_allclose(spin_multiplicities, 3*[3.0] + 5*[1.0] + 1*[1.0])

        (
            energies, term_symbols,
            spin_multiplicities, spatial_multiplicities, state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        nrows = len(energies)
        data = {
            "element": [mol.atom] * nrows,
            "basis": [mol.basis] * nrows,
            "method": ["FCI"] * nrows,
            "term": term_symbols,
            "state index": state_indices,
            "spin multiplicity": spin_multiplicities,
            "spatial multiplicity": spatial_multiplicities,
            "energy (Hartree)": energies,
            r"ΔE (eV)": (energies - energies.min()) * HARTREE2EV
        }
        df_fci = pandas.DataFrame.from_dict(data)
        return df_fci

    def experiment(self):
        # Experimental data comes from the NIST Atomic Spectra Database.
        data_exp = {
            "element": [self.mol.atom] * 3,
            "method": ["experiment"] * 3,
            "configuration": [r"2p⁴"] * 3,
            "term": [r"1³P", r"1¹D", r"1¹S"],
            "state index": [1,1,1],
            "spin multiplicity": [3,1,1],
            "spatial multiplicity": [3,5,1],
            r"ΔE (eV)": [0.0, 1.967, 4.190]
        }
        df_exp = pandas.DataFrame.from_dict(data_exp)
        return df_exp

    def initial_matrix_density(self, spin_type: SpinType):
        # CAS: 4 electrons in 3 orbitals (px,py,pz)
        nelec = 4
        norb = 3
        msmd = MultistateMatrixDensityCAS.from_guess(
            self.mol, norb, nelec,
            guess="hcore",
            spin_symmetry=False, spin_type=spin_type,
        )
        return msmd


available_atoms = [
    Hydrogen,                                                      Helium,
    Lithium, Beryllium, Boron, Carbon, Nitrogen, Oxygen,
]

@flow
def atomic_benchmark_calculations(
    atoms = [Helium(basis="cc-pvdz")],
    xc_functionals_list = [
        # (xc_name, xc_functional_class)
        ('LDA', LDA),
    ],
    spin_types = SpinType
) -> pandas.DataFrame:
    """
    Perform FCI and MSDFT calculation with different multistate functionals
    on atoms and create a table that compares the excitation energies with
    experimental data from NIST's Atomic Spectra Database.

    :param atoms: atoms to be included in the benchmark set
    :type atoms: list of `AtomicMultipletCalculation`

    :param xc_functionals_list: composite multistate functionals that should
        be included in the benchmark calculation.
    :type xc_functionals_list: list of tuples (xc_name, xc_functional_class)
        `xc_name` is the name that will be used in the table,
        `xc_functional_class` is an instance of `PureXCFunctional` or
        `GGAHybridXCFunctional`.

    :param spin_types: Determines how the electronic spin degrees of the matrix
        density are treated when constructing the Hamiltonian, see doc-string
        of `HamiltonianSemilocal`
    :type spin_types: list of SpinType

    :return df: table with FCI, MSDFT and experimental atomic energies
        of the lowest few states for all atoms in the benchmark set
    :type df: pandas.DataFrame
    """
    dataframes = []
    for atom in atoms:
        df_exp = atom.experiment()
        df_exp["spin type"] = "INVARIANT"
        print(df_exp)
        df_fci = atom.full_configuration_interaction()
        df_fci["spin type"] = "INVARIANT"
        print(df_fci)

        dataframes_ = [df_exp, df_fci]
        for spin_type in spin_types:
            for xc_name, xc_functional_class in xc_functionals_list:
                xc_functional = xc_functional_class(atom.mol)
                try:
                    df_msdft = atom.multistate_dft(
                        xc_name,
                        xc_functional.exchange,
                        xc_functional.correlation,
                        xc_functional.exact_exchange,
                        spin_type=spin_type, use_cuda=True
                    )
                except torch.OutOfMemoryError as exception:
                    print(exception)
                    print("GPU does not have enough memory, try to run on CPU...")
                    # Try to run on the CPU, which is slower but has more memory.
                    df_msdft = atom.multistate_dft(
                        xc_name,
                        xc_functional.exchange,
                        xc_functional.correlation,
                        xc_functional.exact_exchange,
                        spin_type=spin_type, use_cuda=False
                    )
                df_msdft["spin type"] = spin_type.name
                print(df_msdft)
                dataframes_.append(df_msdft)
        df_atom = pandas.concat(dataframes_)

        # atomic number
        df_atom["Z"] = pyscf.data.elements.charge(atom.mol.atom)
        # For creating a pivot table the columns 'basis', 'active space' and 'spin type'
        # have to be present even for the experimental data and the FCI calculations.
        df_atom["basis"] = atom.mol.basis
        nelec, nocc = df_msdft["active space"][0]
        df_atom["active space"] = [(nelec, nocc)] * len(df_atom)
        # Map term symbols to confiurations.
        for term, configuration in zip(df_exp["term"], df_exp["configuration"]):
            df_atom.loc[df_atom["term"] == term, "configuration"] = configuration

        dataframes.append(df_atom)

    # combine calculations for different atoms
    df = pandas.concat(dataframes)
    print(df)
    return df


def atomic_pivot_table(df: pandas.DataFrame):
    """
    Present atomic benchmark calculations in a nice looking pivot table
    """
    method_list = df["method"].unique()
    pivot_table = pandas.pivot_table(df,
        values=["energy (Hartree)", "ΔE (eV)"],
        index=["Z", "element", "basis", "active space", "configuration", "term"],
        columns=["method", "spin type"]
    )
    print(pivot_table)
    # sort by experimental excitation energy
    pivot_table.sort_values(by=("ΔE (eV)", "experiment", "INVARIANT"), inplace=True)
    # sort by atomic number Z
    pivot_table.sort_index(
        level="Z",
        sort_remaining=False,
        inplace=True
    )
    # Remove rows with NaN
    pivot_table = pivot_table.dropna()

    # Remove unnecessary indices
    pivot_table = pivot_table.droplevel(level="Z")

    # Compute mean absolute errors for each column relative to FCI (for absolute energies)
    # or experiment (for relative energies).
    mean_errors = []
    for column in pivot_table.columns:
        value = pivot_table[column]
        if column[0] == "energy (Hartree)":
            # Mean absolute error of total energy relative to FCI
            # <|E(MSDFT)-E(FCI)|> (in Hartree)
            reference = pivot_table[("energy (Hartree)", "FCI", "INVARIANT")]
        else:
            # Mean absolute error in excitation energy relative to experiment
            # <|ΔE(MSDFT)-ΔE(experiment)|> (in eV)
            reference = pivot_table[("ΔE (eV)", "experiment", "INVARIANT")]
        mean_error = abs(value-reference).mean()
        mean_errors.append(mean_error)

    errors_row = pandas.Series(data=mean_errors, index=pivot_table.columns)
    pivot_table.loc[("mean absolute error", "", "", "", ""),:] = errors_row

    # Round energies to significant number of digits.
    decimals_Hartree = 4
    decimals_eV = 2
    rounding_settings = {}
    for method in method_list:
        for spin_type in SpinType:
            rounding_settings.update({
                # Total energies
                ("energy (Hartree)", method, spin_type.name): decimals_Hartree,
                # Excitation energies
                ("ΔE (eV)", method, spin_type.name): decimals_eV,
            })
    pivot_table_rounded = pivot_table.round(rounding_settings)
    print(pivot_table_rounded)

    return pivot_table_rounded
