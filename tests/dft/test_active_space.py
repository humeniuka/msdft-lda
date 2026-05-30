#!/usr/bin/env python
# coding: utf-8
import numpy
import numpy.testing
import unittest

from pyscf.fci.addons import _unpack_nelec
from pyscf.fci.cistring import make_strings, num_strings
from pyscf.fci.direct_spin1 import trans_rdm1s
from pyscf.fci.spin_op import contract_ss

from mlmsdft.dft.active_space import int2binvec
from mlmsdft.dft.active_space import ActiveSpace
from mlmsdft.dft.active_space import ActiveSpaceError


# ----------------------------------------------------------------
# Wrapper functions to simplify construction of tests.

def occupation_labels(norb=2, nelec=2, max_level=numpy.inf):
    neleca, nelecb = _unpack_nelec(nelec)
    nelec = neleca+nelecb
    # 2*Sz
    spin = neleca-nelecb
    space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=[spin])
    return space.occupation_labels()

def total_spin_matrix(norb=2, nelec=2, max_level=numpy.inf):
    """ total spin between Slater determinants with the same Sz"""
    neleca, nelecb = _unpack_nelec(nelec)
    nelec = neleca+nelecb
    # 2*Sz
    spin = neleca-nelecb
    space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=[spin])
    return space.total_spin_matrix()

def matrix_density_mo(norb=2, nelec=2, max_level=numpy.inf):
    """ matrix density in MO representation between Slater determinants with the same Sz"""
    neleca, nelecb = _unpack_nelec(nelec)
    nelec = neleca+nelecb
    # 2*Sz
    spin = neleca-nelecb
    space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=[spin])
    return space.matrix_density_mo()
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Reference implementations using pyscf functions (only wors when
# all Slater determinants have the same Sz value)

def excitation_levels(norb: int, nelec: int) -> list:
    """
    Compute the excitation levels for the configurations (Slater determinants)
    in the active space (nelec/norb). The excitation level is the number of orbitals
    by which the configuration differs from the determinant where only the lowest
    orbitals are occupied.

    Example: The excitation level of the determinant with the CI string 01011 is 1,
    since it differs by one occupied orbital from the HF determinant 00111.
    Similarly, the excitation level of 11001 would be 2.

    :param norb: number of orbitals in active space
    :param nelec: number of electrons (same spin) in active space

    :return levels: list of excitation levels
    :rtype levels: list of int

    Example
    -------
    # When we put 2 spin-up electrons into 4 orbitals, there are 1 unexcited, 4 singly excited
    # and one doubly excited Slater determinant.
    >>> excitation_levels(4,2)
    [0, 1, 1, 1, 1, 2]
    """
    assert nelec >= 0
    orb_list = range(0, norb)
    # CI strings for the complete active space
    cistrings = make_strings(orb_list, nelec)
    # HF Slater determinants where the lowest orbitals are occupied
    hf_det = (1 << nelec) - 1
    # By how many orbitals does the determinant differ from the HF determinant?
    levels = []
    for det in cistrings:
        # Orbitals that are occupied in the excited Slater determinants
        # which are not occupied in the HF determinant.
        excitations = (det & (~hf_det))
        # The number of bits set in `excitations` is the excitation level
        # (1 - singly excited, 2 - doubly excited etc.)
        level = excitations.bit_count()
        levels.append(level)

    return levels


def configurations_by_excitation_levels(norb=2, nelec=2, max_level=numpy.inf):
    """
    Out of the complete active space that is obtained by distributing `nelec`
    electrons over `norb` orbitals only those configurations are selected that
    differ by at most `max_level` excitations from the Hartree-Fock determinant.

    :param norb: number of active orbitals
    :type norb: int

    :param nelec: number of active electrons
    :type nelec: int or tuple (neleca,nelecb)

    :param max_level: maximum excitation level relative to the Hartree-Fock
        determinant (0 - only HF, 1 - singly excited, 2 - doubly excited, etc.)
        of configurations that will be included
    :type max_level: int >= 0

    :return selected_confs: tuples with indices of selected configurations
        There are separate indices (ia,ib) into the CI strings for up and down electrons.
    :rtype: list of (ia,ib)
    """
    neleca, nelecb = _unpack_nelec(nelec)

    na = num_strings(norb, neleca)
    nb = num_strings(norb, nelecb)
    # Excitation levels (0 - HF, 1 - singles, 2- doubles) of all
    # determinants in the (norb,nelec) active space.
    levelsa = excitation_levels(norb, neleca)
    levelsb = excitation_levels(norb, nelecb)

    # Select Slater determinants ((ia,ib) components of the CI vectors)
    # with excitation level at most `max_level`.
    selected_confs = []
    for ia in range(0, na):
        for ib in range(0, nb):
            if levelsa[ia] + levelsb[ib] <= max_level:
                selected_confs.append((ia,ib))

    return selected_confs


def total_spin_matrix_ref(norb=2, nelec=2, max_level=numpy.inf):
    """
    Matrix elements of the total spin operator S² between the
    Slater determinants (configurations) I and J

        S2[A,B] = <I|S²|J>

    NOTE: This function is very slow.

    :param norb: number of active (spatial) orbitals
    :type norb: int > 0

    :param nelec: number of active electrons, either total number of
        electrons or tuple (neleca, nelecb) with number of up and down electrons
    :type nelec: int or (int, int)

    :param max_level: maximum excitation level of configurations
        relative to the Hartree-Fock determinant (0 - only HF, 1 - singly
        excited, 2 - doubly excited, etc.) that will be included
    :type max_level: int > 0

    :return s2_matrix: matrix of S² operator
    :rtype s2_matrix: numpy.ndarray of shape (nconf,nconf)

    Example
    -------
    # Spin states in (2,2) active space
    >>> s2 = total_spin_matrix(2,2)
    >>> s2
    array([[ 0.,  0.,  0.,  0.],
        [ 0.,  1., -1.,  0.],
        [ 0., -1.,  1.,  0.],
        [ 0.,  0.,  0.,  0.]])
    # 3 singlet states and one triplet (Sz=0)
    >>> numpy.linalg.eigh(s2)[0]
    array([0., 0., 0., 2.])
    """
    # Number of spin-up and spin-down electrons in active space
    neleca, nelecb = _unpack_nelec(nelec)

    na = num_strings(norb, neleca)
    nb = num_strings(norb, nelecb)

    # Select only those configurations which contain at most `max_level`
    # excitation from the HF determinant.
    selected_confs = configurations_by_excitation_levels(norb, nelec, max_level=max_level)
    # number of configurations with the desired excitation level
    nconf = len(selected_confs)

    # matrix s2[I,J] = <I|S²|J>
    s2_matrix = numpy.zeros((nconf, nconf))
    # Loop over kets |J>
    J = 0
    for (ja,jb) in selected_confs:
        # unit CI vector in the direction of |J>
        fcivecJ = numpy.zeros((na,nb))
        fcivecJ[ja,jb] = 1.0
        fcivecJ = fcivecJ.ravel()
        # compute new CI vector S²|J>
        S2_J = contract_ss(fcivecJ, norb, (neleca, nelecb))
        # Loop over bras <I|
        I = 0
        for (ia,ib) in selected_confs:
            # compute <I|S²|J> by projecting S²|J> onto <I|.
            s2_matrix[I,J] = S2_J[ia,ib]

            I += 1
        J += 1

    return s2_matrix


def matrix_density_mo_ref(norb=2, nelec=2, max_level=numpy.inf):
    """
    spin matrix density between Slater determinants I and J in the MO basis,

        Dmo[s,t,q,p,I,J] = <I|pₛ^+ qₜ|J>   with s,t=0,1 (α,β)

    The spin matrix density in real space is obtained by contracting with the MO
    coefficients MO_{m,p} and with the CI coefficients CI_{I,A}

        Dˢᵗ(r)_{A,B}
            = ∑_{m,n} (
                ∑_{p,q,I,J} Dmo[s,t,p,q,I,J] MO_{m,p} MO_{n,q} CI_{I,A} CI_{J,A} ) 𝛘m(r) 𝛘n(r)
            = ∑_{m,n} Dao[s,m,n,A,B] 𝛘m(r) 𝛘n(r)

    I,J are Slater determinants with the same value of Sz, p,q are active molecular orbitals,
    m,n are atomic orbitals, s are the two spin orientations and A,B are electronic eigenstates.

    Since the Sz values of all Slater determinants are the same,
    the mixed-spin blocks are zero, Dᵅᵝ=Dᵝᵅ=0.

    :param norb: number of active (spatial) orbitals
    :type norb: int > 0

    :param nelec: number of active electrons, either total number of
        electrons or tuple (neleca, nelecb) with number of up and down electrons
    :type nelec: int or (int, int)

    :param max_level: maximum excitation level relative to the Hartree-Fock
        determinant (0 - only HF, 1 - singly excited, 2 - doubly excited, etc.)
        of configurations that will be included
    :type max_level: int > 0

    :return D_mo: state and (transition) state density matrices
        in MO basis for each spin orientation, Dmo[s,t,q,p,I,J] = <I|pₛ^+ qₜ|J>
    :rtype D_mo: numpy.ndarray of shape (2,2,norb,norb,nconf,nconf)
    """
    # Number of spin-up and spin-down electrons in active space
    neleca, nelecb = _unpack_nelec(nelec)

    na = num_strings(norb, neleca)
    nb = num_strings(norb, nelecb)

    # Select only those configurations which contain at most `max_level`
    # excitation from the HF determinant.
    selected_confs = configurations_by_excitation_levels(norb, nelec, max_level=max_level)
    # number of configurations with the desired excitation level
    nconf = len(selected_confs)

    # electronic spins, up or down.
    nspin = 2

    # Dmo[s,p,q,I,J] = <I|pₛ^+ qₜ|J>
    D_mo = numpy.zeros((nspin,nspin,norb, norb, nconf, nconf))

    # Loop over kets |J>
    J = 0
    for (ja,jb) in selected_confs:
        # unit CI vector in the direction of |J>
        fcivecJ = numpy.zeros((na,nb))
        fcivecJ[ja,jb] = 1.0
        fcivecJ = fcivecJ.ravel()

        # Loop over bras <I|
        I = 0
        for (ia,ib) in selected_confs:
            # unit CI vector in the direction of |I>
            fcivecI = numpy.zeros((na,nb))
            fcivecI[ia,ib] = 1.0
            fcivecI = fcivecI.ravel()

            # compute <I|pₛ^+ qₛ|J>
            rdm1a, rdm1b = trans_rdm1s(fcivecI, fcivecJ, norb, nelec)
            # spin-up part
            D_mo[0,0,:,:,I,J] = rdm1a
            # spin-down part
            D_mo[1,1,:,:,I,J] = rdm1b
            # mixed-spin blocks are zero
            # D_mo[0,1,...] = 0.0
            # D_mo[1,0,...] = 0.0

            I += 1
        J += 1

    return D_mo
# ----------------------------------------------------------------


class TestActiveSpace(unittest.TestCase):
    def test_int2binvec(self):
        """
        test conversion of integers to binary representation.
        """
        for i, binvec_ref in [
            # Least significant bits come first
            (0, [0,0,0]),
            (1, [1,0,0,0]),
            (3, [1,1,0,0]),
            (8, [0,0,0,1]),
            (9, [1,0,0,1,0]),
            (227, [1,1,0,0,0,1,1,1,0,0])
        ]:
            mbits = len(binvec_ref)
            binvec = int2binvec(i, mbits)
            numpy.testing.assert_array_equal(binvec, binvec_ref)

    def test_raises_exception_incompatible_Sz(self):
        """
        An exception should be raised if the number of spin-up and spin-down values
        is incompatible with the desired values of Sz=(neleca-nelecb) for the
        Slater determinants
        """
        for nelec, spin in [
            # Odd number of electrons but even Sz
            (3, 0),
            # Even number of electrons but odd Sz
            (2, 1)
        ]:
            with self.subTest(nelec=nelec, spin=spin):
                with self.assertRaises(ActiveSpaceError) as err:
                    norb = 4
                    ActiveSpace(norb, nelec, spin_range=[spin])
                # Check error message
                self.assertIn(
                    'For even (odd) number of electrons, 2*Sz must be even (odd)',
                    str(err.exception)
                )

    def test_occupation_labels_same_Sz(self):
        # 2 electrons in 2 orbitals
        self.assertListEqual(
            occupation_labels(norb=2, nelec=2),
            ['2.', 'ab', 'ba', '.2']
        )
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1)),
            ['2.', 'ab', 'ba', '.2']
        )
        # HF, singly and doubly excited determinants
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1), max_level=2),
            ['2.', 'ab', 'ba', '.2']
        )
        # Only HF and singly excited determinants
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1), max_level=1),
            ['2.', 'ab', 'ba']
        )
        # Only HF state
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1), max_level=0),
            ['2.']
        )
        # 2 electrons in 3 orbitals
        self.assertListEqual(
            occupation_labels(norb=3, nelec=2),
            ['2..', 'ab.', 'a.b', 'ba.', '.2.', '.ab', 'b.a', '.ba', '..2']
        )
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(1,1)),
            ['2..', 'ab.', 'a.b', 'ba.', '.2.', '.ab', 'b.a', '.ba', '..2']
        )
        # Only HF and singly excited determinants
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(1,1), max_level=1),
            ['2..', 'ab.', 'a.b', 'ba.', 'b.a']
        )
        # 2 spin-up electrons in 2 orbitals
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(2,0)),
            ['aa']
        )
        # 2 spin-down electrons in 2 orbitals
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(0,2)),
            ['bb']
        )
        # Single spin-up electron in 3 orbitals
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(1,0)),
            ['a..', '.a.', '..a']
        )
        # 2 spin-up and 1 spin-down electrons in 3 orbitals
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(2,1)),
            ['2a.', 'a2.', 'aab', '2.a', 'aba', 'a.2', 'baa', '.2a', '.a2']
        )
        # Only singly excited determinants
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(2,1), max_level=1),
            ['2a.', 'a2.', 'aab', '2.a', 'baa']
        )

    def test_occupation_labels_mixed_Sz(self):
        """
        Test occupation labels when active space contains determinants
        with different spin projections Sz.
        """
        # 1 electron in 1 orbital
        active_space = ActiveSpace(norb=1, nelec=1)
        self.assertListEqual(
            active_space.occupation_labels(),
            ['b', 'a']
        )
        # 2 electrons in 2 orbitals
        active_space = ActiveSpace(norb=2, nelec=2)
        self.assertListEqual(
            active_space.occupation_labels(),
            ['bb', '2.', 'ab', 'ba', '.2', 'aa']
        )

    def test_total_spin_matrix_same_Sz(self):
        """
        Check matrix elements of S² when all Slater determinants have the same Sz value.
        """
        # 2 electrons in 2 orbitals
        s2 = total_spin_matrix(norb=2, nelec=2)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # same with max_level=2
        s2 = total_spin_matrix(norb=2, nelec=2, max_level=2)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # only HF and singly excited determinants -> 2 singlets, 1 triplet (Sz=0)
        s2 = total_spin_matrix(norb=2, nelec=2, max_level=1)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 2.])
        )
        # only HF determinant -> 1 singlet
        s2 = total_spin_matrix(norb=2, nelec=2, max_level=0)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.])
        )
        # 2 electrons in 2 orbitals -> 3 singlets, 1 triplet (Sz=0)
        s2 = total_spin_matrix(norb=2, nelec=(1,1))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # same with max_level=2
        s2 = total_spin_matrix(norb=2, nelec=(1,1), max_level=2)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # only HF and singly excited determinants -> 2 singlets, 1 triplet (Sz=0)
        s2 = total_spin_matrix(norb=2, nelec=(1,1), max_level=1)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 2.])
        )
        # only HF determinant
        s2 = total_spin_matrix(norb=2, nelec=(1,1), max_level=0)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.])
        )

        # 1 electron in 2 orbitals
        s2 = total_spin_matrix(norb=2, nelec=(1,0))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.75, 0.75])
        )
        # 2 spin-up electrons in 2 orbitals -> 1 triplet state
        s2 = total_spin_matrix(norb=2, nelec=(2,0))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([2.0])
        )
        # 2 spin-up electrons and 1 spin-down electron in 2 orbitals -> 2 doublet states
        s2 = total_spin_matrix(norb=2, nelec=(2,1))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.75, 0.75])
        )

        # Possible expectation values of S² for even numbers of electrons.
        even_num_electrons_s2 = []
        for s in [0,1,2,3]:
            s2 = s*(s+1)
            even_num_electrons_s2.append(s2)
        # Possible expectation values of S² for odd numbers of electrons.
        odd_num_electrons_s2 = []
        for s in [0.5, 1.5, 2.5]:
            s2 = s*(s+1)
            odd_num_electrons_s2.append(s2)

        # Check that eigenvalues of S² are in the expected ranges.
        for norb in range(2,6):
            for neleca in range(1, norb):
                for nelecb in range(0, norb-neleca):
                    if abs(neleca-nelecb) == 0:
                        max_levels = [0,1,2,3,4,numpy.inf]
                    else:
                        max_levels = [0,numpy.inf]
                    for max_level in max_levels:
                        with self.subTest(norb=norb, neleca=neleca, nelecb=nelecb, max_level=max_level):
                            s2 = total_spin_matrix(
                                norb=norb, nelec=(neleca,nelecb), max_level=max_level)
                            eigvals, _ = numpy.linalg.eigh(s2)
                            if (neleca + nelecb) % 2 == 0:
                                # even number of electrons
                                for s2_ in eigvals:
                                    self.assertIn(numpy.round(s2_, 2), even_num_electrons_s2)
                            else:
                                # odd number of electrons
                                for s2_ in eigvals:
                                    self.assertIn(numpy.round(s2_, 2), odd_num_electrons_s2)

                            # Compare with reference implementation
                            s2_ref = total_spin_matrix_ref(
                                norb=norb, nelec=(neleca,nelecb), max_level=max_level)
                            numpy.testing.assert_allclose(s2, s2_ref)

    def test_total_spin_matrix_mixed_Sz(self):
        """
        Check matrix elements of S² between all Slater determinants in the active
        space (having all possible Sz values).
        """
        # 2 electrons in 2 orbitals
        active_space = ActiveSpace(norb=2, nelec=2)
        s2 = active_space.total_spin_matrix()
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            # 3 singlets and 3 triplets
            eigvals, numpy.array([0., 0., 0., 2., 2., 2.])
        )

        # Possible expectation values of S² for even numbers of electrons.
        even_num_electrons_s2 = []
        for s in [0,1,2,3]:
            s2 = s*(s+1)
            even_num_electrons_s2.append(s2)
        # Possible expectation values of S² for odd numbers of electrons.
        odd_num_electrons_s2 = []
        for s in [0.5, 1.5, 2.5]:
            s2 = s*(s+1)
            odd_num_electrons_s2.append(s2)

        # Check that eigenvalues of S² are in the expected ranges.
        for norb in range(2,6):
            for nelec in range(0, 2*norb):
                max_levels = [0,numpy.inf]
                for max_level in max_levels:
                    with self.subTest(norb=norb, nelec=nelec, max_level=max_level):
                        active_space = ActiveSpace(norb, nelec, max_level=max_level)
                        s2 = active_space.total_spin_matrix()
                        eigvals, _ = numpy.linalg.eigh(s2)
                        if nelec % 2 == 0:
                            # even number of electrons
                            for s2_ in eigvals:
                                self.assertIn(numpy.round(s2_, 2), even_num_electrons_s2)
                        else:
                            # odd number of electrons
                            for s2_ in eigvals:
                                self.assertIn(numpy.round(s2_, 2), odd_num_electrons_s2)

    def test_spin_projection_sz(self):
        """ Check <Sz> expectation values for a few active spaces. """
        # 1 electron in 1 orbital
        active_space = ActiveSpace(norb=1, nelec=1)
        numpy.testing.assert_array_equal(
            active_space.spin_projection_sz(),
            numpy.array([-0.5, 0.5])
        )
        # 2 electrons in 2 orbitals
        active_space = ActiveSpace(norb=2, nelec=2)
        numpy.testing.assert_array_equal(
            active_space.spin_projection_sz(),
            numpy.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        )

    def test_matrix_density_mo_single_electron_same_Sz(self):
        """
        spin matrix density for 1 electron in 2 orbitals (only Sz=+1/2)
        """
        norb = 2
        neleca = 1
        nelecb = 0
        D_mo = matrix_density_mo(norb=norb, nelec=(neleca,nelecb))
        # compare with reference implementation
        D_mo_ref = matrix_density_mo_ref(norb=norb, nelec=(neleca,nelecb))
        numpy.testing.assert_allclose(D_mo, D_mo_ref)

    def test_matrix_density_mo_same_Sz(self):
        """
        Check that state spin densities integrate to the number of electrons and
        transition densities integrate to zero.
        All Slater determinants have the same Sz=(neleca-nelecb)/2 value.
        """
        for norb in range(2,6):
            for neleca in range(1, norb):
                for nelecb in range(0, norb-neleca):
                    for max_level in [0,1,2,3,numpy.inf]:
                        with self.subTest(
                            norb=norb, neleca=neleca, nelecb=nelecb, max_level=max_level
                        ):
                            D_mo = matrix_density_mo(norb=norb, nelec=(neleca,nelecb), max_level=max_level)
                            # compare with reference implementation
                            D_mo_ref = matrix_density_mo_ref(norb=norb, nelec=(neleca,nelecb), max_level=max_level)
                            numpy.testing.assert_allclose(D_mo, D_mo_ref)

                            # Check that
                            # ∫ Dˢˢ(r)_{I,J} dr = ∑_{p} Dmo[s,s,p,p,I,J] = nelec[s] δ_{I,J}
                            nelec = (neleca, nelecb)
                            nspin = D_mo.shape[0]
                            nstate = D_mo.shape[-1]
                            for s in range(0, nspin):
                                numpy.testing.assert_allclose(
                                    numpy.einsum('ppIJ->IJ', D_mo[s,s,...]), nelec[s] * numpy.eye(nstate)
                                )

    def test_matrix_density_1e_in_1o_mixed_Sz(self):
        """
        Check matrix densoty for doublet ground state of hydrogen atom.

        The hydrogen atom has only n = 1 electron which can be in either of the two spin states.
        The spinor wavefunctions for spin up and spin down are denoted by
            α=(1) and β=(0)
              (0)       (1)

        The doublet ground state is two-fold degenerate forming a subspace of N = 2 states.
            Ψ1(r,s) = ϕ1s(r) β(s)  with Sz = -1/2
            Ψ2(r,s) = ϕ1s(r) α(s)  with Sz = +1/2

        The spin matrix density is

            D = n (Ψ1 Ψ1ᵀ  Ψ1 Ψ2ᵀ) = n |ϕ1s(r)|² (β·βᵀ  β·αᵀ)
                  (Ψ2 Ψ1ᵀ  Ψ2 Ψ2ᵀ)               (α·βᵀ  α·αᵀ)

        The outer products of the spinors are

            β·βᵀ = (0) · (0 1) = (0 0)
                   (1)           (0 1)

            β·αᵀ = (0) · (1 0) = (0 0)
                   (1)           (1 0)

            α·βᵀ = (1) · (0 1) = (0 1)
                   (0)           (0 0)

            α·αᵀ = (1) · (1 0) = (1 0)
                   (0)           (0 0)

        The spin matrix density can be written as N x N block matrix
        of 2x2 spin blocks:

            Sz= -1/2  +1/2
             α  β  α  β       Sz=
            [0, 0, 0, 0],  α  -1/2
            [0, 1, 1, 0],  β
            [0, 1, 1, 0],  α  +1/2
            [0, 0, 0, 0]   β

        Alternatively it can be written as a 2x2 block matrix of
        NxN spin blocks:

            (Dᵅᵅ Dᵅᵝ)
            (Dᵝᵅ  Dᵝᵝ)

              α       β
            -½ +½   -½ +½
            [0, 0,   0, 0],  -½ α
            [0, 1,   1, 0],  +½ α
            [0, 1,   1, 0],  -½ β
            [0, 0,   0, 0],  +½ β

        Incidentally for this example where N=2, the two types of
        arrangements are the same.
        """
        norb = 1
        nelec = 1
        active_space = ActiveSpace(norb, nelec)
        D = active_space.matrix_density_mo()
        # Build the (2N)x(2N) spin matrix density
        #   (Dᵅᵅ Dᵅᵝ)
        #   (Dᵝᵅ  Dᵝᵝ)
        spin_D = numpy.block([
            [D[0,0,...], D[0,1,...]],
            [D[1,0,...], D[1,1,...]]
        ]).squeeze()
        # expected spin matrix density
        spin_D_ref = numpy.array([
        #      α      β
        #    -½ +½  -½ +½   #
            [0, 0,   0, 0], # -½ α
            [0, 1,   1, 0], # +½ α
            [0, 1,   1, 0], # -½ β
            [0, 0,   0, 0], # +½ β
        ])
        numpy.testing.assert_allclose(spin_D, spin_D_ref)

    def test_matrix_density_mo_mixed_Sz(self):
        """
        Check that state spin densities integrate to the number of electrons and
        transition densities integrate to zero.
        """
        for norb in range(2,6):
            for nelec in range(0, 2*norb):
                for max_level in [0,1,2,3,numpy.inf]:
                    with self.subTest(norb=norb, nelec=nelec, max_level=max_level):
                        # `spin_range=None` means to take all possible Sz values.
                        active_space = ActiveSpace(
                            norb, nelec, max_level=max_level, spin_range=None)
                        D_mo = active_space.matrix_density_mo()

                        # Check that
                        # ∫ ∑_{s} Dˢˢ(r)_{I,J} dr = ∑_{s} ∑_{p} Dmo[s,s,p,p,I,J] = δ_{I,J}
                        nstate = D_mo.shape[-1]
                        numpy.testing.assert_allclose(
                            numpy.einsum('ssppIJ->IJ', D_mo), nelec * numpy.eye(nstate)
                        )

    def test_string_representation(self):
        """ Check that __repr__ and __str__ methods work """
        active_space = ActiveSpace(2, 2)
        repr(active_space)
        str(active_space)


if __name__ == "__main__":
    unittest.main()
