# coding: utf-8
"""
Many-electron states are linear combinations of Slater determinants (configuration)
built from the same set of orthonormal molecular orbitals. The same orbitals are
used from spin-up and spin-down electrons.
"""
import numpy
import warnings

from pyscf.fci.addons import _unpack_nelec
from pyscf.fci.cistring import gen_occslst, make_strings, num_strings
from pyscf.fci.direct_spin1 import trans_rdm1s
from pyscf.fci.spin_op import contract_ss


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

    # CIS or CISD from open-shell reference might contain states with incorrect spin.
    if abs(neleca-nelecb) > 0 and max_level < numpy.inf:
        warnings.warn(
            "configurations_by_excitation_levels: For neleca != nelecb, the subspace spanned "
            f"by determinants with excitation level <= {max_level} contains states with incorrect spin."
        )

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


def occupation_labels(norb=2, nelec=2, max_level=numpy.inf):
    """
    Labels for all configurations in active space (nelec/norb) with at most
    `max_level` excitations.

    The occupation string shows which orbitals are doubly
    occupied (2), singly occupied (a or b) or empty (.)
    e.g.: '222ab...' for a HOMO-LUMO excited determinant.

    :param norb: number of active (spatial) orbitals
    :type norb: int > 0

    :param nelec: number of active electrons, either total number of
        electrons or tuple (neleca, nelecb) with number of up and down electrons
    :type nelec: int or (int, int)

    :param max_level: maximum excitation level relative to the Hartree-Fock
        determinant (0 - only HF, 1 - singly excited, 2 - doubly excited, etc.)
        of configurations that will be included

    :return occupation_strings: labels for all configurations
    :rtype occupation_strings: list of str

    Example
    -------
    # Configurations in a (2,2) active space
    >>> occupation_labels(2,2)
    ['2.', 'ab', 'ba', '.2']
    # Only single excitations
    >>> occupation_labels(2,2, max_level=1)
    ['2.', 'ab', 'ba']
    """
    active_orbitals = range(0, norb)
    neleca, nelecb = _unpack_nelec(nelec)
    occslsta = gen_occslst(active_orbitals, neleca)
    occslstb = gen_occslst(active_orbitals, nelecb)

    selected_confs = configurations_by_excitation_levels(norb, nelec, max_level=max_level)

    occupation_strings = []
    for (ia,ib) in selected_confs:
        occsa = occslsta[ia]
        occsb = occslstb[ib]
        # The occupation string shows which orbitals are doubly
        # occupied (2), singly occupied (a or b) or empty (.)
        # e.g.: '222ab...' for a HOMO-LUMO excited determinant.
        occupation_string = ''
        for o in active_orbitals:
            if o in occsa and o in occsb:
                # orbital is doubly occupied
                occupation_string += '2'
            elif o in occsa:
                # orbital is singly occupied by spin-up electron
                occupation_string += 'a'
            elif o in occsb:
                # orbital is singly occupied by spin-down electron
                occupation_string += 'b'
            else:
                # orbital is unoccupied
                occupation_string += '.'
        occupation_strings.append(occupation_string)
    return occupation_strings


def total_spin_matrix(norb=2, nelec=2, max_level=numpy.inf):
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


def matrix_density_mo(norb=2, nelec=2, max_level=numpy.inf):
    """
    spin matrix density between Slater determinants I and J in the MO basis,

        Dmo[s,q,p,I,J] = <I|pₛ^+ qₛ|J>   with s=0,1 (α,β)

    The spin matrix density in real space is obtained by contracting with the MO
    coefficients MO_{m,p} and with the CI coefficients CI_{I,A}

        Dˢ(r)_{A,B}
            = ∑_{m,n} (
                ∑_{p,q,I,J} Dmo[s,p,q,I,J] MO_{m,p} MO_{n,q} CI_{I,A} CI_{J,A} ) 𝛘m(r) 𝛘n(r)
            = ∑_{m,n} Dao[s,m,n,A,B] 𝛘m(r) 𝛘n(r)

    I,J are Slater determinants, p,q are active molecular orbitals, m,n are
    atomic orbitals, s are the two spin orientations and A,B are electronic eigenstates.

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
        in MO basis for each spin orientation, Dmo[s,q,p,I,J] = <I|pₛ^+ qₛ|J>
    :rtype D_mo: numpy.ndarray of shape (2,norb,norb,nconf,nconf)
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

    # Dmo[s,p,q,I,J] = <I|pₛ^+ qₛ|J>
    D_mo = numpy.zeros((nspin,norb, norb, nconf, nconf))

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
            D_mo[0,:,:,I,J] = rdm1a
            # spin-down part
            D_mo[1,:,:,I,J] = rdm1b

            I += 1
        J += 1

    return D_mo
