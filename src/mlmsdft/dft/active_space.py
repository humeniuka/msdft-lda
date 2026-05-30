# coding: utf-8
"""
Many-electron states are linear combinations of Slater determinants (configuration)
built from the same set of orthonormal molecular orbitals. The same orbitals are
used from spin-up and spin-down electrons.
"""
import numpy
from typing import List
import warnings

from pyscf.fci.addons import _unpack_nelec
from pyscf.fci.cistring import make_strings


class ActiveSpaceError(ValueError):
    pass


def int2binvec(i: int, mbits: int):
    """
    convert the integer `i` to a vector of bits,
    where the least significant bits come first
    and which is padded with zeros up to length `mbits`

    :param i: integer whose binary representation is sought
    :type i: int

    :param mbits: length of bit vector
    :type mbints: int

    :return binvec: binary representation of `i`
    :rtype binvec: array of 0s and 1s
    """
    vi = list(map(int, list(bin(i)[2:])))
    pad = [0 for j in range(0, mbits-len(vi))]
    binvec = pad + vi
    # least significant bits should come first
    binvec.reverse()
    return numpy.array(binvec)


def s2_slater_determinants(detI, detJ, spatial, spins):
    """
    compute matrix element of total spin operator S²
    between two Slater determinants, <I|S²|J>

    :param detI, detJ: bra (I) and ket (J) Slater derminants,
        the alpha and beta occupations of the molecular orbitals are encoded in
        the bit-strings of the integers
    :type detI, detJ: int

    :param spatial: 0-based indices into the spatial parts of the molecular orbitals
        e.g. [0,1,2,3, 0,1,2,3]
              |--β--|  |--α--|
    :type spatial: list of int

    :param spins: direction of the electronic spins of each spin orbital
        (+1 for alpha, -1 for beta)
        e.g. [-1,-1,-1,-1, 1,1,1,1]
              |----β----|  |--α--|
    :type spins: list of int

    :return s2: matrix element <I|S²|J>
    :rtype s2: float
    """
    # number of spin molecular orbitals
    nmo = len(spins)

    # binary representation of determinants
    Ibin = int2binvec(detI, nmo)
    Jbin = int2binvec(detJ, nmo)
    # total number of electrons
    nelec = numpy.sum(Ibin)

    # occupied spin orbitals in detI which are not in detJ
    dif_orbsI = numpy.where(int2binvec(detI&(detI^detJ), nmo) == 1)[0]
    dif_orbsJ = numpy.where(int2binvec(detJ&(detJ^detI), nmo) == 1)[0]

    S2 = 0.0
    phase = 1.0
    if len(dif_orbsI) == 0 and len(dif_orbsJ) == 0:
        # one body part
        S2 += 3.0/4.0 * nelec
        # two body part
        for i in range(0, nmo):
            for j in range(0, nmo):
                if i != j:
                    dS2 = 1.0/4.0 * spins[i] * spins[j]
                    if (spatial[i] == spatial[j]) and (spins[i] * spins[j] < 0.0):
                        dS2 -= 1.0/2.0
                    S2 += Ibin[i]*Jbin[j] * dS2
    elif len(dif_orbsI) == 1 and len(dif_orbsJ) == 1:
        pass
    elif len(dif_orbsI) == 2 and len(dif_orbsJ) == 2:
        r,s = dif_orbsJ
        p,q = dif_orbsI
        # sign changes from moving creation operators to the front
        phase = pow(-1,
            numpy.sum(Jbin[0:r]) +
            numpy.sum(Jbin[0:s]) +
            numpy.sum(Ibin[0:p]) +
            numpy.sum(Ibin[0:q])
        )
        if spatial[p] == spatial[r] and spatial[q] == spatial[s]:
            if spins[p] == +1 and spins[q] == -1 and spins[r] == -1 and spins[s] == +1:
                S2 += 1.0
            if spins[p] == -1 and spins[q] == +1 and spins[r] == +1 and spins[s] == -1:
                S2 += 1.0
        if spatial[p] == spatial[s] and spatial[q] == spatial[r]:
            if spins[p] == +1 and spins[q] == -1 and spins[r] == +1 and spins[s] == -1:
                S2 -= 1.0
            if spins[p] == -1 and spins[q] == +1 and spins[r] == -1 and spins[s] == +1:
                S2 -= 1.0
    else:
        pass
    return (phase * S2)

def trans_rdm1s_slater_determinants(
    detI: int, detJ: int, spatial: List[int], spins: List[int]):
    """
    compute matrix element of the density operator pₛ^+ qₜ
    between two Slater determinants, <I|pₛ^+ qₜ|J>

    :param detI, detJ: bra (I) and ket (J) Slater derminants,
        the alpha and beta occupations of the molecular orbitals are encoded in
        the bit-strings of the integers
    :type detI, detJ: int

    :param spatial: 0-based indices into the spatial parts of the molecular orbitals
        e.g. [0,1,2,3, 0,1,2,3]
              |--β--|  |--α--|
    :type spatial: list of int

    :param spins: direction of the electronic spins of each spin orbital
        (+1 for alpha, -1 for beta)
        e.g. [-1,-1,-1,-1, 1,1,1,1]
              |----β----|  |--α--|
    :type spins: list of int

    :return rdm1aa, rdm1ab, rdm1ba, rdm1bb: one-particle reduced (transition) density
        matrices Dᵅᵅ, Dᵅᵝ, Dᵝᵅ, Dᵝᵝ in the MO basis.
        Dˢᵗ[q,p] = <I|pₛ^+ qₜ|J>
    :rtype rdm1aa, rdm1ab, rdm1ba, rdm1bb: 4 numpy.ndarray's of shape (norb,norb) each
    """
    # number of spin molecular orbitals
    nmo = len(spins)
    # number of spatial orbitals
    norb = nmo//2
    assert nmo == 2*norb

    # binary representation of determinants
    Ibin = int2binvec(detI, nmo)
    Jbin = int2binvec(detJ, nmo)

    # Check total number of electrons
    nelecI = numpy.sum(Ibin)
    nelecJ = numpy.sum(Jbin)
    if nelecI != nelecJ:
        raise ValueError(
            "Bra and ket determinant must have same total number of electrons, "
            f"got {numpy.sum(Ibin)} != {numpy.sum(Jbin)}"
        )

    # occupied spin orbitals in detI which are not in detJ
    dif_orbsI = numpy.where(int2binvec(detI&(detI^detJ), nmo) == 1)[0]
    # occupied spin orbitals in detJ which are not in detI
    dif_orbsJ = numpy.where(int2binvec(detJ&(detJ^detI), nmo) == 1)[0]

    # αα part of reduced 1-particle (transition) density matrix
    rdm1aa = numpy.zeros((norb,norb))
    # αβ part
    rdm1ab = numpy.zeros((norb,norb))
    # βα part
    rdm1ba = numpy.zeros((norb,norb))
    # ββ part
    rdm1bb = numpy.zeros((norb,norb))

    if len(dif_orbsI) == 0 and len(dif_orbsJ) == 0:
        # bra and ket are the same, |I> = |J>
        for i in range(0, nmo):
            # i is a spin orbital and m is a spatial orbital index.
            m = spatial[i]
            if Ibin[i] > 0:
                if spins[i] == +1:
                    # orbital i is spin-up
                    rdm1aa[m,m] = 1.0
                elif spins[i] == -1:
                    # orbital i is spin-down
                    rdm1bb[m,m] = 1.0
                else:
                    raise ValueError("`spins` orientation must be -1 or +1")
        # rdm1ab and rdm1ba are zero since pₛ^+ qₜ changes Sz=(neleca-nelecb)/2
        # if the spins s != t.
    elif len(dif_orbsI) == 1 and len(dif_orbsJ) == 1:
        # bra and ket differ by one spin orbital
        ps, = dif_orbsI
        qt, = dif_orbsJ
        # sign changes from moving creation operators to the front
        phase = pow(-1, numpy.sum(Ibin[0:ps]) + numpy.sum(Jbin[0:qt]))
        # spatial indices of spin orbitals
        p = spatial[ps]
        q = spatial[qt]
        # spin direction (+1 for alpha, -1 for beta)
        s = spins[ps]
        t = spins[qt]
        # According to Eqn. 5.4.21 in McWeeny's book, the (transition) density matrices
        # are defined as
        #   Dˢᵗ[q,p] = <I|pₛ^+ qₜ|J>
        if (s == +1) and (t == +1):
            rdm1aa[q,p] = phase
        elif (s == +1) and (t == -1):
            rdm1ab[q,p] = phase
        elif (s == -1) and (t == +1):
            rdm1ba[q,p] = phase
        elif (s == -1) and (t == -1):
            rdm1bb[q,p] = phase
        else:
            raise ValueError("`spins` orientation must be -1 or +1")
    else:
        pass

    return rdm1aa, rdm1ab, rdm1ba, rdm1bb


class ActiveSpace:
    def __init__(self, norb: int, nelec: int, max_level=numpy.inf, spin_range=None):
        """
        Slater determinants built from `nelec` electrons in `norb` orbitals.

        Out of the complete active space that is obtained by distributing `nelec`
        electrons over `norb` spatial orbitals only those configurations are selected that
        differ by at most `max_level` excitations from the Hartree-Fock determinant.
        A distribution of spin-up and spin-down electrons is accepted if for the expectation
        value Sz of the z-component of the total spin vector, 2*Sz=(neleca - nelecb)
        lies within `spin_range`.

        :param norb: number of active spatial orbitals
        :type norb: int

        :param nelec: number of active electrons (spin up and spin down)
        :type nelec: int or tuple (neleca,nelecb)

        :param max_level: maximum excitation level relative to the Hartree-Fock
            determinant (0 - only HF, 1 - singly excited, 2 - doubly excited, etc.)
            of configurations that will be included
        :type max_level: int >= 0

        :param spin_range: Acceptable values of 2*Sz=neleca-nelecb,
            e.g. spin_range=[-1,0,1]
            If spin_range == None, all possible 2*Sz values are considered,
            spin_range=range(-nelec,nelec+1,2)
            Examples:
                1 electron  -> spin_range = [-1,1]      -> Sz = [-1/2, +1/2]
                2 electrons -> spin_range = [-2,0,2]    -> Sz = [-1,0,+1]
                3 electrons -> spin_range = [-3,-1,1,3] -> Sz = [-3/2,-1/2,+1/2,+3/2]
        :type spin_range: list of int or None
        """
        self.norb = norb
        self.max_level = max_level

        # Number of spin-up and spin-down electrons in active space
        neleca, nelecb = _unpack_nelec(nelec)
        nelec = neleca+nelecb
        self.nelec = nelec
        # CIS or CISD from open-shell reference might contain states with incorrect spin.
        if abs(neleca-nelecb) > 0 and max_level < numpy.inf:
            warnings.warn(
                "For neleca != nelecb, the subspace spanned by determinants with "
                f"excitation level <= {max_level} might contain states with incorrect spin."
            )

        if spin_range is None:
            # Consider all possible Sz values for `nelec` electrons.
            #   2*Sz = -nelec  :  all electrons are in spin-down orbials
            #   2*Sz = -nelec+1
            #   ...
            #   2*Sz = nelec-1
            #   2*Sz = nelec   :  all electrons are in spin-up orbitals
            spin_range = range(-nelec, nelec+1, 2)
        # Check that spin range is valid. If the number of electrons is even,
        for spin in spin_range:
            if (spin % 2 != self.nelec % 2):
                raise ActiveSpaceError(
                    "For even (odd) number of electrons, 2*Sz must be even (odd), "
                    f"got spin_range={spin_range}"
                )
        self.spin_range = list(spin_range)

    def slater_determinants(self) -> List[int]:
        """
        Generate list of all Slater determinants in the active space.

        The determinants are encoded as integers in binary format.
        If the i-th bit is set, the i-th spin orbital is occupied.
        The `norb` least significant bits are the occupation numbers of the
        spin-down orbitals followed by the occupation numbers for spin-up
        orbitals (reading the bitstring from right to left)

        For a single value of Sz the order of the determinants is the same
        as in pyscf's FCI code.

        Example:
            5 electrons in 5 orbitals (neleca=3, nelecb=2)
               HF det = 227 = 0b11100011
               -> split into alpha and beta strings:
                  beta  occupation string = 00011 (less significant half)
                  alpha occupation string = 00111 (more significant half)
               -> If i-th bit is set, the i-th spatial orbital is occupied:
                  spatial orbitals occupied by beta electrons  = [0,1]
                  spatial orbitals occupied by alpha electrons = [0,1,2]
                  NOTE: The least significant bit (i=0) is the right-most one.

        :return determinants: list of integers whose bit representation encodes the
            orbital occupation numbers
        :rtype determinants: list of int
        """
        determinants = []
        # List of spatial orbitals
        orb_list = range(0, self.norb)
        for neleca in range(0, self.nelec+1):
            nelecb = self.nelec - neleca
            # Select subspace of Slater determinants based on Sz value.
            Sz = (neleca - nelecb)/2
            if int(2*Sz) not in self.spin_range:
                continue
            # HF Slater determinant where the lowest orbitals are occupied.
            # HF determinant for spin-up electrons
            hf_det_a = (1 << neleca)-1
            # ... and for spin-down electrons
            hf_det_b = (1 << nelecb)-1
            # combined HF determinant,
            # beta occupation are stored in the less significant half
            # of the bit string, alpha occupation in the more significant half.
            hf_det = (hf_det_a << self.norb) + hf_det_b
            # alpha strings
            cistrings_a = make_strings(orb_list, neleca)
            # beta strings
            cistrings_b = make_strings(orb_list, nelecb)
            for det_a in cistrings_a:
                for det_b in cistrings_b:
                    # combine alpha and beta strings into single determinant
                    det = (int(det_a) << self.norb) + int(det_b)
                    # Orbitals that are occupied in the excited Slater determinants
                    # which are not occupied in the HF determinant.
                    excitations = (det & (~hf_det))
                    # The number of bits set in `excitations` is the excitation level
                    # (1 - singly excited, 2 - doubly excited etc.)
                    level = excitations.bit_count()
                    # Select Slater determinants with excitation level at most `max_level`.
                    if level <= self.max_level:
                        determinants.append(det)
        return determinants

    def occupation_labels(self) -> List[str]:
        """
        Labels for all configurations in active space.

        The occupation string shows which orbitals are doubly
        occupied (2), singly occupied (a or b) or empty (.)
        e.g.: '222ab...' for a HOMO-LUMO excited determinant.

        :return occupation_strings: labels for all configurations
        :rtype occupation_strings: list of str

        Example
        -------
        # Configurations in a (2,2) active space, no spin polarization
        >>> space = ActiveSpace(2,2,spin_range=[0])
        >>> space.occupation_labels()
        ['2.', 'ab', 'ba', '.2']
        # Only single excitations
        >>> space = ActiveSpace(2,2,spin_range=[0], max_level=1)
        >>> space.occupation_labels()
        ['2.', 'ab', 'ba']
        # All determinants
        >>> space = space = ActiveSpace(2,2)
        >>> space.occupation_labels()
        # Sz=                          Sz=
        # -1 |      Sz=0              | +1
        ['aa', '2.', 'ab', 'ba', '.2', 'bb']
        """
        determinants = self.slater_determinants()
        occupation_strings = []
        for det in determinants:
            occs = int2binvec(det, 2*self.norb)
            # Indices of spatial orbitals occupied by spin-down electrons come first.
            occsb = numpy.where(occs[:self.norb] == 1)[0]
            # Indices of spatial orbitals occupied by spin-up electrons
            occsa = numpy.where(occs[self.norb:] == 1)[0]
            # The occupation string shows which orbitals are doubly
            # occupied (2), singly occupied (a or b) or empty (.)
            # e.g.: '222ab...' for a HOMO-LUMO excited determinant.
            occupation_string = ''
            for o in range(0, self.norb):
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

    def total_spin_matrix(self) -> numpy.ndarray:
        """
        Matrix elements of the total spin operator S² between the
        Slater determinants (configurations) I and J

            S2[A,B] = <I|S²|J>

        :return s2_matrix: matrix of S² operator
        :rtype s2_matrix: numpy.ndarray of shape (ndet,ndet)

        Example
        -------
        # Spin states in (2,2) active space
        >>> space = ActiveSpace(2,2, spin_range=[0])
        >>> s2 = space.total_spin_matrix()
        >>> s2
        array([[ 0.,  0.,  0.,  0.],
            [ 0.,  1., -1.,  0.],
            [ 0., -1.,  1.,  0.],
            [ 0.,  0.,  0.,  0.]])
        # 3 singlet states and one triplet (Sz=0)
        >>> numpy.linalg.eigh(s2)[0]
        array([0., 0., 0., 2.])
        """
        active_orbitals = list(range(0, self.norb))
        # Indices into spatial orbitals
        spatial = active_orbitals + active_orbitals
        # Spin of spin orbitals (+1 -> up, -1 -> down)
        spins = [-1] * self.norb + [1] * self.norb

        determinants = self.slater_determinants()
        # Number of determinants
        ndet = len(determinants)

        # matrix s2[I,J] = <I|S²|J>
        s2_matrix = numpy.zeros((ndet, ndet))
        # Loop over bras <I|
        for I in range(0, ndet):
            detI = determinants[I]
            # Loop over kets |J>
            for J in range(0, ndet):
                detJ = determinants[J]
                s2_matrix[I,J] = s2_slater_determinants(detI, detJ, spatial, spins)

        return s2_matrix

    def spin_projection_sz(self) -> numpy.ndarray:
        """
        expectation values of z-component of spin vector for
        Slater determinants I in active space, <I|Sz|I>.

        :return sz: projection of spin vector on quantization axis for each
            determinant, sz[I] = <I|Sz|I>
        :rtype sz: numpy.ndarray of shape (,ndet)
        """
        determinants = self.slater_determinants()
        # Number of determinants
        ndet = len(determinants)
        sz = numpy.zeros(ndet)
        for I in range(0, ndet):
            detI = determinants[I]
            occs = int2binvec(detI, 2*self.norb)
            # number of occupied beta orbitals
            nelecb = numpy.sum(occs[:self.norb])
            # number of occupied alpha orbitals
            neleca = numpy.sum(occs[self.norb:])
            # spin projection on z-axis
            sz[I] = (neleca-nelecb)/2.0
        return sz

    def matrix_density_mo(self) -> numpy.ndarray:
        """
        spin matrix density between Slater determinants I and J in the MO basis,

            Dmo[s,t,q,p,I,J] = <I|pₛ^+ qₜ|J>   with s,t=(0,0), (0,1), (1,0), (1,1)
                                                         αα,    αβ,    βα,     ββ

        The spin matrix density in real space is obtained by contracting with the MO
        coefficients MO_{m,p} and with the CI coefficients CI_{I,A}

            Dˢᵗ(r)_{A,B}
                = ∑_{m,n} (
                    ∑_{p,q,I,J} Dmo[s,t,p,q,I,J] MO_{m,p} MO_{n,q} CI_{I,A} CI_{J,A} ) 𝛘m(r) 𝛘n(r)
                = ∑_{m,n} Dao[s,t,m,n,A,B] 𝛘m(r) 𝛘n(r)

        I,J are Slater determinants, p,q are active molecular orbitals, m,n are
        atomic orbitals, s and t are the spin orientations and A,B are electronic eigenstates.

        :return D_mo: state and (transition) state density matrices
            in MO basis for each spin orientation, Dmo[s,t,q,p,I,J] = <I|pₛ^+ qₜ|J>
        :rtype D_mo: numpy.ndarray of shape (2,2,norb,norb,ndet,ndet)
        """
        active_orbitals = list(range(0, self.norb))
        # Indices into spatial orbitals
        spatial = active_orbitals + active_orbitals
        # Spin of spin orbitals (+1 -> up, -1 -> down)
        spins = [-1] * self.norb + [1] * self.norb

        determinants = self.slater_determinants()
        # Number of determinants
        ndet = len(determinants)
        # electronic spins, up or down.
        nspin = 2

        # Dmo[s,t,q,p,I,J] = <I|pₛ^+ qₜ|J>
        D_mo = numpy.zeros((nspin,nspin, self.norb,self.norb, ndet,ndet))
        # Loop over bras <I|
        for I in range(0, ndet):
            detI = determinants[I]
            # Loop over kets |J>
            for J in range(0, ndet):
                detJ = determinants[J]
                # compute <I|pₛ^+ qₜ|J>
                rdm1aa,rdm1ab,rdm1ba,rdm1bb = trans_rdm1s_slater_determinants(
                    detI, detJ, spatial, spins)
                # αα part
                D_mo[0,0,:,:,I,J] = rdm1aa
                # αβ part
                D_mo[0,1,:,:,I,J] = rdm1ab
                # βα part
                D_mo[1,0,:,:,I,J] = rdm1ba
                # ββ part
                D_mo[1,1,:,:,I,J] = rdm1bb

        return D_mo

    def __repr__(self):
        text = (
            self.__class__.__name__ +
            f"(active orbitals: {self.norb}," +
            f" active electrons: {self.nelec}," +
            f" maximum excitation level: {self.max_level})"
        )
        return text
