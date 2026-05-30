# coding: utf-8
"""
Parameterizations of the matrix density in terms of linear combinations of
spin-adapted configurations that are built from the same set of orthonormal
molecular orbitals. The molecular orbital coefficients depend on orbital
rotations

    C(U) = C₀ U = C₀ exp(R)

where R = -Rᵀ is an antisymmetric parameter matrix, so that U=exp(R) is unitary.
C₀ is an initial guess.
"""
from abc import ABC, abstractmethod
import numpy
import scipy.linalg
import torch
from torch import Size, Tensor
import torch.linalg

from pyscf.dft import numint
from pyscf.fci.addons import _unpack_nelec
import pyscf.gto
import pyscf.scf
from pyscf.scf.hf import get_hcore

from mlmsdft.dft.active_space import ActiveSpace
from mlmsdft.dft.active_space import ActiveSpaceError
from mlmsdft.dft.spin import SpinType


class MultistateMatrixDensity(torch.nn.Module, ABC):
    """
    Base class for multistate matrix density.
    """
    def __init__(self, mol: pyscf.gto.Mole):
        super().__init__()
        self.mol = mol

    @property
    @abstractmethod
    def number_of_states(self):
        pass

    @property
    def number_of_electrons(self):
        return self.mol.tot_electrons()

    @property
    def device(self):
        """ Identify device where parameters of matrix density are stored """
        device_ = torch.device('cpu')
        # In principle different parameters could be on different devices,
        # but we assume all are on the same device.
        for param in self.parameters():
            device_ = param.device
            break
        return device_

    @abstractmethod
    def density_matrices_ao(self) -> Tensor:
        """
        construct the density matrices for the state and transition densities
        in the basis of atomic orbitals, Dˢᵦᵧᵢⱼ, such that the matrix density
        can be expressed as

            Dˢᵢⱼ(r) = ∑ᵦ ∑ᵧ Dˢᵦᵧᵢⱼ 𝛘ᵦ(r) 𝛘ᵧ(r),

        where i,j = 1,...,Nstate enumerate electronic states, β,γ=1,...,Nbasis
        enumerate atomic orbitals 𝛘(r) and s distinguishes between the matrix
        densities for spin up (s=0) and spin down (s=1).

        :return dm: state (i==j) and transition (i != j) density matrices in
            in the AO basis, dm[s,b,g,i,j] = Dˢᵦᵧᵢⱼ
        :rtype dm: Tensor of shape (2,Nbasis,Nbasis,Nstate,Nstate)
        """
        pass

    @abstractmethod
    def basis_transformation(self, L: Tensor):
        """
        apply a basis transformation to the matrix density

            D'(r) = L D(r) Lᵀ

        :param L: orthogonal matrix representing the basis transformation
            among the electronic states
        :type L: Tensor of shape (Nstate,Nstate)

        NOTE: The parameters of the matrix density can be modified in-place
        and detached from the computational graph.
        """
        pass

    @abstractmethod
    def spin_multiplicity(self):
        """
        Spin multiplicity (2*S+1) of spin multiplets.

        Pyscf only calculates the Sz=0 component of a multiplet
        Sz=-S,-S+1,...,0,...,S-1,S. The other states of the multiplet have the
        same charge density and are accounted for by weighting the energy
        with the spin multiplicity.

        :return multiplicity: spin multiplicity 2*S+1
        :rtype multiplicity: int Tensor of shape (Nstate,)
        """
        pass

    def state_weights(self):
        """
        Multiplicities (due to spin degeneracy) that are used to weight the
        state energies when calculating the subspace energy.

        :return weights: weights of state in subspace energy
        :rtype weights: int Tensor of shape (Nstate,)
        """
        return self.spin_multiplicity()

    # There is no `forward` method. This is because the matrix density is not
    # differentiable with respect to the grid coordinates,
    # because the atomic orbitals are evaluated using pyscf and numpy.
    def evaluate(self, coords: numpy.ndarray, dm_ao: Tensor = None, need_laplacian=True):
        """
        evaluate the multistate spin matrix density Dˢᵗ(r), its gradient ∇Dˢᵗ(r)
        and its Laplacian ∇²Dˢᵗ(r) on a grid.

        Nspin is the number of spin channels (one for spin-up the other for spin-down)
        Nstate is the number of electronic states
        Nbasis is the number of basis functions (atomic orbitals)
        Ncoord is the number of grid points

        :param coords: The Cartesian coordinates of the grid r
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param dm_ao: state (i==j) and transition (i != j) density matrices in
            in the AO basis, dm_ao[s,t,b,g,i,j] = Dˢᵗᵦᵧᵢⱼ
            If no `dm_ao` is provided (None), the matrix density is computed using
            the member function `density_matrices_ao()`.
        :type dm_ao: Tensor of shape (2,2,Nbasis,Nbasis,Nstate,Nstate) or None

        :param need_laplacian: Whether to calculate the Laplacian or not
            If False, the returned lapl_D is None.
            Computing the Laplacian increases the memory footprint by an order
            of magnitude because of the unfortunate way it is calculated using
            einsum(...).
        :type need_laplacian: bool

        :return: D, grad_D, lapl_D
        :rtype: tuple of Tensor
            `D` - Tensor of shape (Nspin,Nspin,Ncoord,Nstate,Nstate),
                blocks of spin matrix density Dᵅᵅᵢⱼ(r),Dᵅᵝᵢⱼ(r),Dᵝᵅᵢⱼ(r) and Dᵝᵝᵢⱼ(r)
                D[s,t,r,i,j] = Dˢᵗᵢⱼ(coord[r,:])
            `grad_D` - Tensor of shape (Nspin,Nspin,Ncoord,3,Nstate,Nstate)
                gradient of spin matrix density,
                grad_D[s,t,r,a,i,j] = ∇ₐDˢᵗᵢⱼ(coord[r,:]) with a = 0(x),1(y),2(z)
                and s,t=0(up),1(down)
            `lapl_D` - Tensor of shape (Nspin,Nspin,Ncoord,Nstate,Nstate) or None
                Laplacian of spin matrix density (if `need_laplacian==True`),
                lapl_D[s,t,r,i,j] = ∇²Dˢᵗᵢⱼ(coord[r,:])
        """
        if dm_ao is None:
            # compute density matrices in the AO basis for states (i == j)
            # and transition states (i != j) Dˢᵗᵦᵧᵢⱼ
            dm_ao = self.density_matrices_ao()

        # Evaluate atomic orbitals 𝛘ₐ(r) on the grid using pyscf
        # The orbital values and their gradients and Laplacian are returned in a single
        # array of shape (10,ncoord,nbasis).
        ao_value_all = numint.eval_ao(self.mol, coords, deriv=2)

        # Convert numpy arrays to torch Tensors
        ao_value_all = torch.from_numpy(ao_value_all).to(
            dtype=dm_ao.dtype, device=dm_ao.device)
        # value AO(r)
        ao_value = ao_value_all[0,:,:]
        # gradient d(AO)/dx, d(AO)/dy, d(AO)/dz
        grad_ao_value = ao_value_all[1:4,:,:]

        # (transition) density in AO basis
        # Dˢᵗᵢⱼ(r) = ∑ᵦ ∑ᵧ Dˢᵗᵦᵧᵢⱼ 𝛘ᵦ(r) 𝛘ᵧ(r)
        D = torch.einsum(
            'stabij,ra,rb->strij', dm_ao, ao_value, ao_value
        )
        # gradients of (transition) density
        # ∇Dˢᵗᵢⱼ(r) = ∑ᵦ ∑ᵧ Dˢᵗᵦᵧᵢⱼ ∇𝛘ᵦ(r) 𝛘ᵧ(r) + Dˢᵗᵦᵧᵢⱼ 𝛘ᵦ(r) ∇𝛘ᵧ(r))
        grad_D = (
            torch.einsum('stabij,gra,rb->strgij', dm_ao, grad_ao_value, ao_value) +
            torch.einsum('stabij,ra,grb->strgij', dm_ao, ao_value, grad_ao_value)
        )

        if need_laplacian:
            # Laplacian ∇²(AO)(r) = d^2(AO)/dx^2 + d^2(AO)/dy^2 + d^2(AO)/dz^2
            lapl_ao_value = ao_value_all[4,:,:] + ao_value_all[7,:,:] + ao_value_all[9,:,:]
            # Laplacian of (transition) density
            # ∇²Dˢᵗ(r) = ∑ᵦ ∑ᵧ Dˢᵗᵦᵧᵢⱼ [ (∇²𝛘ᵦ)(𝛘ᵧ) + 2 (∇𝛘ᵦ)·(∇𝛘ᵧ) + (𝛘ᵦ)(∇²𝛘ᵧ) ]
            lapl_D = (
                torch.einsum('stabij,ra,rb->strij',   dm_ao, lapl_ao_value, ao_value) +
              2*torch.einsum('stabij,gra,grb->strij', dm_ao, grad_ao_value, grad_ao_value) +
                torch.einsum('stabij,ra,rb->strij',   dm_ao, ao_value,      lapl_ao_value)
            )
        else:
            # Laplacian is not needed by the calling functional.
            lapl_D = None

        return D, grad_D, lapl_D

    def _check_input(self, X: Tensor, name: str = 'X', expected_size: Size = None):
        """
        Check that the input tensor X has the required compatible shape.
        If the input passes the test, it is returned. This can be used in
        the following pattern to check the size of an input variable:

        >>> X = self.check_input(X, 'X', Size([2,3]))
        """
        error_msg = f"Input '{name}' has to be of size {expected_size}, but got {X.size()}"
        if X.size() != expected_size:
            raise ValueError(error_msg)
        return X


def antisymmetric_matrix(elements: Tensor, n: int) -> Tensor:
    """
    Construct an antisymmetric n x n matrix from the unique elements
    above the diagonal.

    :param elements: unique elements of above the diagonal
    :type elements: Tensor of shape (n*(n-1)/2,)

    :param n: dimension of antisymmetric matrix, must match the length
        of `elements`
    :type n: int

    :return A: antisymmetric (n,n) matrix
    :rtype A: Tensor of shape (n,n)
    """
    if elements.size() != Size([(n*(n-1))//2]):
        raise ValueError(
            f"An antisymmetric matrix of dimensions (n,n)=({n},{n}) "
            f"has n*(n-1)/2={(n*(n-1))/2} unique elements."
        )
    # indices of upper triangle above the diagonal
    rows, cols = torch.triu_indices(n, n, offset=1, device=elements.device)
    A = torch.zeros(n,n, dtype=elements.dtype, device=elements.device)
    A[rows,cols] = elements
    A[cols,rows] = -elements

    return A


def orbital_guess(mol: pyscf.gto.Mole, guess="random", seed=None) -> numpy.ndarray:
    """
    create initial guess for molecular orbital (MO) coefficients, either using random numbers
    or the eigenvectors of the core Hamiltonian (kinetic + nuclear operator).
    If `guess` is a numpy array, it is taken as the initial MO coefficients after
    checking its shape.

    :param mol: molecule with atomic coordinates and basis set
    :type mol: pyscf.gto.Mole

    :param guess: How to generate initial guess for molecular orbitals.
    :type guess: str ('random', 'hcore', 'rohf') or numpy.ndarray with MO coefficients

    :param seed: Seed for random number generator, only seed in combination
        with guess='random'
    :type seed: int or None

    :return: MO coefficients
    :rtype: numpy.ndarray of shape (nao,nmo)
    """
    # number of basis functions
    nao = mol.nao_nr()

    if isinstance(guess, numpy.ndarray):
        # check size
        if guess.shape != (nao,nao):
            raise ValueError(
                f"Guess for initial orbital coefficients has to be a {nao,nao} matrix "
                "got {guess.shape}."
            )
        mo_coeff = guess
    elif isinstance(guess, str):
        # overlap matrix
        overlap = mol.intor_symmetric('int1e_ovlp')
        match guess:
            case 'random':
                rng = numpy.random.default_rng(seed)
                # Random, symmetric Hamiltonian
                hamiltonian = rng.random((nao, nao))
                # symmetrize H
                hamiltonian = 0.5*(hamiltonian + hamiltonian.T)
                # Diagonalize H to get orthonormal, but random MO coefficients
                _, mo_coeff = scipy.linalg.eigh(hamiltonian, overlap)
            case 'hcore':
                # core Hamiltonian (kinetic + nuclear)
                hamiltonian_core = get_hcore(mol)
                mo_energy, mo_coeff = scipy.linalg.eigh(hamiltonian_core, overlap)
            case 'rohf':
                # Take molecular orbitals from a restricted Hartree-Fock calculation.
                rohf = pyscf.scf.ROHF(mol)
                # silent
                rohf.verbose = 0
                rohf.kernel()
                mo_coeff = rohf.mo_coeff
            case _:
                raise NotImplementedError(f'Initial guess "{guess}" not implemented')
    else:
        raise ValueError(f"Initial guess must be str or 2D numpy.ndarray, got {guess}")

    return mo_coeff


def reorder_active_orbitals(mol: pyscf.gto.Mole, mo_coeff, active_orbitals: list, nelecas: int):
    """
    reorder the molecular orbitals such that the active orbitals, whose indices are specified
    in the list `active_orbitals` become the frontier orbitals.

    Let's say the active space should consist of the HOMO-2,HOMO and LUMO. Initially the MOs are
    ordered as

        [...,H-2,H-1,H,L,...]


    With `active_orbitals = [H-2,H,L]` and `nelecas = 4` the MOs will be reordered as

        [...,H-1,H-2,H,L,...]

    :param mol: molecule with the number of electrons
    :type mol: pyscf.gto.Mole

    :param mo_coeff: molecular orbitals coefficients before reordering
    :type mo_coeff: numpy.ndarray of shape (nao,nao)

    :param active_orbitals: 0-based indices of active occupied and virtual orbitals
        Instead of integers, expressions such as 'H-1', 'HOMO-1', 'L+1' or 'LUMO+1'
        can be used that will be converted to integers.
    :type active_orbitals: list of int or str

    :param nelecas: number of active electrons
    :type nelecas: int > 0

    :return mo_coeff_reordered: reordered MO coefficients
        If nclosed are the number of doubly occupied orbitals, such that nclosed+nelecas=nelec
        and ncas=len(active_orbitals) are the number of active orbitals, the frontier orbitals
        `mo_coeff[:,nclosed:nclosed+ncas]` will contain the MO coefficients of the active orbitals.
    :rtype mo_coeff_reordered: numpy.ndarray of shape (nao,nao)
    """
    # total number of electrons
    nelec = mol.tot_electrons()
    # number of doubly occupied orbitals
    nclosed = (nelec - nelecas)//2
    # number of active orbitals
    norb = len(active_orbitals)
    # Indices of HOMO and LUMO
    # For odd number of electrons, the HOMO is actually the SOMO (singly occupied MO).
    HOMO = (nelec+1)//2-1
    LUMO = HOMO+1

    # Convert expressions such as HOMO-1 or H-1 to integers
    for i,orb_index in enumerate(active_orbitals):
        if isinstance(orb_index, str):
            orb_index = eval(orb_index,
                # allow no built-in functions
                {"__builtins__": {}},
                # variables that can be used to specify orbitals
                {"H": HOMO, "HOMO": HOMO, "L": LUMO, "LUMO": LUMO}
            )
        active_orbitals[i] = orb_index

    # Indices where the active orbitals should be put.
    frontier_orbitals = numpy.arange(nclosed, nclosed+norb)

    assert len(frontier_orbitals) == len(numpy.unique(active_orbitals))

    # Reorder orbitals
    mo_coeff_reordered = numpy.copy(mo_coeff)
    for f,a in zip(frontier_orbitals, active_orbitals):
        # swap positions a and f
        mo_coeff_reordered[:,f] = mo_coeff[:,a]
        mo_coeff_reordered[:,a] = mo_coeff[:,f]

    return mo_coeff_reordered


class MultistateMatrixDensityKohnSham(MultistateMatrixDensity):
    """
    Matrix density for a single Kohn-Sham determinant
    """
    # factory methods
    @staticmethod
    def from_guess(mol: pyscf.gto.Mole, guess='hcore', seed=None):
        """
        create initial guess for molecular orbital coefficients of Kohn-Sham state,
        either using random numbers or the eigenvectors of the core Hamiltonian
        (kinetic + nuclear operator)

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole

        :param guess: How to generate initial guess for molecular orbitals.
        :type guess: str ('random', 'hcore', 'rohf') or numpy.ndarray with MO coefficients

        :param seed: Seed for random number generator, only seed in combination
            with guess='random'
        :type seed: int or None
        """
        mo_coeff = orbital_guess(mol, guess=guess, seed=seed)
        orbital_coefficients = torch.from_numpy(mo_coeff).to(dtype=torch.double)
        # Initial orbital rotation matrix is the identity U = exp(R) = Id  =>  R = 0
        _, nmo = mo_coeff.shape
        # Antisymmetric (n,n) matrix R has only n*(n-1)/2 unique elements.
        nrot = (nmo*(nmo-1))//2
        orbital_rotation_params = torch.nn.Parameter(
            data=torch.zeros(nrot).to(dtype=torch.double), requires_grad=True)

        msmd = MultistateMatrixDensityKohnSham(
            mol, orbital_coefficients, orbital_rotation_params)

        return msmd

    def __init__(
        self,
        mol: pyscf.gto.Mole,
        orbital_coefficients: Tensor,
        orbital_rotation_params: torch.nn.Parameter
    ):
        """
        Density of a doubly occupied Kohn-Sham Slater determinant.

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole

        :param orbital_coefficients: initial guess for molecular orbital
            coefficients
        :type orbital_coefficients: Tensor of shape (Nbasis,Norbital)
            This tensor must not be attached to the computational graph.

        :param orbital_rotation_params: flattened array with the elements
            of the triangle above the diagonal of the antisymmetric
            matrix R, such that U=exp(R) is an orthogonal transformation
            in the orbital space.
        :type orbital_rotation_params: Parameter tensor of shape (Norbital*(Norbital-1)/2,)
        """
        super().__init__(mol)
        # NOTE: self.orbital_rotation_params cannot be wrapped
        # as nn.Parameter since this detaches the computational graph,
        # so that gradcheck does not work anymore.

        # MO coefficients are not optimized, so they should be detached
        # from the computational graph.
        self.register_buffer("mo_coeff_guess", orbital_coefficients.detach())
        # The initial coefficients should not be optimized as they are chosen
        # to be orthonormal with respect to the overlap matrix.
        self.mo_coeff_guess.requires_grad_(False)
        # Parameters of orbital rotations
        self.nao, self.nmo = orbital_coefficients.size()
        # The triangle above the diagonal has nmo*(nmo-1)/2 elements.
        # Therefore there are nrot parameters for the orbital rotation
        nrot = (self.nmo*(self.nmo-1))//2
        self.orbital_rotation_params = self._check_input(
            orbital_rotation_params,
            name = 'orbital_rotation_params',
            expected_size=Size([nrot])
        )

    @property
    def number_of_states(self):
        """ Kohn-Sham Slater determinant describes only a single state """
        return 1

    def orbital_coefficients(self):
        """
        Molecular orbital coefficients

            C' = C exp(R)

        after applying the current orbital rotation

        :return C': MO coefficients, C'[:,i], are the coefficients
            of the i-th MO.
        :rtype: Tensor of shape (Nbasis,Norbital)
        """
        # Bring orbital rotation parameters into matrix shape.
        R = antisymmetric_matrix(self.orbital_rotation_params, self.nmo)
        # create orthogonal transformation U = exp(R)
        U = torch.matrix_exp(R)
        # rotate orbitals, C' = C.U
        mo_coeff = self.mo_coeff_guess @ U

        return mo_coeff

    def density_matrices_ao(self) -> Tensor:
        """
        construct the density matrices for the state and transition densities
        in the basis of atomic orbitals, Dˢᵗᵦᵧᵢⱼ, such that the spin matrix density
        can be expressed as

            Dˢᵗᵢⱼ(r) = ∑ᵦ ∑ᵧ Dˢᵗᵦᵧᵢⱼ 𝛘ᵦ(r) 𝛘ᵧ(r),

        where i,j = 1,...,Nstate enumerate electronic states, β,γ=1,...,Nbasis
        enumerate atomic orbitals 𝛘(r) and s,t distinguish the matrix densities
        for spin-up and spin-down and mixed spins.

        For a single determinant, Dˢᵗᵦᵧᵢⱼ is zero if s != t.

        :return dm: state (i==j) and transition (i != j) density matrices in
            in the AO basis, dm[s,t,b,g,i,j] = Dˢᵗᵦᵧᵢⱼ
        :rtype dm: Tensor of shape (2,2,Nbasis,Nbasis,Nstate,Nstate)
        """
        # rotated MO coefficients
        C = self.orbital_coefficients()
        nao, nmo = C.size()
        # spin-up and spin-down
        nspin = 2

        # Occupation numbers (0 or 1)
        occupation_numbers = torch.zeros((nspin,nspin,nmo), dtype=C.dtype, device=C.device)
        # Count spin up and spin down electrons
        nelec_up, nelec_down = self.mol.nelec
        for i_up in range(0, nelec_up):
            # aa block
            occupation_numbers[0,0, i_up] += 1
        for i_down in range(0, nelec_down):
            # bb block
            occupation_numbers[1,1, i_down] += 1

        # Build matrix density for a single Kohn-Sham state
        dm_ao = torch.einsum('ai,sti,bi->stab', C, occupation_numbers, C)
        # Add dimensions for state indices
        dm_ao = torch.reshape(dm_ao, (nspin, nspin, nao, nao, 1,1))

        return dm_ao

    def basis_transformation(self, L: Tensor):
        """
        apply a basis transformation to the matrix density

            D'(r) = L D(r) Lᵀ

        :param L: orthogonal matrix representing the basis transformation
            among the electronic states
        :type L: Tensor of shape (Nstate,Nstate)

        NOTE: The parameters of the matrix density can be modified in-place
        and detached from the computational graph.
        """
        # Kohn-Sham matrix density has only a single state (Nstate=1)
        # so there is nothing to transform
        pass

    def spin_multiplicity(self):
        """
        Spin multiplicity (2*S+1) of spin multiplets.

        Pyscf only calculates one component of a multiplet
        Sz=-S,-S+1,...,0,...,S-1,S. The other states of the multiplet have the
        same charge density and are accounted for by weighting the energy
        with the spin multiplicity.

        :return multiplicity: spin multiplicity 2*S+1
        :rtype multiplicity: int Tensor of shape (Nstate,)
        """
        # mol.spin is 2*Sz
        multiplicity = int(self.mol.spin+1)
        return torch.tensor([multiplicity])


class MultistateMatrixDensityCAS(MultistateMatrixDensity):
    """
    Matrix density for wavefunctions spanning the complete active space (CAS) generated
    by distributinng electrons over active orbitals.
    """
    # factory methods
    @staticmethod
    def from_guess(
        mol: pyscf.gto.Mole,
        norb: int,
        nelec: int,
        spin_symmetry=True,
        spin_type=SpinType.UNPOLARIZED,
        max_level=numpy.inf,
        guess='hcore',
        seed=None
    ):
        """
        create initial guess for molecular orbital (MO) coefficients and configuration interaction
        (CI) coefficients for wavefunctions built from distributing active electrons over
        active orbitals.

        The wavefunctions are expanded as linear combinations of spin-adapted configuration functions.
        The Slater determinants are constructed from the same set of orthonormal molecular orbitals.
        Only configurations with total spin <S²> = S*(S+1) are included. The desired spin is taken from
        `mol.spin`, which gives the number of unpaired electrons 2S.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param norb: number of active (spatial) orbitals
        :type norb: int > 0

        :param nelec: number of active electrons, either total number of
            electrons or tuple (neleca, nelecb) with number of up and down electrons
        :type nelec: int or (int, int)

        :param spin_symmetry: Wether to restrict calculation to spin subspace.
            True - Only spin-adapted configuration state functions with the desired spin
                ((2S = `mol.spin`) are included in the subspace.
            False - subspaced is spanned by all spin-adapted configuration state functions
                irrespective of `mol.spin`.
            NOTE: Although the Hamiltonian does not couple different spin manifolds, the molecular
            orbitals are optimized to minimize the average energy of all states in the subspace,
            therefore `spin_symmetry` does affect the final energies (cf. SA-CASSCF).
        :type spin_symmetry: bool

        :param spin_type: Determines which spin projections Sz are included in the
            active subspace. The options are:
            POLARIZED, UNPOLARIZED: Only a single Sz=(neleca-nelecb)/2 component
                is included. For example, for triplet states only one of the three components
                would be present.
            INVARIANT, INVARIANT_MIX: All possible Sz components are included. For example,
                for triplet states all three components with Sz=-1,0,+1 would be present.
        :type spin_type: SpinType

        :param max_level: maximum excitation level relative to the Hartree-Fock
            determinant (0 - only HF, 1 - singly excited, 2 - doubly excited, etc.)
            of configurations that will be included
        :type max_level: int >= 0

        :param guess: How to generate initial guess for molecular orbitals.
            `guess` can be one of the following:
                'random' - using random numbers
                'hcore'  - using eigenvectors of the core Hamiltonian
                           (kinetic + nuclear operator)
                'rohf'   - take molecular orbitals from ROHF calculation
            Alternatively, `guess` can also be a numpy.ndarray with the MO coefficients.
        :type guess: str, 'random', 'hcore' or 'rohf'

        :param seed: Seed for random number generator, only used in combination
            with guess='random'
        :type seed: int or None
        """
        # Initial guess for MO Coefficients
        mo_coeff = orbital_guess(mol, guess=guess, seed=seed)
        orbital_coefficients = torch.from_numpy(mo_coeff).to(dtype=torch.double)
        # Initial orbital rotation matrix is the identity U_mo = exp(R_mo) = Id  =>  R = 0
        _, nmo = mo_coeff.shape
        # Antisymmetric (nmo,nmo) matrix R has only nmo*(nmo-1)/2 unique elements.
        nrot_mo = (nmo*(nmo-1))//2
        orbital_rotation_params = torch.nn.Parameter(
            data=torch.zeros(nrot_mo).to(dtype=torch.double), requires_grad=True)

        # Active space defines which Slater determinants are included in subspace.
        if spin_type in [SpinType.INVARIANT, SpinType.INVARIANT_MIX]:
            # Include Slater determinants with all possible spin projections.
            # This is needed to make the subspace invariant to spatial rotations,
            # which mix states with different Sz values.
            active_space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=None)
        else:
            # Only Slater determinants with a single Sz value are included.
            # 2*Sz
            neleca, nelecb = _unpack_nelec(nelec)
            spin = neleca-nelecb
            active_space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=[spin])

        # Construct matrix of total spin operator S² in the active space.
        # It can happen that the active space is empty if (norb, nelec)
        # are incompatible.
        s2_matrix = active_space.total_spin_matrix()
        s2_eigvals, _ = numpy.linalg.eigh(s2_matrix)
        # Total number of Slater determinants
        ndet = len(s2_eigvals)
        if ndet == 0:
            raise ActiveSpaceError(
                f"Active space (norb={norb}, nelec={nelec}) is empty, "
                "check that neleca,nelecb <= norb."
            )
        if spin_symmetry:
            # `mol.spin is the number of unpaired electrons 2*S, i.e. the difference between the
            # number of up and down electrons.
            s_target = mol.spin/2
            # Count the number of spin-adapted configuration functions with the desired spin `s`.
            nstate = 0
            for s2 in s2_eigvals:
                if numpy.round(s2, decimals=2) == s_target*(s_target+1):
                    nstate += 1
            if nstate == 0:
                raise ActiveSpaceError(
                    f"There are no states with total spin <S²>=S(S+1)={s_target*(s_target+1)} "
                    f"in the active space (norb={norb}, nelec={nelec}). "
                    f"Eigenvalues of S² = {s2_eigvals}"
                )
        else:
            # Use all spin-adapted configuration functions (irrespective of spin)
            nstate = ndet

        # Start with diagonal matrix, eigenstates ~ spin-adapted configuration state functions
        state_coefficients = torch.eye(nstate, dtype=torch.double)
        # Initial state rotation matrix is the identity U_ci = exp(R_ci) = Id  =>  R = 0
        # Antisymmetric (nstate,nstate) matrix R has only nstate*(nstate-1)/2 unique elements.
        nrot_ci = (nstate*(nstate-1))//2
        state_rotation_params = torch.nn.Parameter(
            data=torch.zeros(nrot_ci).to(dtype=torch.double), requires_grad=True)

        msmd = MultistateMatrixDensityCAS(
            mol, norb, nelec,
            orbital_coefficients, orbital_rotation_params,
            state_coefficients, state_rotation_params,
            spin_symmetry=spin_symmetry,
            spin_type=spin_type,
            max_level=max_level
        )

        return msmd

    def __init__(
        self,
        mol: pyscf.gto.Mole,
        norb: int,
        nelec: int,
        orbital_coefficients: Tensor,
        orbital_rotation_params: torch.nn.Parameter,
        state_coefficients: Tensor,
        state_rotation_params: torch.nn.Parameter,
        spin_symmetry=True,
        spin_type=SpinType.UNPOLARIZED,
        max_level=numpy.inf
    ):
        """
        Matrix density for complete-active space.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param norb: number of active (spatial) orbitals
        :type norb: int > 0

        :param nelec: number of active electrons, either total number of
            electrons or tuple (neleca, nelecb) with number of up and down electrons
        :type nelec: int or (int, int)

        :param orbital_coefficients: initial guess for molecular orbital
            coefficients
        :type orbital_coefficients: Tensor of shape (Nbasis,Norbital)
            This tensor must not be attached to the computational graph.

        :param orbital_rotation_params: flattened array with the elements
            of the triangle above the diagonal of the antisymmetric
            matrix R_mo, such that U_mo=exp(R_mo) is an orthogonal transformation
            in the orbital space.
        :type orbital_rotation_params: Parameter tensor of shape (Norbital*(Norbital-1)/2,)

        :param state_coefficients: initial guess for coefficients of wavefunctions
            in the basis of spin-adapted configurations.
        :type state_coefficients: Tensor of shape (Nstate,Nstate)
            This tensor must not be attached to the computational graph.

        :param state_rotation_params: flattened array with the elements
            of the triangle above the diagonal of the antisymmetric
            matrix R_ci, such that U_ci=exp(R_ci) is an orthogonal transformation
            in the space of spin-adapted configuration functions.
        :type state_rotation_params: Parameter tensor of shape (Nstate*(Nstate-1)/2,)

        :param spin_symmetry: Wether to restrict calculation to spin subspace.
            True - Only spin-adapted configuration state functions with the desired spin
                ((2S = `mol.spin`) are included in the subspace.
            False - subspaced is spanned by all spin-adapted configuration state functions
                irrespective of `mol.spin`.
            NOTE: Although the Hamiltonian does not couple different spin manifolds, the molecular
            orbitals are optimized to minimize the average energy of all states in the subspace,
            therefore `spin_symmetry` does affect the final energies (cf. SA-CASSCF).
        :type spin_symmetry: bool

        :param spin_type: Determines which spin projections Sz are included in the
            active subspace. The options are:
            POLARIZED, UNPOLARIZED: Only a single Sz=(neleca-nelecb)/2 component
                is included. For example, for triplet states only one of the three components
                would be present.
            INVARIANT, INVARIANT_MIX: All possible Sz components are included. For example,
                for triplet states all three components with Sz=-1,0,+1 would be present.
        :type spin_type: SpinType

        :param max_level: maximum excitation level relative to the Hartree-Fock
            determinant (0 - only HF, 1 - singly excited, 2 - doubly excited, etc.)
            of configurations that will be included
        :type max_level: int >= 0
        """
        super().__init__(mol)
        # number of basis functions and molecular orbitals
        self.nao, self.nmo = orbital_coefficients.size()
        # There are two matrix densities, one for spin-up the other for spin-down.
        self.nspin = 2

        # Consistency check: number of active electrons
        neleca, nelecb = _unpack_nelec(nelec)
        if neleca+nelecb > mol.tot_electrons():
            raise ActiveSpaceError(
                f"More active electrons ({neleca+nelecb}) than total number of electrons ({mol.tot_electrons})"
            )
        # If the number of total electrons is odd (even) the number of active electrons
        # must be odd (even), too. Otherwise it is not possible to separate the orbitals
        # into those that are doubly occupied in all configurations and those whose
        # occupation varies.
        if (mol.tot_electrons() % 2 != (neleca+nelecb) % 2):
            raise ActiveSpaceError(
                "For odd (even) total number of electrons, number of active electrons must "
                f"be odd (even) as well, got {mol.tot_electrons()} electrons in total, "
                f"but {neleca+nelecb} active electrons."
            )
        # number of orbitals that are doubly occupied in all configurations.
        ndouble = (mol.tot_electrons() - (neleca+nelecb))//2
        if norb > self.nmo - ndouble:
            raise ActiveSpaceError(
                f"More active orbitals ({norb}) than molecular orbitals (nmo={self.nmo}) "
                f"minus doubly occupied orbitals (ndouble={ndouble}), (nmo-ndouble={self.nmo-ndouble})."
            )

        # Active space defines which Slater determinants are included in subspace.
        self.spin_type = spin_type
        neleca, nelecb = _unpack_nelec(nelec)
        nelec = neleca+nelecb
        if spin_type in [SpinType.INVARIANT, SpinType.INVARIANT_MIX]:
            # Include Slater determinants with all possible spin projections.
            # This is needed to make the subspace invariant to spatial rotations,
            # which mix states with different Sz values.
            self.active_space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=None)
        else:
            # Only Slater determinants with a single Sz value are included.
            # 2*Sz
            spin = neleca-nelecb
            self.active_space = ActiveSpace(norb, nelec, max_level=max_level, spin_range=[spin])

        # NOTE: self.orbital_rotation_params and self.state_rotation_params
        # cannot be wrapped as nn.Parameter since this detaches the computational graph,
        # so that gradcheck does not work anymore.

        # Eigenvectors of S² operator transform to spin-adapted basis
        s2_matrix = self.active_space.total_spin_matrix()
        s2_eigvals, s2_eigvecs = numpy.linalg.eigh(s2_matrix)
        # Total number of Slater determinants
        self.ndet = len(s2_eigvals)

        self.spin_symmetry = spin_symmetry
        if spin_symmetry:
            # Spin of target states, `mol.spin is the number of unpaired electrons 2*S.
            s_target = mol.spin/2
            # indices of selected spin states
            selected = []
            # Select eigenvalues matching the desired spin.
            for i_spin, s2 in enumerate(s2_eigvals):
                if numpy.round(s2, decimals=2) == s_target*(s_target+1):
                    selected.append(i_spin)
            nstate = len(selected)
            selected = numpy.array(selected)
            if nstate == 0:
                raise ActiveSpaceError(
                    f"There are no states with total spin <S²>=S(S+1)={s_target*(s_target+1)} "
                    f"in the active space (norb={norb}, nelec={nelec})."
                )
        else:
            # Use all spin states
            selected = numpy.arange(0, self.ndet)
            nstate = self.ndet

        # Store matrix elements of S² in the basis of Slater determinants
        self.register_buffer("s2_matrix", torch.from_numpy(s2_matrix).to(dtype=torch.double))

        # Store transformation from Slater determinants to spin-adapted CSF.
        self.register_buffer("det_to_csf",
            torch.from_numpy(s2_eigvecs[:,selected]).to(dtype=torch.double)
        )

        # Number of states with desired spin
        self.nstate = nstate

        # The initial guess for MO coefficients is not optimized, so it should be detached
        # from the computational graph.
        self.register_buffer("mo_coeff_guess", orbital_coefficients.detach())
        # The initial coefficients should not be optimized as they are chosen
        # to be orthonormal with respect to the overlap matrix.
        self.mo_coeff_guess.requires_grad_(False)

        # Parameters of orbital rotations
        # The triangle above the diagonal has nmo*(nmo-1)/2 elements.
        # Therefore there are nrot parameters for the orbital rotation.
        nrot_mo = (self.nmo*(self.nmo-1))//2
        self.orbital_rotation_params = self._check_input(
            orbital_rotation_params,
            name = 'orbital_rotation_params',
            expected_size=Size([nrot_mo])
        )

        # Initial guess for state coefficients is not optimized, so it should be detached
        # from the computational graph.
        self.register_buffer("ci_coeff_guess", state_coefficients.detach())
        # The initial coefficients should not be optimized as they are chosen
        # to be orthonormal.
        self.ci_coeff_guess.requires_grad_(False)

        # Parameters of state rotations
        # The triangle above the diagonal has nstate*(nstate-1)/2 elements.
        # Therefore there are nrot_ci parameters for the state rotation.
        nrot_ci = (self.nstate*(self.nstate-1))//2
        self.state_rotation_params = self._check_input(
            state_rotation_params,
            name = 'state_rotation_params',
            expected_size=Size([nrot_ci])
        )

        # The density matrices in the AO basis depend on
        #  - the MO coefficients (optimized via rotation matrix U_mo)
        #  - the CI coefficients (optimized via rotation matrix U_ci)
        #  - the transformation from Slater determinants to spin-adapted CSFs  (constant)

        # (Spin) density matrices in basis of active MOs (p,q) and Slater determinants (I,J)
        #   D_mo[s,t,q,p,I,J] = <I|pₛ^+ qₜ|J>
        dm_active_mo = self.active_space.matrix_density_mo()
        # Add diagonal part of density matrices for doubly occupied orbitals
        dm_mo = numpy.zeros((self.nspin, self.nspin, self.nmo, self.nmo, self.ndet, self.ndet))
        # loop over doubly occupied orbitals
        for imo in range(0, ndouble):
            # loop over Slater determinants
            for idet in range(0, self.ndet):
                for ispin in range(0, self.nspin):
                    dm_mo[ispin,ispin,imo,imo,idet,idet] = 1.0
        # loop over active orbitals
        for imo in range(0, norb):
            for jmo in range(0, norb):
                dm_mo[:,:,ndouble+imo,ndouble+jmo,:,:] = dm_active_mo[:,:,imo,jmo,:,:]
        # virtual orbitals - nothing to do

        # Transform with selected eigenvectors of S²
        #   D_mo_spin[s,t,q,p,S,T] = <S|pₛ^+ qₜ|T> where S and T are eigenfunctions of S².
        dm_mo_spin = numpy.einsum(
            'stpqIJ,IS,JT->stpqST',
            dm_mo,
            s2_eigvecs[:,selected], s2_eigvecs[:,selected],
            optimize='greedy'
        )

        # D_mo_spin is constant. Here '_spin' refers to the total spin S², not the individual
        # electronic spins.
        self.register_buffer("dm_mo_spin", torch.from_numpy(dm_mo_spin).to(dtype=torch.double))
        self.dm_mo_spin.requires_grad_(False)

    @property
    def number_of_states(self):
        """ Number of states in active space with desired spin """
        return self.nstate

    @property
    def number_of_determinants(self):
        """ Number of Slater determinants in active space """
        return self.ndet

    def orbital_coefficients(self):
        """
        Molecular orbital coefficients

            C_mo' = C_mo(guess) exp(R_mo)

        after applying the current orbital rotation

        :return C_mo': MO coefficients, C_mo'[:,i], are the coefficients
            of the i-th MO.
        :rtype: Tensor of shape (Nbasis,Norbital)
        """
        # Bring orbital rotation parameters into matrix shape.
        R_mo = antisymmetric_matrix(self.orbital_rotation_params, self.nmo)
        # create orthogonal transformation U = exp(R)
        U_mo = torch.matrix_exp(R_mo)
        # rotate orbitals, C_mo' = C_mo.U_mo
        mo_coeff = self.mo_coeff_guess @ U_mo

        return mo_coeff

    def state_coefficients(self):
        """
        State coefficients in the basis of spin-adapted configurations

            C_ci' = C_ci(guess) exp(R_ci)

        after applying the current state rotation

        :return C_ci': CI coefficients, C_ci'[:,I], are the coefficients
            of the I-th wavefunction in the basis of spin-adapted configuration functions.
        :rtype: Tensor of shape (Nstate,Nstate)
        """
        # Bring state rotation parameters into matrix shape.
        R_ci = antisymmetric_matrix(self.state_rotation_params, self.nstate)
        # create orthogonal transformation U_ci = exp(R_ci)
        U_ci = torch.matrix_exp(R_ci)
        # rotate orbitals, C_ci' = C_ci.U_ci
        ci_coeff = self.ci_coeff_guess @ U_ci

        return ci_coeff

    def determinant_coefficients(self):
        """
        Coefficients of wavefunctions in the basis of Slater determinant

        :return C_det: CI coefficients, C_det[:,I] are the coefficients
            of the I-th wavefunction in the basis of Slater determinants
        :rtype: Tensor of shape (Ndet, Nstate) with Ndet >= Nstate
        """
        # in basis of spin-adapted configuration functions
        ci_coeff_csf = self.state_coefficients()
        # transform to basis of Slater determinants
        ci_coeff_det = torch.einsum('si,ds->di', ci_coeff_csf, self.det_to_csf)
        return ci_coeff_det

    def occupation_labels(self):
        """
        Labels for all Slater determinants in active space (nelec/norb).

        The occupation string shows which orbitals are doubly
        occupied (2), singly occupied (a or b) or empty (.)
        e.g.: '222ab...' for a HOMO-LUMO excited determinant.

        :return occupation_strings: labels for all Slater determinants
        :rtype occupation_strings: list of str
        """
        return self.active_space.occupation_labels()

    def spin_s2_expectation(self):
        """
        Expectation value of the total spin operator S² of states

        :return s2: expectation value,
            s2[I] = <I|S²|I> = S*(S+1)
        :rtype s2: Tensor of shape (Nstate,)
        """
        ci_coeff_det = self.determinant_coefficients()
        s2 = torch.einsum('st,si,ti->i', self.s2_matrix, ci_coeff_det, ci_coeff_det)

        return s2

    def spin_multiplicity(self):
        """
        Spin multiplicity (2*S+1) of spin multiplets.

        :return multiplicity: spin multiplicity 2*S+1
        :rtype multiplicity: int Tensor of shape (Nstate,)
        """
        # expectation value of spin <S²>
        s2 = self.spin_s2_expectation()
        # We can find the spin S by solving the quadratic equation
        # <S²> = S*(S+1) for S, which gives
        # S = 1/2 (sqrt(1 + 4*<S²>) - 1)
        # The spin multiplicity is
        # 2*S+1 = sqrt(1 + 4*<S²>)
        multiplicity = torch.sqrt(1.0 + 4*s2)
        multiplicity = torch.round(multiplicity, decimals=0).int()
        return multiplicity

    def state_weights(self):
        """
        Multiplicities (due to spin degeneracy) that are used to weight the
        state energies when calculating the subspace energy.

        :return weights: weights of state in subspace energy depend on spin type,
            POLARIZED, UNPOLARIZED: spin multiplicity 2*S+1
            INVARIANT, INVARIANT_MIX: 1
        :rtype weights: int Tensor of shape (Nstate,)
        """
        sz = self.active_space.spin_projection_sz()
        if len(numpy.unique(sz)) == 1:
            # The subspace contains only a single Sz-component of the spin multiplets.
            # For instance for a triplet state, only the Sz=+1 state is present.
            # The energies of the states have to be weights by the factor (2*S+1).
            # The other states of the multiplet have the same charge density and are
            # accounted for by weighting the energy with the spin multiplicity.
            weights = self.spin_multiplicity()
        else:
            # All components of a multiplet are contained explicitly in the
            # subspace. There is no need to weight the states by their
            # spin multiplicity (weights=1).
            weights = torch.ones(self.nstate)
        return weights

    def density_matrices_ao(self) -> Tensor:
        """
        construct the (spin) density matrices for the state and transition densities
        in the basis of atomic orbitals, Dˢᵗᵦᵧᵢⱼ, such that the matrix density
        can be expressed as

            Dˢᵗᵢⱼ(r) = ∑ᵦ ∑ᵧ Dˢᵗᵦᵧᵢⱼ 𝛘ᵦ(r) 𝛘ᵧ(r),

        where i,j = 1,...,Nstate enumerate electronic states, β,γ=1,...,Nbasis
        enumerate atomic orbitals 𝛘(r) and s,t distinguishes the matrix densities
        for spin-up and spin-down and mixed spins.

        :return dm: state (i==j) and transition (i != j) density matrices in
            in the AO basis, dm[s,t,b,g,i,j] = Dˢᵗᵦᵧᵢⱼ
        :rtype dm: Tensor of shape (2,2,Nbasis,Nbasis,Nstate,Nstate)
        """
        # rotated MO coefficients
        mo_coeff = self.orbital_coefficients()
        # rotated CI coefficients
        ci_coeff = self.state_coefficients()

        # Transform with CI coefficients
        # s,t: spin-adapted configurations, i,j: CI states
        dm_mo_ci = torch.einsum('...st,si,tj->...ij',
            self.dm_mo_spin, ci_coeff, ci_coeff)

        # Transform with MO coefficients
        # s,t: spin up or down, m,n: MO indices, a,b: AO indices, i,j: CI states
        dm_ao = torch.einsum('stmnij,am,bn->stabij',
            dm_mo_ci, mo_coeff, mo_coeff)

        return dm_ao

    def basis_transformation(self, L: Tensor):
        """
        apply a basis transformation to the matrix density

            D'(r) = L D(r) Lᵀ

        :param L: orthogonal matrix representing the basis transformation
            among the electronic states
        :type L: Tensor of shape (Nstate,Nstate)

        NOTE: The parameters of the matrix density can be modified in-place
        and detached from the computational graph.
        """
        # Current state coefficients C_ci
        ci_coeff = self.state_coefficients()
        # rotate CI coefficients C_ci -> C_ci Lᵀ, set them as new guess ...
        self.register_buffer("ci_coeff_guess", torch.einsum('ij,kj->ik', ci_coeff.detach(), L))
        # ... and reset state rotation to the identity, exp(0) = Id
        nrot_ci = (self.nstate*(self.nstate-1))//2
        # R_ci = 0
        self.state_rotation_params = torch.nn.Parameter(
            data=torch.zeros(nrot_ci).to(dtype=torch.double, device=ci_coeff.device),
            requires_grad=True
        )

    def __repr__(self):
        text = self.__class__.__name__ + f"({self.active_space})"
        return text

    def __str__(self):
        """
        Show
            - molecular geometry
            - basis set
            - CI vectors in basis of Slater determiants
        """
        # <I|S²|I>
        spin_s2 = self.spin_s2_expectation().detach().cpu().numpy()
        ci_coeff_det = self.determinant_coefficients().detach().cpu().numpy()
        occ_labels = self.occupation_labels()

        text = ""
        # molecular geometry
        text += "== Molecular Geometry\n"
        text += f"charge= {self.mol.charge} spin= {self.mol.spin}\n"
        text += self.mol.tostring(format='raw')
        text += "\n\n"
        # active space
        text += f"== {self.active_space}/{self.mol.basis}\n"
        for istate in range(0, self.number_of_states):
            text += f"state= {istate}  spin <S²>= {numpy.round(spin_s2[istate], decimals=2)}\n"
            for idet in range(0, self.number_of_determinants):
                coeff = ci_coeff_det[idet,istate]
                text += f"    {coeff:+7.4f}  {occ_labels[idet]}\n"

        return text

    def _zero_transition_densities(self):
        """
        Set the off-diagonal elements of the matrix density D(r) to zero,
        i.e. Dᵢⱼ(r) = 0 for i ≠ j.

        This underscore method only exists for debugging purposes.
        """
        identity = torch.eye(
            self.number_of_states, dtype=self.dm_mo_spin.dtype, device=self.dm_mo_spin.device)
        # replace Dᵢⱼ(r) by Dᵢᵢ(r) δᵢⱼ
        self.register_buffer("dm_mo_spin", self.dm_mo_spin * identity)
        self.dm_mo_spin.requires_grad_(False)
