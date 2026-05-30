# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy

import pyscf.dft
import pyscf.gto

from torch import Tensor
import torch.linalg
import torch.nn
import torch.optim

from mlmsdft.dft.density import MultistateMatrixDensity
from mlmsdft.dft.hartree import HartreeFunctionalAO
from mlmsdft.dft.kinetic import KineticFunctionalAO
from mlmsdft.dft.nuclear import NuclearFunctionalAO
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import concat_spin_blocks, spin_trace
from mlmsdft.dft.xc import lda_x_dirac, lda_c_chachiyo
# Replace scalar functions by matrix functionals.
import mlmsdft.nn.functional as MF
# wrapper around optimizer written in numpy
from mlmsdft.optim.torch_optimizer import WrappedOptimizer

class Hamiltonian(ABC):
    """
    Abstract base class
    """
    @abstractmethod
    def matrix_elements(self, msmd: MultistateMatrixDensity) -> Tensor:
        """
        Implement the Hamiltonian functional H[D(r)] that maps the
        matrix density to the Hamiltonian in the subspace of electronic states.
        """
        pass


class HamiltonianSemilocal(Hamiltonian):
    def __init__(
            self,
            mol: pyscf.gto,
            kinetic_functional: Callable = None,
            exchange_functional: Callable = lda_x_dirac,
            correlation_functional: Callable = lda_c_chachiyo,
            exact_exchange_functional: Callable = None,
            spin_type: SpinType = SpinType.UNPOLARIZED,
            grid_level: int = 3,
            grid_chunks: int = 1
        ):
        """
        Hamiltonian with semilocal kinetic, exchange and correlation functionals.

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole

        :param kinetic_functional, exchange_functional, correlation_functional:
            Semilocal kinetic, exchange and correlation functionals.
            If kinetic_functional is None, the kinetic energy is calculated from
            the wavefunction.
        :type kinetic_functional, exchange_functional, correlation_functional:
            Callables taking three tensors with the matrix density D(r), its gradient ∇D(r)
            and possibly also the Laplacian ∇²D as arguments

        :param exact_exchange_functional: Functional for exact (Hartree-Fock) exchange.
            The exact exchange is a non-local functional that depends on the density field
            Dᵢⱼ(r,r') = ∑ᵣ ∑ₛ Dᵣₛ,ᵢⱼ 𝛘ᵣ(r) 𝛘ₛ(r') (D=Dᵅᵅ+Dᵝᵝ,Dᵅᵅ,Dᵝᵝ or supermatrix)
        :type exact_exchange_functional: Callable taking the matrix density in the
            atomic orbitals (AO) basis, Dᵣₛ,ᵢⱼ, where r and s are AO indices and
            i,j are electronic states. If None, no exact exchange is added.

        :param spin_type: Determines how the electronic spin degrees of the matrix
            density are treated when constructing the Hamiltonian. The options are:
            UNPOLARIZED: The Hamiltonian is constructed from the spin-traced matrix density,
                Dᵢⱼ=Dᵅᵢⱼ(r)+Dᵝᵢⱼ(r). The (exact) exchange is calculated as 2*X[D/2]ᵢⱼ.
            POLARIZED: The (exact) exchange part of the Hamiltonian is computed separately for the
                spin-up and spin-down matrix densities, Xᵢⱼ = X[Dᵅ]ᵢⱼ + X[Dᵝ]ᵢⱼ.
            INVARIANT:
                The (exact) exchange part of the Hamiltonian is computed from the (2N)x(2N)
                supermatrix containing the NxN same-spin blocks Dᵅᵅ and Dᵝᵝ on the diagonal and
                the mixed-spin blocks Dᵅᵝ and Dᵝᵅ on the off-diagonal:
                Xᵢⱼ=spin_trace(X[(Dᵅᵅ Dᵅᵝ \\ Dᵝᵅ Dᵝ)])ᵢⱼ
                The resulting exchange energy matrix is invariant under rotations of the
                electronic spins, provided that all components of a spin multiplet (Sz=-S,...,S)
                are included in the subspace.
            INVARIANT_MIX:
                The (exact) exchange part of the Hamiltonian is the average of 50% of the
                unpolarized and 50% of the invariant exchange parts,
                Xᵢⱼ=1/2 { 2 X[D/2]ᵢⱼ +  spin_trace(X[(Dᵅᵅ Dᵅᵝ \\ Dᵝᵅ Dᵝ)])ᵢⱼ }
        :type spin_type: SpinType

        :param grid_level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type grid_level: int

        :param grid_chunks: If there is not enough memory (on the CPU or GPU)
            to hold all arrays, the grid is divided into `grid_chunks` parts,
            the functionals are evaluated on the smaller chunks of the
            grid and are summed into the Hamiltonian matrix at the end.
        """
        self.grid_level = grid_level
        # Coordinate grid is divided in `grid_chunks` chunks to reduce memory footprint.
        assert isinstance(grid_chunks, int) and grid_chunks >= 1, "`grid_chunks` should be an integer >= 1"
        self.grid_chunks = grid_chunks

        self.mol = mol
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = grid_level
        self.grids.build()
        # The external potential energy and the Hartree-part
        # of the electron-electron repulsion are calculated
        # using the AO representation of the matrix density.
        self.nuclear_ao = NuclearFunctionalAO(mol)
        self.hartree_ao = HartreeFunctionalAO(mol)
        # If no orbital-free kinetic functional is provided (kinetic_functional = None)
        # the kinetic energy is computed from the wavefunctions
        self.kinetic_ao = KineticFunctionalAO(mol)
        # Kinetic, correlation and exchange energies are
        # computed on a real-space grid.
        self.kinetic = kinetic_functional
        self.exchange = exchange_functional
        self.correlation = correlation_functional
        # Exact exchange (EXX)
        self.exact_exchange = exact_exchange_functional
        # Should the exchange energy be calculated from
        # (i) the total matrix density Dᵢⱼ=Dᵅᵢⱼ+Dᵝᵢⱼ as
        #   Xᵢⱼ = 2 * X[D/2]ᵢⱼ  if spin_type == UNPOLARIZED
        # or
        # (ii) for each spin type separately,
        #   Xᵢⱼ = X[Dᵅ]ᵢⱼ + X[Dᵝ]ᵢⱼ  if spin_type == POLARIZED
        # or
        # (iii) from the supermatrix
        #                       (Dᵅᵅᵢⱼ Dᵅᵝᵢⱼ)
        #   Xᵢⱼ = spin_trace{ X[(           )] }
        #                       (Dᵝᵅᵢⱼ  Dᵝᵝᵢⱼ)
        # if spin_type == INVARIANT
        # or
        # (iv) as an equal mix of the unpolarized and invariant exchange energies
        #   Xᵢⱼ = 1/2 * (X^{unpol} + X^{inv})
        #                                              (Dᵅᵅ Dᵅᵝ)
        #       = 1/2 * ( 2 * X[D/2]ᵢⱼ + spin_trace{ X[(       )] }ᵢⱼ )
        #                                              (Dᵝᵅ  Dᵝᵝ)
        # if spin_type == INVARIANT_MIX
        # ?
        if spin_type not in SpinType:
            raise ValueError(f"`spin_type` must be instance of SpinType, got {spin_type}")
        self.spin_type = spin_type
        # Does any of the functionals need the Laplacian of the density?
        self.need_laplacian = (
            getattr(self.kinetic, "need_laplacian", False) or
            getattr(self.exchange, "need_laplacian", False) or
            getattr(self.correlation, "need_laplacian", False)
        )

    def __call__(self, msmd: MultistateMatrixDensity) -> Tensor:
        return self.matrix_elements(msmd)

    def matrix_elements(self, msmd: MultistateMatrixDensity) -> Tensor:
        """
        Compute the Hamiltonian matrix H[D(r)]ᵢⱼ in the basis of electronic states at a
        given matrix density D(r).

        :param msmd: multistate matrix density
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return hamiltonian: Hamiltonian matrix Hᵢⱼ
        :rtype hamiltonian: Tensor of shape (Nstate,Nstate)
        """
        # Check that the same geometry and basis is used for defining
        # the Hamiltonian and the matrix density.
        assert id(self.mol) == id(msmd.mol)
        # Use AO representation to compute external potential,
        # Hartree matrices, exact exchange matrices and kinetic energy
        # if there is not kinetic density functional.
        spin_dm = msmd.density_matrices_ao()
        # sum over spins, Dᵦᵧᵢⱼ = Dᵅᵅᵦᵧᵢⱼ + Dᵝᵝᵦᵧᵢⱼ
        dm = torch.einsum('ss...->...', spin_dm)
        # electron-nuclei attraction
        Ven = self.nuclear_ao(dm)
        # Hartree-part of electron-electron repulsion
        J = self.hartree_ao(dm)

        # nuclear repulsion energy between ions is the same
        # for all electronic states
        Vnn = self.mol.get_enuc() * torch.eye(msmd.number_of_states).to(
            dtype=Ven.dtype, device=Ven.device)

        # Hamiltonian matrix (without T, X and C)
        H = Vnn + Ven + J

        if self.kinetic is None:
            # compute T from atomic orbital representation of matrix density
            T = self.kinetic_ao(dm)
            # Add kinetic energy to Hamiltonian, if the kinetic energy is calculated
            # with an orbital-free functional it is added further down.
            H = H + T

        # Exact exchange is calculated from density matrices in AO basis.
        if self.exact_exchange is not None:
            match self.spin_type:
                case SpinType.UNPOLARIZED:
                    # Compute exact exchange matrix from total charge density.
                    # Assuming that the spin-up and spin-down densities are the same,
                    # Dᵅᵅ(r,r') = Dᵝᵝ(r,r') = D(r,r')/2,
                    # the exact exchange is calculated as
                    # K[Dᵅᵅ(r,r')]+K[Dᵝᵝ(r,r')] = 2*K[D(r,r')/2]ᵢⱼ.
                    K = 2.0*self.exact_exchange(dm/2.0)
                case SpinType.POLARIZED:
                    # Densities for spin-up and spin-down contribute separately to the
                    # exchange.
                    #   K = ax * K[Dᵅᵅ(r,r')]ᵢⱼ + ax * K[Dᵝᵝ(r,r')]ᵢⱼ
                    #     = ax * (
                    #       -1/2 ∑ₖ ∫∫' Dᵅᵅᵢₖ(r,r') Dᵅᵅₖⱼ(r',r)/|r-r'| +
                    #       -1/2 ∑ₖ ∫∫' Dᵝᵝᵢₖ(r,r') Dᵝᵝₖⱼ(r',r)/|r-r'| )
                    K = (
                        # K[Dᵅᵅ]ᵢⱼ
                        self.exact_exchange(spin_dm[0,0,...]) +
                        # K[Dᵝᵝ]ᵢⱼ
                        self.exact_exchange(spin_dm[1,1,...])
                    )
                case SpinType.INVARIANT:
                    # Stack spin blocks into a (2N)*(2N) supermatrix:
                    # (Dᵅᵅ Dᵅᵝ)
                    # (Dᵝᵅ  Dᵝᵝ)
                    super_spin_dm = concat_spin_blocks(spin_dm)
                    # The exact exchange matrix functional operates on the supermatrix
                    # to get a (2N)*(2N) exchange matrix
                    # (Exxᵅᵅ Exxᵅᵝ)          (Dᵅᵅ Dᵅᵝ)
                    # (           ) = Exx { (       ) }
                    # (Exxᵝᵅ  Exxᵝᵝ)          (Dᵝᵅ  Dᵝᵝ)
                    # By spin-tracing out the off-diagonal block we obtain the N*N exchange
                    # matrix summed over spins:
                    #                                  (Dᵅᵅ Dᵅᵝ)
                    # Exxᵅᵅ + Exxᵝᵝ = spin_trace( Exx { (       ) } )
                    #                                  (Dᵝᵅ  Dᵝᵝ)
                    K = spin_trace(self.exact_exchange(super_spin_dm))
                case SpinType.INVARIANT_MIX:
                    # Exact exchange from spin matrix density.
                    super_spin_dm = concat_spin_blocks(spin_dm)
                    K_inv = spin_trace(self.exact_exchange(super_spin_dm))
                    # Exact exchange from total charge matrix density
                    K_unpol = 2.0*self.exact_exchange(dm/2.0)
                    # Average of unpolarized and invariant treatment.
                    K = 0.5 * (K_unpol + K_inv)
                case _:
                    raise ValueError(f"`spin_type` must be instance of SpinType, got {self.spin_type}")
            H = H + K

        # Loop over chunks of grid points and associated integration weights.
        for coords, weights in zip(
                numpy.array_split(self.grids.coords, self.grid_chunks),
                numpy.array_split(self.grids.weights, self.grid_chunks)):
            # evaluate semilocal kinetic, exchange and correlation functionals
            # Inputs are D(r) and ∇D(r) on the integration grid.
            # NOTE: ∇²D is only calculated if it is needed by any of the functionals.
            spin_D, grad_spin_D, lapl_spin_D = msmd.evaluate(
                coords, dm_ao=spin_dm, need_laplacian=self.need_laplacian)

            # Sum over spin
            D = torch.einsum('ss...->...', spin_D)
            grad_D = torch.einsum('ss...->...', grad_spin_D)
            # If ∇²D is not needed by any functional, it is set to None.
            if lapl_spin_D is not None:
                lapl_D = torch.einsum('ss...->...', lapl_spin_D)
            else:
                lapl_D = None

            # kinetic (t), exchange (x) and correlation (c) energy densities
            if self.kinetic is not None:
                t = self.kinetic(D, grad_D, lapl_D)

            if self.exchange is not None:
                match self.spin_type:
                    case SpinType.UNPOLARIZED:
                        # The spin-restricted/unpolarized version of the exchange functional assumes
                        # that the spin-up and spin-down parts of the matrix density are the same,
                        # D(r) = Dᵅ(r)+Dᵝ(r) = 2 Dᵅ(r)
                        # Therefore, one can reconstruct the spin matrix density from the
                        # charge matrix density,
                        #   Dᵅ(r) = D(r)/2
                        # and similarly for the gradient
                        #   ∇Dᵅ(r) = ∇D(r)/2
                        # and the Laplacian
                        #   ∇²Dᵅ(r) = ∇²D(r)/2
                        # Since, the spin matrix densities for up and down are the same,
                        # the exchange energy is only calculated once and then multiplied by two.
                        # xed[Dᵅ(r),Dᵝ(r)] = xed[Dᵅ(r)] + xed[Dᵝ(r)] = 2 xed[Dᵅ(r)]
                        if lapl_D is None:
                            # Functional does not depend on ∇²D(r)/2
                            x = 2.0 * self.exchange(D/2.0, grad_D/2.0, None)
                        else:
                            x = 2.0 * self.exchange(D/2.0, grad_D/2.0, lapl_D/2.0)
                    case SpinType.POLARIZED:
                        # Compute the exchange energy for spin-up and spin-down electrons
                        # separately and add it,
                        # xed[Dᵅᵅ(r),Dᵝᵝ(r)] = xed[Dᵅᵅ(r)] + xed[Dᵝᵝ(r)]
                        if lapl_spin_D is None:
                            # Functional does not depend on ∇²Dᵅᵅ(r) or ∇²Dᵝᵝ(r)
                            x = (
                                # xed(        Dᵅᵅ(r),          ∇Dᵅᵅ(r)   ) +
                                self.exchange(spin_D[0,0,...], grad_spin_D[0,0,...], None) +
                                # xed(        Dᵝᵝ(r),           ∇Dᵝᵝ(r)    )
                                self.exchange(spin_D[1,1,...], grad_spin_D[1,1,...], None)
                            )
                        else:
                            x = (
                                # xed(        Dᵅᵅ(r),          ∇Dᵅᵅ(r),              ∇²Dᵅᵅ(r)  ) +
                                self.exchange(spin_D[0,0,...], grad_spin_D[0,0,...], lapl_spin_D[0,0,...]) +
                                # xed(        Dᵝᵝ(r),           ∇Dᵝᵝ(r),               ∇²Dᵝᵝ(r)  )
                                self.exchange(spin_D[1,1,...], grad_spin_D[1,1,...], lapl_spin_D[1,1,...])
                            )
                    case SpinType.INVARIANT:
                        # Stack spin blocks into a (2N)*(2N) supermatrix:
                        # (Dᵅᵅ Dᵅᵝ)
                        # (Dᵝᵅ Dᵝᵝ)
                        super_spin_D = concat_spin_blocks(spin_D)
                        # similarly for gradient
                        # (∇Dᵅᵅ ∇Dᵅᵝ)
                        # (∇Dᵝᵅ ∇Dᵝᵝ)
                        super_grad_spin_D = concat_spin_blocks(grad_spin_D)
                        # and for Laplacian
                        # (∇²Dᵅᵅ ∇²Dᵅᵝ)
                        # (∇²Dᵝᵅ ∇²Dᵝᵝ)  if available
                        if lapl_spin_D is None:
                            super_lapl_spin_D = None
                        else:
                            super_lapl_spin_D = concat_spin_blocks(lapl_spin_D)
                        # The exchange energy density matrix functional operates on the supermatrix
                        # to get a (2N)*(2N) matrix with the exchange energy density
                        # (xedᵅᵅ(r) xedᵅᵝ(r))         (Dᵅᵅ(r) Dᵅᵝ(r))
                        # (                 ) = xed[ (             ) ]
                        # (xedᵝᵅ(r)  xedᵝᵝ(r))         (Dᵝᵅ(r)  Dᵝᵝ(r))
                        # By spin-tracing out the off-diagonal block we obtain the N*N matrix for
                        # the exchange energy density:
                        #                                       (Dᵅᵅ(r) Dᵅᵝ(r))
                        # xedᵅᵅ(r) + xedᵝᵝ(r) = spin_trace( xed[ (             ) ] )
                        #                                       (Dᵝᵅ(r)  Dᵝᵝ(r))
                        x = spin_trace(
                            self.exchange(super_spin_D, super_grad_spin_D, super_lapl_spin_D)
                        )
                    case SpinType.INVARIANT_MIX:
                        # Stack spin blocks into a (2N)*(2N) supermatrix:
                        super_spin_D = concat_spin_blocks(spin_D)
                        # similarly for gradient
                        super_grad_spin_D = concat_spin_blocks(grad_spin_D)
                        # and for Laplacian if available
                        if lapl_spin_D is None:
                            super_lapl_spin_D = None
                        else:
                            super_lapl_spin_D = concat_spin_blocks(lapl_spin_D)
                        # Exchange from spin matrix density
                        x_inv = spin_trace(
                            self.exchange(super_spin_D, super_grad_spin_D, super_lapl_spin_D)
                        )
                        # Exchange from charge matrix density
                        if lapl_D is None:
                            # Functional does not depend on ∇²D(r)/2
                            x_unpol = 2.0 * self.exchange(D/2.0, grad_D/2.0, None)
                        else:
                            x_unpol = 2.0 * self.exchange(D/2.0, grad_D/2.0, lapl_D/2.0)
                        # Average of unpolarized and invariant treatment.
                        x = 0.5 * (x_unpol + x_inv)
                    case _:
                        raise ValueError(f"`spin_type` must be instance of SpinType, got {self.spin_type}")
            else:
                x = torch.zeros_like(D, dtype=D.dtype, device=D.device)

            if self.correlation is not None:
                # Some correlation functions operate on the total charge density, while
                # others expect separate matrix densities for spin-up and spin-down.
                if getattr(self.correlation, "need_spin_density", False):
                    # Spin-polarized correlation functionals mix the spin-up and spin-down
                    # densities in a complicated way that breaks the rotational invariance.
                    c = self.correlation(
                        # Only keep the same-spin blocks
                        # Dᵅᵅ(r) and Dᵝᵝ(r)
                        torch.einsum('ss...->s...', spin_D),
                        # ∇Dᵅᵅ and ∇Dᵝᵝ
                        torch.einsum('ss...->s...', grad_spin_D),
                        # ∇²Dᵅᵅ and ∇²Dᵝᵝ is available
                        None if lapl_spin_D is None else torch.einsum('ss...->s...', lapl_spin_D),
                        spin_polarized=(self.spin_type == SpinType.POLARIZED)
                    )
                else:
                    c = self.correlation(D, grad_D, lapl_D)
            else:
                c = torch.zeros_like(D, dtype=D.dtype, device=D.device)

            # integrate energy densities
            weights = torch.from_numpy(weights).to(dtype=D.dtype, device=D.device)
            # reshape weights so that they can be multiplied with matrices
            # using the broadcasting rules
            #   weights -> (...,1,1)
            #   t          (...,n,n)
            weights = weights.unsqueeze(-1).unsqueeze(-1)
            X = torch.sum(x * weights, 0)
            C = torch.sum(c * weights, 0)

            # Add exchange and correlation energy to Hamiltonian
            H = H + X + C

            if self.kinetic is not None:
                T = torch.sum(t * weights, 0)
                # Add kinetic energy from orbital-free functional
                H = H + T

        # Check for NaN's
        if H.isnan().any():
            print("Hamiltonian")
            print(H)
            raise RuntimeError(
                "Hamiltonian contains NaN's, there is probably a bug in any of the functionals."
            )

        return H


def minimize_subspace_energy(
    hamiltonian: Hamiltonian,
    msmd: MultistateMatrixDensity,
    state_average=None,
    **optimizer_kwds
):
    """
    Find the matrix density of the lowest few electronic states by
    minimizing the staged-averaged energy.

    :param state_average: number of states M to average over
        If `state_average` is None, the average extends over all states (M=N)
    :type state_average: int <= N or None
    """
    optimizer = WrappedOptimizer(msmd.parameters(), **optimizer_kwds)
    # Enable all parameters for optimization.
    for param in msmd.parameters():
        param.requires_grad = True

    # objective function
    def closure():
        optimizer.zero_grad()
        # compute objective function, that is the subspace
        # energy to minimize.
        H = hamiltonian(msmd)
        # Average energy of the lowest M electronic states to get the subspace energy
        #
        #  E = 1/M ∑ₐ λₐ   a=1,...,M   λ(1) <= λ(2) <= ...<= λ(M) < ... <= λ(N)
        #
        # where λₐ are the eigenvalues of the Hamiltonian H[D(r)].
        #
        # If M=`state_average` is not specified (None), all states are included in the average,
        #
        #   E = 1/N tr(H) = 1/N ∑ᵢ H[D(r)]ᵢᵢ
        #
        # NOTE: In POLARIZED/UNPOLARIZED calculations,
        # the subspace only includes wavefunctions with one value of Sz.
        # Therefore, if the subspace contains states with different spins Sᵢ,
        # the state energies have to be weighted with the spin multiplicity, 2*Sᵢ+1.
        # The total number of states, counting all components of the spin multiplets,
        # is given by
        #
        #   N =  ∑ᵢ (2*Sᵢ+1)
        #
        # and the average energy of the subspace is
        #
        #   E = 1/N ∑ᵢ (2*Sᵢ+1) H[D(r)]ᵢᵢ
        #
        # In INVARIANT calculations, all components of the multiplet are present,
        # and the state weights are set to 1.
        with torch.no_grad():
            # The weight is an integer, it does not require gradients.
            weights = msmd.state_weights()
        subspace_energy = MF.trace_average(H, weights=weights, subspace_dim=state_average)

        # There is no need to do a backward pass at this point.
        # The optimizer will do that if needed. The line searches
        # to find the best step length do not require gradients.

        return subspace_energy

    # Find stationary point of subspace energy
    optimizer.step(closure)

    # Diagonalize Hamiltonian at stationary point ...
    H = hamiltonian(msmd)
    eigvals, eigvecs = torch.linalg.eigh(H)

    # ... and transform optimized matrix density to the basis of the eigenstates
    # D(r) -> Uᵀ D(r) U
    msmd.basis_transformation(eigvecs.T)

    return eigvals, msmd
