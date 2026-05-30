#!/usr/bin/env python
# coding: utf-8
import numpy
import pyscf.gto
import torch
from torch import Tensor
import torch.linalg
import torch.testing
from tqdm import tqdm
import types
import unittest
import warnings

from mlmsdft.dft.density import MultistateMatrixDensity
from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.density import MultistateMatrixDensityKohnSham
from mlmsdft.dft.hamiltonian import Hamiltonian
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import concat_spin_blocks, spin_trace
from mlmsdft.dft.xc import lda_c_chachiyo, lda_x_dirac
from mlmsdft.dft.pure import (
    LDA,
)
import mlmsdft.nn.functional as MF

from dft.fixture import FixtureMixin
# Need to change name of class to avoid running unittest on it.
from dft.test_density import TestMultistateMatrixDensityCAS as MultistateMatrixDensityCASTest

kinetic_functionals = [
    # Here `None` means that the kinetic energy matrix is not computed from
    # the density but from the wavefunctions.
    None,
]

correlation_functionals = {
    "lda": [lda_c_chachiyo],
}

exchange_functionals = {
    "lda": [lda_x_dirac],
}

# Classes for instantiating pure composite functionals.
# These functionals have to be intantiated with one dummy argument,
# which is not used, so that they have the same interface as the
# hybrid functionals
pure_xc_functionals = {
    "LDA": LDA,
}


# This member function is dynamically attached to an instance of HamiltonianSemilocal
# for testing purposes
def _matrix_elements_single_chunk(self: HamiltonianSemilocal, msmd: MultistateMatrixDensity) -> Tensor:
    """
    Compute the Hamiltonian matrix H[D(r)]ᵢⱼ in the basis of electronic states at a
    given matrix density D(r). This function is a copy of HamiltonianSemilocal.matrix_elements(...)
    with the only difference that the integration over grid points is performed in a single
    without splitting the grid into chunks.

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
                # (Dᵝᵅ Dᵝᵝ)
                super_spin_dm = concat_spin_blocks(spin_dm)
                # The exact exchange matrix functional operates on the supermatrix
                # to get a (2N)*(2N) exchange matrix
                # (Exxᵅᵅ Exxᵅᵝ)          (Dᵅᵅ Dᵅᵝ)
                # (           ) = Exx { (       ) }
                # (Exxᵝᵅ Exxᵝᵝ )          (Dᵝᵅ Dᵝᵝ)
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

    # evaluate semilocal kinetic, exchange and correlation functionals
    # Inputs are D(r) and ∇D(r) on the integration grid.
    # NOTE: ∇²D is only calculated if it is needed by any of the functionals.
    spin_D, grad_spin_D, lapl_spin_D = msmd.evaluate(
        self.grids.coords, need_laplacian=self.need_laplacian)

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
                # (Dᵝᵅ  Dᵝᵝ)
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
                spin_polarized=True
            )
        else:
            c = self.correlation(D, grad_D, lapl_D)
    else:
        c = torch.zeros_like(D, dtype=D.dtype, device=D.device)

    # integrate energy densities
    weights = torch.from_numpy(self.grids.weights).to(dtype=D.dtype, device=D.device)
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

    return H


class TestHamiltonian(unittest.TestCase, FixtureMixin):
    def test_minimize_kohn_sham_energy_h2(self):
        """
        Minimize energy of a Kohn-Sham determinant starting from
        a matrix density with core hamiltonian guess.
        """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="hcore")
        for kinetic_functional in kinetic_functionals:
            with self.subTest(kinetic_functional=str(kinetic_functional)):
                hamiltonian = HamiltonianSemilocal(mol, kinetic_functional=kinetic_functional)
                self.check_minimize_subspace_energy(msmd, hamiltonian)

    def test_minimize_cas_energy_h2(self):
        """
        Minimize subspace energy of a (2,2) complete active space starting from
        a matrix density with core hamiltonian guess.

        TODO: This long test should be broken up in many smaller tests.
        """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        devices = ['cpu']
        if torch.cuda.is_available():
            # Run tests of GPU, too
            devices.append('cuda')
        else:
            print("CUDA not available, tests are only run on CPU.")

        # Run tests of available devices.
        for device in devices:
            for spin_symmetry in [True, False]:
                for max_level in [1, numpy.inf]:
                    # Loop over available kinetic funtional
                    for kinetic_functional in [None]:
                        # Name of kinetic functional
                        if kinetic_functional is None:
                            kinetic_functional_name = "AO"
                        else:
                            kinetic_functional_name = kinetic_functional.__class__.__name__
                        # Loop over types of xc-functionals
                        for xc_type in ["lda"]:
                            # Loop over available xc-functionals
                            for exchange_functional in exchange_functionals[xc_type]:
                                for correlation_functional in correlation_functionals[xc_type]:
                                    # Whether to calculated the exchange energy from the total
                                    # density, separately for each spin type or from the spin supermatrix.
                                    for spin_type in SpinType:
                                        with self.subTest(
                                            device=device,
                                            spin_symmetry=spin_symmetry,
                                            max_level=max_level,
                                            t=kinetic_functional_name,
                                            x=exchange_functional.__name__,
                                            c=correlation_functional.__name__,
                                            spin_type=spin_type
                                        ):
                                            msmd = MultistateMatrixDensityCAS.from_guess(
                                                mol, 2, 2,
                                                spin_symmetry=spin_symmetry,
                                                spin_type=spin_type,
                                                max_level=max_level,
                                                guess="hcore"
                                            )
                                            hamiltonian = HamiltonianSemilocal(
                                                mol,
                                                kinetic_functional=kinetic_functional,
                                                exchange_functional=exchange_functional,
                                                correlation_functional=correlation_functional,
                                                spin_type=spin_type
                                            )
                                            self.check_minimize_subspace_energy(msmd, hamiltonian, device=device)

    def test_minimize_cas_energy_h2_reduced_subspace_average(self):
        """
        Minimize subspace energy of a (2,2) complete active space.

        The number of states to average over is varied.
        """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        for spin_symmetry in [True, False]:
            for max_level in [1, numpy.inf]:
                for state_average in [1,2,3,4]:
                    for spin_type in [SpinType.UNPOLARIZED]:
                        with self.subTest(
                            spin_symmetry=spin_symmetry,
                            max_level=max_level,
                            state_average=state_average,
                            spin_type=spin_type,
                        ):
                            msmd = MultistateMatrixDensityCAS.from_guess(
                                mol, 2, 2,
                                spin_symmetry=spin_symmetry,
                                spin_type=spin_type,
                                max_level=max_level,
                                guess="hcore"
                            )
                            xc_functional = LDA(mol)
                            hamiltonian = HamiltonianSemilocal(
                                mol,
                                exchange_functional = xc_functional.exchange,
                                correlation_functional = xc_functional.correlation,
                                spin_type = spin_type
                            )
                            self.check_minimize_subspace_energy(
                                msmd, hamiltonian,
                                state_average=min(state_average, msmd.number_of_states)
                            )

    def check_minimize_subspace_energy(
        self,
        msmd: MultistateMatrixDensity,
        hamiltonian: Hamiltonian,
        state_average=None,
        device='cpu'
    ):
        """
        Solve for lowest few electronic states by minimizing the subspace energy
        and check that at the minimum the gradient on the parameters of the
        matrix density is zero.

        :param state_average: How many states should be averaged over?
            None means to include all states in the subspace
        :type state_average: int or None
        """
        # Move matrix density to CPU or GPU.
        msmd.to(device=device)

        # Minimize the state-averaged energy
        energies, msmd = minimize_subspace_energy(hamiltonian, msmd,
            state_average=state_average,
            # show how energy and gradient norm decreases during minimization.
            debug=1
        )

        # check that Hamiltonian is diagonal in the basis of eigenstates
        H = hamiltonian(msmd)
        torch.testing.assert_close(H, torch.diag(energies))

        # Check that the matrix density is a stationary point, i.e. that the
        # gradients on the parameters vanish.
        msmd.zero_grad()
        H = hamiltonian(msmd)
        # Average energy of all electronic states, weighing states with their spin multiplicity
        # E = 1/N tr(H) = 1/N ∑ᵢ H[D(r)]ᵢᵢ
        #   = ∑ᵢ (2 Sᵢ + 1) H[D(r)]ᵢᵢ / (∑ⱼ (2 Sⱼ + 1))
        # In INVARIANT calculations, all components of the multiplet are present,
        # and the state weights are set to 1.
        with torch.no_grad():
            # The weight is an integer, it does not require gradients.
            weights = msmd.state_weights()
        subspace_energy = MF.trace_average(H, weights=weights, subspace_dim=state_average)
        # compute gradients
        subspace_energy.backward()

        # Check that gradients are close to zero.
        for param in msmd.parameters():
            if torch.isnan(param.grad).all():
                # `param` is most likely a constant. Taken the gradient of a constant gives NaN
                warnings.warn(
                    "Gradients on parameter are all NaN. The parameter is probably a constant."
                )
            else:
                torch.testing.assert_close(param.grad, torch.zeros_like(param.grad), atol=1.0e-5, rtol=1.0e-5)

    def check_lda_kohn_sham_energy_vs_pyscf(self, mol):
        """
        For a single Kohn-Sham state the same total energy should be obtained as with pyscf's RKS.
        """
        assert mol.spin == 0, (
            "MultistateMatrixDensityKohnSham only works for closed-shell singlet states"
        )
        # Minimize energy of a Kohn-Sham determinant starting from a Hcore guess matrix density.
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="hcore")
        hamiltonian = HamiltonianSemilocal(
            mol,
            # compute kinetic energy from wavefunctions
            kinetic_functional = None,
            exchange_functional = lda_x_dirac,
            correlation_functional = lda_c_chachiyo
        )

        # Minimize the state-averaged energy. There is only a single state, so this is the same
        # as minimizing the ground state energy.
        energies, msmd = minimize_subspace_energy(hamiltonian, msmd)

        kohn_sham_energy = energies[0].detach().numpy()

        # pyscf should give the same energy.
        rks = pyscf.dft.RKS(mol)
        rks.xc = 'LDA_X,LDA_C_CHACHIYO'
        rks.verbose = 0
        rks.kernel()
        kohn_sham_energy_pyscf = rks.e_tot

        self.assertAlmostEqual(kohn_sham_energy, kohn_sham_energy_pyscf, places=6)

    def test_lda_kohn_sham_energy_vs_pyscf(self):
        """
        Compare Kohn-Sham energies for LDA functionals with pyscf
        """
        for name, mol in tqdm(self.create_test_molecules_closed_shell().items()):
            with self.subTest(molecule=name):
                self.check_lda_kohn_sham_energy_vs_pyscf(mol)

    def check_composite_pure_xc_functional_vs_pyscf(
        self, mol, xc_functional_class, xc_code, spin_type=SpinType.UNPOLARIZED):
        """
        For a single Kohn-Sham state the same total energy should be obtained as with pyscf's RKS.
        """
        assert mol.spin == 0, (
            "MultistateMatrixDensityKohnSham only works for closed-shell singlet states"
        )
        xc_functional = xc_functional_class(mol)
        # Pure functionals do not have any exact exchange.
        self.assertEqual(xc_functional.exact_exchange, None)

        # Minimize energy of a Kohn-Sham determinant starting from a Hcore guess matrix density.
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="hcore")
        hamiltonian = HamiltonianSemilocal(
            mol,
            # compute kinetic energy from wavefunctions
            kinetic_functional = None,
            exchange_functional = xc_functional.exchange,
            correlation_functional = xc_functional.correlation,
            exact_exchange_functional = xc_functional.exact_exchange,
            spin_type = spin_type
        )

        # Minimize the state-averaged energy. There is only a single state, so this is the same
        # as minimizing the ground state energy.
        energies, msmd = minimize_subspace_energy(hamiltonian, msmd)

        kohn_sham_energy = energies[0].detach().numpy()

        if 'MGGA' in xc_code:
            # For BR89LYP we just check that the code runs through without errors.
            # We cannot compare the results with pyscf, since Laplacians are not implemented
            # for Meta GGAs in pyscf.
            return
        # pyscf should give the same energy.
        rks = pyscf.dft.RKS(mol)
        rks.xc = xc_code
        rks.verbose = 0
        rks.kernel()
        kohn_sham_energy_pyscf = rks.e_tot

        self.assertAlmostEqual(kohn_sham_energy, kohn_sham_energy_pyscf, places=6)

    def test_composite_pure_xc_functional_vs_pyscf(self):
        """
        Compare Kohn-Sham energies for pure composite xc-functional with pyscf
        """
        mol = self.create_test_molecules()['hydrogen molecule']
        for spin_type in SpinType:
            for xc_functional_class, xc_code in [
                    (LDA, 'LDA_X,LDA_C_CHACHIYO'),
            ]:
                # For a single, closed-shell state the spin-polarized and unpolarized
                # calculations should give the same results.
                with self.subTest(
                    xc_functional=xc_functional_class.__name__,
                    spin_type=spin_type,
                ):
                    self.check_composite_pure_xc_functional_vs_pyscf(
                        mol, xc_functional_class, xc_code, spin_type=spin_type)

    def check_basis_transformation_cas(self, msmd: MultistateMatrixDensityCAS):
        """
        Check that after transforming the matrix density with eigenvectors
        of the Hamiltonian, the Hamiltonian matrix becomes diagonal.
        """
        hamiltonian = HamiltonianSemilocal(msmd.mol)

        # Diagonalize Hamiltonian ...
        H = hamiltonian(msmd)
        eigvals, eigvecs = torch.linalg.eigh(H)

        # ... and transform optimized matrix density to the basis of the eigenstates
        msmd.basis_transformation(eigvecs.T)

        # Hamiltonian after basis transformation should be diagonal.
        H = hamiltonian(msmd)
        eigvals, eigvecs = torch.linalg.eigh(H)
        torch.testing.assert_close(H, torch.diag(eigvals))

    def test_basis_transformation_cas(self):
        """ Check that matrix density transforms as expected under a change of basis """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        for spin_symmetry in [True, False]:
            for spin_type in SpinType:
                for max_level in [1, 2, numpy.inf]:
                    with self.subTest(
                        spin_symmetry=spin_symmetry,
                        spin_type=spin_type,
                        max_level=max_level
                    ):
                        msmd = MultistateMatrixDensityCASTest.create_random_matrix_density_cas(
                            mol, 2, 2,
                            spin_symmetry=spin_symmetry, spin_type=spin_type, max_level=max_level
                        )
                        self.check_basis_transformation_cas(msmd)

    def test_spin_invariant_multiplet_degeneracy_lda(self):
        """
        Check that all components of the triplet states in the CAS(2,2) of H2
        are degenerate but different in energy from the open-shell singlet state:

            E(S0) ≠ E(S1) ≠ E(S2) ≠ E(T1,Sz=-1) = E(T1,Sz=0) = E(T1,Sz=+1)
        """
        # Hydrogen molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        msmd = MultistateMatrixDensityCASTest.create_random_matrix_density_cas(
            # 2 electrons in 2 orbitals gives rise to 3 singlet states and 3 triplet states
            mol, 2, 2,
            # any spin S²
            spin_symmetry=False,
            # any spin projection Sz=-S,...,+S
            spin_type=SpinType.INVARIANT,
        )

        xc_functional = LDA(msmd.mol)
        hamiltonian = HamiltonianSemilocal(
            msmd.mol,
            # exchange, correlation and exact exchange parts of hybrid functional.
            exchange_functional = xc_functional.exchange,
            correlation_functional = xc_functional.correlation,
            exact_exchange_functional = xc_functional.exact_exchange,
            spin_type = SpinType.INVARIANT
        )

        # Diagonalize Hamiltonian ...
        H = hamiltonian(msmd)
        eigvals, eigvecs = torch.linalg.eigh(H)

        # ... and transform optimized matrix density to the basis of the eigenstates
        msmd.basis_transformation(eigvecs.T)

        # Check that the energy levels show the correct degeneracies due to spin multiplets
        s2 = msmd.spin_s2_expectation().detach().cpu().numpy()
        s2 = numpy.round(s2, decimals=2)
        energies = eigvals.detach().cpu().numpy()
        # Check for degeneracy within 1.0e-8
        energies = numpy.round(energies, decimals=8)
        # The energies of the three singlet states should be all different
        singlet_energies = energies[s2 == 0.0]
        self.assertEqual( len(numpy.unique(singlet_energies)), 3 )
        # The triplet energies should be all the same
        triplet_energies = energies[s2 == 2.0]
        self.assertEqual( len(numpy.unique(triplet_energies)), 1 )
        # The triplet energies should be different from any singlet energy
        # E(S0) ≠ E(S1) ≠ E(S2) ≠ E(T1,Sz=-1) = E(T1,Sz=0) = E(T1,Sz=+1)
        self.assertEqual( len(numpy.unique(energies)), 4 )

    def check_matrix_elements_chunking(
        self,
        hamiltonian_ref: HamiltonianSemilocal,
        msmd: MultistateMatrixDensity
    ):
        """
        The Hamiltonian matrix elements are computed by splitting the integration
        grid into chunks so that each chunk fits into memory. This test checks
        that the final H matrix does not depend on the number of chunks.
        """
        # overwrite method for computing matrix elements
        hamiltonian_ref.matrix_elements = types.MethodType(
            _matrix_elements_single_chunk, hamiltonian_ref)
        # reference Hamiltonian
        H_ref = hamiltonian_ref(msmd)

        for chunks in [1,2,3,4]:
            # copy of hamiltonian_ref with chunking
            hamiltonian = HamiltonianSemilocal(
                hamiltonian_ref.mol,
                kinetic_functional = hamiltonian_ref.kinetic,
                exchange_functional = hamiltonian_ref.exchange,
                correlation_functional = hamiltonian_ref.correlation,
                grid_level = hamiltonian_ref.grid_level,
                grid_chunks = chunks
            )
            H = hamiltonian(msmd)
            torch.testing.assert_close(H, H_ref)

    def test_matrix_elements_chunking(self):
        """
        Test that hamiltonian does not depend on how many chunks are used to compute it.
        """
        mol = self.create_test_molecules()["hydrogen molecule"]

        devices = ['cpu']
        if torch.cuda.is_available():
            # Run tests of GPU, too
            devices.append('cuda')
        else:
            print("CUDA not available, tests are only run on CPU.")

        for device in devices:
            # Random matrix density
            for msmd in tqdm(self.create_random_matrix_densities(mol)):
                msmd.to(device)
                for kinetic_functional in kinetic_functionals:
                    with self.subTest(device=device, kinetic_functional=str(kinetic_functional)):
                        hamiltonian = HamiltonianSemilocal(mol, kinetic_functional=kinetic_functional)
                        self.check_matrix_elements_chunking(hamiltonian, msmd)


if __name__ == "__main__":
    unittest.main()
