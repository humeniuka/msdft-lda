#!/usr/bin/env python
# coding: utf-8
import numpy
import torch
from torch import Size
import unittest

import pyscf.dft

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.xc import lda_x_dirac
from mlmsdft.dft.xc import lda_c_chachiyo

from nn.test_functional import random_orthogonal_matrix
from nn.test_functional import random_tensor
from nn.test_functional import (
    check_autograd,
    check_transformation_property
)

lda_xc_functionals = [
    lda_x_dirac,
    lda_c_chachiyo,
]


class TestLDAMatrixFunctionals(unittest.TestCase):

    def test_autograd(self):
        # Loop over matrix functions that need to be tested.
        for func in lda_xc_functionals:
            # Loop over matrix dimensions.
            for n in [1,2,3]:
                with self.subTest(function=func, dimension=n):
                    size = Size([2,1])
                    check_autograd(func, size, n, positive_definite=True)

    def test_transformation_property(self):
        # Loop over matrix functions that need to be tested.
        for func in lda_xc_functionals:
            # Loop over size of tensor (excluding matrix dimensions)
            for size in [torch.Size([1]), torch.Size([2,3])]:
                # Loop over matrix dimensions.
                for n in [1,2,3]:
                    with self.subTest(function=func, size=size, dimension=n):
                        check_transformation_property(
                            func, size, n, positive_definite=True)


class TestLDAFunctionalsVsLibxc(unittest.TestCase):
    """
    compare LDA exchange and correlation matrix functionals
    with libxc's implementation on random scalar densities
    """
    def test_lda_c_chachiyo(self):
        ncoord = 10
        # random positive definite 1x1 matrix density
        nstate = 1
        size = Size([ncoord,nstate,nstate])
        D = random_tensor(size)**2
        # compute correlation energy density using our own implementation
        ced = lda_c_chachiyo(D)[:,0,0]

        # compute correlation energy density using libxc
        # rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,tau)
        # but only den has to be non-zero for an LDA functional.
        rho = numpy.zeros((1, ncoord))
        rho[0,:] = D[:,0,0].numpy()
        exc, _, _, _ = pyscf.dft.libxc.eval_xc(',LDA_C_CHACHIYO', rho)
        # correlation energy density, εᶜ(r) ρ(r)
        ced_ref = torch.from_numpy(exc * rho[0,:]).to(
            dtype=ced.dtype, device=ced.device)
        # compare
        torch.testing.assert_close(ced, ced_ref, rtol=1.0e-5, atol=1.0e-5)

    def test_lda_x_dirac(self):
        ncoord = 10
        # random positive definite 1x1 matrix density
        nstate = 1
        size = Size([ncoord,nstate,nstate])
        D = random_tensor(size)**2
        # compute exchange energy density using our own implementation
        # For ρᵅ=ρᵝ=ρ/2, we can calculate the exchange energy as
        # xed = xed[ρᵅ]+xed[ρᵝ] = 2*xed[ρ/2]
        xed = 2.0 * lda_x_dirac(D/2.0)[:,0,0]

        # compute exchange energy density using libxc
        # rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,tau)
        # but only den has to be non-zero for an LDA functional.
        rho = numpy.zeros((1, ncoord))
        rho[0,:] = D[:,0,0].numpy()
        exc, _, _, _ = pyscf.dft.libxc.eval_xc('LDA_X,', rho)
        # exchange energy density, εₓ(r) ρ(r)
        xed_ref = torch.from_numpy(exc * rho[0,:]).to(
            dtype=xed.dtype, device=xed.device)
        # compare
        torch.testing.assert_close(xed, xed_ref, rtol=1.0e-5, atol=1.0e-5)


class TestMatrixFunctionalsTransformationPropertyOnGrid(unittest.TestCase):
    """
    Check that the matrix functionals are equivariant for realistic matrix density
    at all grid points of the multicenter integration grid
    """
    def check_transformation_property(self, functional):
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'He 0 0 0',
            basis = '6-31g',
            charge = 0,
            spin = 0)
        # CAS(2,2) matrix density
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=True, guess="hcore")
        # number of states
        nstate = msmd.number_of_states

        # generate a multicenter integration grid
        grids = pyscf.dft.gen_grid.Grids(mol)
        grids.level = 5 # 3
        grids.build()

        # Random orthogonal transformation matrix.
        with torch.random.fork_rng():
            torch.manual_seed(345345)
            L = random_orthogonal_matrix(nstate)
        Lt = torch.transpose(L, 0, 1)
        # Check  that L really is orthogonal, i.e. L Lᵀ = 1
        Id = torch.matmul(L, Lt)
        torch.testing.assert_close(Id, torch.eye(nstate, dtype=L.dtype))

        # evaluate semilocal kinetic, exchange and correlation functionals
        # Inputs are D(r), ∇D(r) and ∇²D on the integration grid.
        D, grad_D, lapl_D = msmd.evaluate(grids.coords)

        # Compute F(X1,X2,...,)
        Xs = (D, grad_D, lapl_D)
        F_X = functional(*Xs)
        # transform input arguments, Xi' = L Xi Lᵀ
        Xps = []
        for X in Xs:
            Xp = torch.einsum('ia,...ab,bj->...ij', L, X, Lt)
            Xps.append(Xp)
        # transform output, F' = L F(X1,X2,...) Lᵀ
        Fp = torch.einsum('ia,...ab,bj->...ij', L, F_X, Lt)
        # Compute F of transformed output, F(L X1 Lᵀ, L X2 Lᵀ)
        F_Xp = functional(*Xps)
        # Check that F' = L F(X1,X2) Lᵀ = F(X1',X2') = F(L X1 Lᵀ,L X2 Lᵀ)
        torch.testing.assert_close(Fp, F_Xp)

    def test_equivariance_lda_x_dirac(self):
        self.check_transformation_property(lda_x_dirac)

    def test_equivariance_lda_c_chachiyo(self):
        self.check_transformation_property(lda_c_chachiyo)


if __name__ == "__main__":
    unittest.main()
