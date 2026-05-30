#!/usr/bin/env python
# coding: utf-8
from abc import ABC
import pyscf.dft
import pyscf.gto
import torch
from torch import Tensor
from torch.autograd import gradcheck
import torch.testing
from tqdm import tqdm
import unittest

from mlmsdft.dft.density import MultistateMatrixDensityKohnSham
import mlmsdft.dft.kinetic as kinetic

from dft.fixture import FixtureMixin
from nn.test_functional import random_tensor


class KineticSemilocalBase(torch.nn.Module, ABC):
    def integrate_kinetic_energy(
        self,
        density: Tensor,
        grad_density: Tensor,
        lapl_density: Tensor,
        weights: Tensor,
        available_memory=1<<30):
        """
        compute the matrix of the kinetic energy operator in the subspace
        of electronic states by evaluating the kinetic energy functional T[D(r)]
        on the matrix density D(r):

            Tᵢⱼ = <Ψᵢ|-1/2 ∑ₙ∇ₙ²|Ψⱼ> = T[D(r)]ᵢⱼ

                = ∫ t(D(r), ∇D(r))ᵢⱼ dr

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).

        The kinetic energy density t(D(r), ∇D(r), ∇²D(r))ᵢⱼ is integrated numerically
        assuming the volume elements for each grid point are given in `weights`.

            Tᵢⱼ = ∑ₖ tᵢⱼ(k) weights(k)

        :param density: matrix density summed over spins, Dᵢⱼ = Dᵅᵢⱼ(r) + Dᵝᵢⱼ(r)
            density[...,i,j] = Dᵢⱼ
        :type density: Tensor of shape (...,n,n)

        :param grad_density: gradient of matrix density summed over spins,
        :type grad_density: Tensor of shape (...,3,n,n)
            grad_density[...,a,i,j] = ∇ₐDᵢⱼ with a = 0(x),1(y),2(z)

        :param lapl_density: Laplacian of matrix density summed over spins,
        :type lapl_density: Tensor of shape (...,n,n)
            lapl_density[...,i,j] = ∇²Dᵢⱼ

        :param weights: integration weights on the grid
        :type weights: Tensor  of shape (...)

        :param available_memory: The amount of memory (in bytes) that can be
           allocated for the kinetic energy density. If more memory is needed,
           the KED is evaluated in multiple chunks. (1<<30 corresponds to 1Gb)
           Since more memory is needed for intermediate quantities, this limit
           is only a rough estimate.
        :type available_memory: int

        :return kinetic_matrix: The kinetic energy matrix Tᵢⱼ in the subspace
           of the electronic states i,j=1,...,n
        :rtype kinetic_matrix: Tensor of shape (n,n)
        """
        # number of electronic states in the subspace
        n = density.size()[-1]
        # number of grid points
        ncoord = torch.numel(weights)
        # matrix element of the kinetic energy operator <i|Top|j>
        kinetic_matrix = torch.zeros(
            torch.Size([n,n]),
            dtype=density.dtype,
            requires_grad=density.requires_grad,
            device=density.device
        )

        # If the resulting array that holds the kinetic energy density
        # exceeds `available_memory`, the KED is evaluated on smaller chunks
        # of the grid and summed into the kinetic matrix at the end.
        # The factor 100 is arbitrary.
        needed_memory = 100 * kinetic_matrix.element_size() * n**2 * torch.numel(weights)
        number_of_chunks = max(1, (needed_memory + available_memory) // available_memory)
        # There cannot be more chunks than grid points.
        number_of_chunks = min(ncoord, number_of_chunks)

        # Loop over chunks of density descriptors and associated integration weights.
        for density_, grad_density_, lapl_density_, weights_ in zip(
            torch.split(density, number_of_chunks, dim=0),
            torch.split(grad_density, number_of_chunks, dim=0),
            torch.split(lapl_density, number_of_chunks, dim=0),
            torch.split(weights, number_of_chunks, dim=0)):

            # Evaluate the kinetic energy density on the grid.
            ked_ = self.forward(density_, grad_density_, lapl_density_)

            # The matrix of the kinetic energy operator in the subspace is obtained
            # by integrating T_{i,j}(r) over space
            #
            #   Tᵢⱼ = ∫ KEDᵢⱼ(r) dr
            #
            kinetic_matrix = kinetic_matrix + torch.einsum('...ij,...->ij', ked_, weights_)

        return kinetic_matrix

    
class KineticVonWeizsaecker(KineticSemilocalBase):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy density in the subspace.

    The von-Weizsäcker functional for the electronic ground state

                      (∇ρ)²
           t[ρ] = 1/8 ----
                       ρ

    is turned into a matrix-density functional by replacing the density
    with the matrix density, ρ(r) -> D(r),

           t[D]ᵢⱼ = 1/8 ∑ₖ∑ₗ ∇Dᵢₖ D⁻¹ₖₗ ∇Dₗⱼ

    The matrix-inverse of D is placed symmetrically between the gradients,
    but this is not the only possibility.
    """
    def forward(self, density: Tensor, grad_density: Tensor, lapl_dummy: Tensor = None) -> Tensor:
        """
        compute von-Weizsäcker kinetic energy density t[D]ᵢⱼ

        :param density: matrix density summed over spins, Dᵢⱼ = Dᵅᵅᵢⱼ(r) + Dᵝᵝᵢⱼ(r)
            density[...,i,j] = Dᵢⱼ
        :type density: Tensor of shape (...,n,n)

        :param grad_density: gradient of matrix density summed over spins,
        :type grad_density: Tensor of shape (...,3,n,n)
            grad_density[...,a,i,j] = ∇ₐDᵢⱼ with a = 0(x),1(y),2(z)

        :return ked: kinetic energy density t[D]ᵢⱼ
            ked[...,i,j] = tᵢⱼ
        :rtype ked: Tensor of shape (...,n,n)

        where the indices i,j=1,...,n run over the number of electronic states.
        """
        # (pseudo) inverse D⁻¹
        invD = torch.linalg.pinv(
            density,
            # The matrix density is symmetric.
            hermitian=True,
            # Treat small singular values (< sigma_max * rtol) as 0.
            rtol=1.0e-12)
        #
        # KED_{i,j}(r) = 1/8 ∑ₖ∑ₗ ∇D_{i,k} D⁻¹_{k,l} ·∇D_{l,j}
        #
        ked = 1.0/8.0 * torch.einsum(
            '...aik,...kl,...alj->...ij', grad_density, invD, grad_density)

        return ked


class TestKineticFunctionalAO(unittest.TestCase, FixtureMixin):
    def check_kinetic_integrals_kohn_sham(self, msmd: MultistateMatrixDensityKohnSham):
        """
        For a 1-electron system and a single state the von-Weizsäcker
        kinetic energy functional should be exact.
        """
        assert msmd.number_of_electrons == 1
        assert msmd.number_of_states == 1
        # generate a multicenter integration grid
        grids = pyscf.dft.gen_grid.Grids(msmd.mol)
        grids.level = 4
        grids.build()

        # Evaluate density on the integration grid.
        D, grad_D, lapl_D = msmd.evaluate(grids.coords)
        # sum over spin
        D = torch.einsum('ss...->...', D)
        grad_D = torch.einsum('ss...->...', grad_D)
        lapl_D = torch.einsum('ss...->...', lapl_D)

        weights = torch.from_numpy(grids.weights).to(
            dtype=D.dtype, device=D.device)

        # compute kinetic energy from D(r)ᵢⱼ
        kinetic_functional = KineticVonWeizsaecker()
        kinetic_matrix_ref = kinetic_functional.integrate_kinetic_energy(D, grad_D, lapl_D, weights)

        # Compute Tᵢⱼ = ∑ᵦ ∑ᵧ <𝛘ᵦ|-1/2 ∇²|𝛘ᵧ> Dᵦᵧᵢⱼ from AO representation
        # of spin-traced matrix density.
        dm = torch.einsum('ss...->...', msmd.density_matrices_ao())
        kinetic_functional_ao = kinetic.KineticFunctionalAO(msmd.mol)
        kinetic_matrix = kinetic_functional_ao(dm)

        torch.testing.assert_close(kinetic_matrix, kinetic_matrix_ref, atol=1.0e-5, rtol=1.0e-5)

        # The diagonal elements of the kinetic matrix should be all positive.
        self.assertTrue(torch.all(torch.diag(kinetic_matrix) >= 0.0))

    def test_kinetic_integrals_kohn_sham(self):
        for name, mol in tqdm(self.create_test_molecules_single_electron().items()):
            with self.subTest(molecule=name):
                # Kohn-Sham (single state)
                msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random", seed=222)
                self.check_kinetic_integrals_kohn_sham(msmd)

    def check_autograd_kohn_sham(self, mol):
        """
        Compare analytical and numerical gradients of kinetic matrix T[D(r)]ᵢⱼ
        with respect to parameters of matrix density.
        """
        kinetic_functional_ao = kinetic.KineticFunctionalAO(mol)

        msmd_0 = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random", seed=12345)
        orbital_coefficients = msmd_0.orbital_coefficients()

        def wrapper_function(orbital_rotation_params):
            msmd = MultistateMatrixDensityKohnSham(
                mol, orbital_coefficients, orbital_rotation_params
            )
            # representation of spin-traced matrix density in AO basis, Dᵦᵧᵢⱼ
            density_matrices_ao = torch.einsum('ss...->...', msmd.density_matrices_ao())
            # compute kinetic matrix T[D(r)]ᵢⱼ
            outputs = kinetic_functional_ao(density_matrices_ao)
            return outputs

        _, nmo = orbital_coefficients.size()
        # Random orbital rotation parameters.
        nrot = (nmo*(nmo-1))//2
        orbital_rotation_params = random_tensor(torch.Size([nrot]))
        orbital_rotation_params.requires_grad_(True)

        # Compare analytical gradients computed by backpropagation
        # finite difference approximation.
        inputs = (orbital_rotation_params,)
        gradcheck(wrapper_function, inputs)

    def test_autograd_kohn_sham(self):
        """
        Check that automatic differentiation works for D(U_mo,r) w/r/t U_mo.
        """
        for name, mol in tqdm(self.create_test_systems_minimal().items()):
            with self.subTest(molecule=name):
                self.check_autograd_kohn_sham(mol)


if __name__ == "__main__":
    unittest.main()
