#!/usr/bin/env python
# coding: utf-8
import pyscf.dft
import pyscf.gto

import torch
from torch import Size, Tensor
from torch.autograd import gradcheck
import torch.testing
from tqdm import tqdm
import unittest

from mlmsdft.dft.density import MultistateMatrixDensity
from mlmsdft.dft.density import MultistateMatrixDensityKohnSham
import mlmsdft.dft.nuclear as nuclear
from mlmsdft.dft.operator1e import OneElectronOperatorAO

from dft.fixture import FixtureMixin
from nn.test_functional import random_tensor


class NuclearFunctional(torch.nn.Module):
    def __init__(self, mol: pyscf.gto.Mole):
        """
        For testing purposes only.

        The external potential is the interaction energy between the electrons
        and the nuclei. The matrix elements of the nuclear potential in the basis
        of electronic states Ψᵢ is

        Vᵢⱼ = <Ψᵢ| ∑ₘ∑ₙ(-Zₘ)/|rₙ-Rₘ| |Ψⱼ> = ∫ ∑ₘ(-Zₘ)/|r-Rₘ| Dᵢⱼ(r) dr

            = ∫ v(r) D(r)ᵢⱼ dr

        :param mol: The nuclear charges Zₘ and coordinates Rₘ of the atoms
            in the molecule define the nuclear potential.
        :type mol: pyscf.gto.Mole
        """
        super().__init__()
        self.mol = mol

    def forward(self, coords: Tensor, density: Tensor) -> Tensor:
        """
        compute nuclear potential energy density on the integration grid

            ned[D]ᵢⱼ(r) = ∑ₘ(-Zₘ)/|r-Rₘ| Dᵢⱼ(r) = v(r) Dᵢⱼ(r)

        :param coords: coordinates r at which to calculate the product
            of the nuclear potential and the matrix density
        :type coords: Tensor of shape (...,3)

        :param density: matrix density summed over spins, Dᵢⱼ = Dᵅᵢⱼ(r) + Dᵝᵢⱼ(r)
            density[...,i,j] = Dᵢⱼ
        :type density: Tensor of shape (...,n,n)

        :return ned: nuclear potential energy density ned[D]ᵢⱼ
            ned[...,i,j] = v(r) Dᵢⱼ(r)
        :rtype ned: Tensor of shape (...,n,n)

        where the indices i,j=1,...,n run over the number of electronic states.
        """
        # Convert nuclear charges Z and positions R to torch tensors of the
        # same type and on the same device as `coords`.
        atom_charges = torch.from_numpy(self.mol.atom_charges()).to(
            dtype=coords.dtype, device=coords.device)
        atom_coords = torch.from_numpy(self.mol.atom_coords()).to(
            dtype=coords.dtype, device=coords.device)

        # Use broadcasting rules to compute |r-Rₘ| without an explicit for loop over atoms.
        #  coords         (...,3)  --unsqueeze-->   (...,    1,3)
        #  atom_coords                              (    natom,3)
        #  r_nuc_elec                               (...,natom)
        r_nuc_elec = torch.linalg.norm(coords.unsqueeze(-2) - atom_coords, dim=-1)
        # V(r) = ∑ₘ(-Zₘ)/|r-Rₘ|
        potential = -torch.sum(atom_charges / r_nuc_elec, dim=-1)

        # product V(r) Dᵢⱼ(r)
        # broadcasting rules
        #  potential     (...) --unsqueeze--> (...,1,1)
        #  density                            (...,n,n)
        ned = potential.unsqueeze(-1).unsqueeze(-1) * density

        return ned

    def integrate_potential_energy(
        self,
        coords: Tensor,
        density: Tensor,
        weights: Tensor) -> Tensor:
        """
        compute the matrix of the nuclear attraction operator in the subspace
        of electronic states D(r):

            Vᵢⱼ = ∫ ∑ₘ(-Zₘ)/|r-Rₘ| Dᵢⱼ(r) dr

                = ∫ v(r) Dᵢⱼ(r) dr

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).

        The potential energy density nedᵢⱼ(r) = v(r) Dᵢⱼ(r) is integrated numerically
        assuming the volume elements for each grid point are given in `weights`.

            Vᵢⱼ = ∑ₖ nedᵢⱼ(k) weights(k)

        :param coords: coordinates of the integration grid
        :type coords: Tensor of shape (...,3)

        :param density: matrix density summed over spins, Dᵢⱼ = Dᵅᵢⱼ(r) + Dᵝᵢⱼ(r)
            density[...,i,j] = Dᵢⱼ
        :type density: Tensor of shape (...,n,n)

        :param weights: integration weights on the grid
        :type weights: Tensor  of shape (...)

        :return potential_matrix: The nuclear potential energy matrix Vᵢⱼ in the subspace
           of the electronic states i,j=1,...,n
        :rtype potential_matrix: Tensor of shape (n,n)
        """
        # Evaluate the nuclear potential energy density on the grid.
        ned = self.forward(coords, density)

        # The matrix of the kinetic energy operator in the subspace is obtained
        # by integrating Vᵢⱼ(r) over space
        #
        #   Vᵢⱼ = ∫ v(r) Dᵢⱼ(r) dr = ∫ nedᵢⱼ(r) dr
        #
        potential_matrix = torch.einsum('...ij,...->ij', ned, weights)

        return potential_matrix


class TestNuclearPotential(unittest.TestCase, FixtureMixin):
    def check_nuclear_integrals(self, msmd: MultistateMatrixDensity):
        """
        compare the numerical integrals Vᵢⱼ = ∫ v(r) D(r)ᵢⱼ dr with the
        analytic molecular integrals using the AO basis, Vᵢⱼ(ref) = ∑ᵦ ∑ᵧ <𝛘ᵦ|v(r)|𝛘ᵧ> Dᵦᵧᵢⱼ.
        """
        # compute ∑ᵦ ∑ᵧ <𝛘ᵦ|v(r)|𝛘ᵧ> Dᵦᵧᵢⱼ from spin-traced matrix density
        dm = torch.einsum('ss...->...', msmd.density_matrices_ao())
        nuclear_functional_ao = nuclear.NuclearFunctionalAO(msmd.mol)
        nuclear_matrix_ref = nuclear_functional_ao(dm)

        # generate a multicenter integration grid
        grids = pyscf.dft.gen_grid.Grids(msmd.mol)
        grids.level = 4
        grids.build()

        # Evaluate density on the integration grid.
        D, _, _ = msmd.evaluate(grids.coords)
        # Sum over spins
        D = torch.einsum('ss...->...', D)
        coords = torch.from_numpy(grids.coords).to(
            dtype=D.dtype, device=D.device)
        weights = torch.from_numpy(grids.weights).to(
            dtype=D.dtype, device=D.device)

        nuclear_functional = NuclearFunctional(msmd.mol)
        # Compute Vᵢⱼ = ∫ v(r) D(r)ᵢⱼ
        nuclear_matrix = nuclear_functional.integrate_potential_energy(coords, D, weights)
        # The effective core potentials have to be added separately
        nuclear_matrix += OneElectronOperatorAO.apply(dm, msmd.mol, 'ECPscalar')

        torch.testing.assert_close(nuclear_matrix, nuclear_matrix_ref, atol=1.0e-5, rtol=1.0e-5)

        # The diagonal elements of the nuclear matrix should be all negative.
        self.assertTrue(torch.all(torch.diag(nuclear_matrix) <= 0.0))

    def test_nuclear_integrals(self):
        # NOTE: For periodic systems the matrix elements of the
        # long-ranged Coulomb interaction cannot be calculated by
        # integrating over the unit cell. Therefore these tests
        # only work for molecules.
        for name, mol in tqdm(self.create_test_molecules().items()):
            for msmd in self.create_random_matrix_densities(mol):
                with self.subTest(molecule=name, msmd=msmd.__class__.__name__):
                    self.check_nuclear_integrals(msmd)

    def check_autograd(self, mol):
        """
        Compare analytical and numerical gradients of nuclear matrix V[D(r)]ᵢⱼ
        with respect to parameters matrix density.
        """
        nuclear_functional_ao = nuclear.NuclearFunctionalAO(mol)

        msmd_0 = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random")
        orbital_coefficients = msmd_0.orbital_coefficients()

        def wrapper_function(orbital_rotation_params):
            msmd = MultistateMatrixDensityKohnSham(
                mol, orbital_coefficients, orbital_rotation_params
            )
            # representation of matrix density in AO basis summed over spins,
            # Dᵦᵧᵢⱼ = Dᵅᵅᵦᵧᵢⱼ + Dᵝᵝᵦᵧᵢⱼ
            density_matrices_ao = torch.einsum('ss...->...', msmd.density_matrices_ao())
            # compute nuclear matrix V[D(r)]ᵢⱼ
            outputs = nuclear_functional_ao(density_matrices_ao)
            return outputs

        nao, nmo = orbital_coefficients.size()
        # Random orbital rotation parameters.
        nrot = (nmo*(nmo-1))//2
        orbital_rotation_params = random_tensor(Size([nrot]))
        orbital_rotation_params.requires_grad_(True)

        # Compare analytical gradients computed by backpropagation
        # finite difference approximation.
        inputs = (orbital_rotation_params,)
        gradcheck(wrapper_function, inputs)

    def test_autograd(self):
        """
        Check that automatic differentiation works for D(X,r) w/r/t X.
        """
        for name, mol in tqdm(self.create_test_systems_minimal().items()):
            with self.subTest(molecule=name):
                self.check_autograd(mol)


if __name__ == "__main__":
    unittest.main()
