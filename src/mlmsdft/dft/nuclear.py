# -*- coding: utf-8 -*-
import pyscf.gto

import torch
from torch import Tensor

from mlmsdft.dft.operator1e import OneElectronOperatorAO


class NuclearFunctionalAO(torch.nn.Module):
    def __init__(self, mol: pyscf.gto.Mole):
        """
        external (or nuclear) potential

        Vᵢⱼ[D(r)] = ∫ v(r) D(r)ᵢⱼ dr

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole
        """
        super().__init__()
        self.mol = mol

    def forward(self, density_matrices_ao: Tensor) -> Tensor:
        """
        compute the nuclear potential matrix using the AO integrals.

            Vᵢⱼ = ∫ v(r) D(r)ᵢⱼ dr = ∑ᵦ ∑ᵧ <𝛘ᵦ|v(r)|𝛘ᵧ> Dᵦᵧᵢⱼ

        :param density_matrices_ao: state (i==j) and transition (i != j) density matrices in
            in the AO basis summed over spin, density_matrices_ao[b,g,i,j] = Dᵦᵧᵢⱼ = Dᵅᵦᵧᵢⱼ+Dᵝᵦᵧᵢⱼ
        :type density_matrices_ao: Tensor of shape (Nbasis,Nbasis,Nstate,Nstate)

        :return nuclear_matrix: nuclear attraction operator in basis
            of electronic states, Vᵢⱼ
        :rtype nuclear_matrix: Tensor of shape (Nstate,Nstate)
        """
        nuclear_matrix = (
            # Vᵢⱼ
            OneElectronOperatorAO.apply(density_matrices_ao, self.mol, 'int1e_nuc') +
            # Vᵢⱼ(ecp), contribution from effective core potentials to external potential
            OneElectronOperatorAO.apply(density_matrices_ao, self.mol, 'ECPscalar')
        )

        return nuclear_matrix
