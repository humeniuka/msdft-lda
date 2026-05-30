# coding: utf-8
"""
kinetic energy density

    KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>
"""
import pyscf.gto

from torch import Tensor
import torch.nn

from mlmsdft.dft.operator1e import OneElectronOperatorAO


class KineticFunctionalAO(torch.nn.Module):
    def __init__(self, mol: pyscf.gto.Mole):
        """
        kinetic energy matrix

        Tᵢⱼ[D(r)] = <Ψᵢ|-1/2 ∑ₙ∇ₙ²|Ψⱼ>

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole
        """
        super().__init__()
        self.mol = mol

    def forward(self, density_matrices_ao: Tensor) -> Tensor:
        """
        compute the kinetic energy matrix using the AO integrals.

            Tᵢⱼ = ∑ᵦ ∑ᵧ <𝛘ᵦ|-1/2 ∇²|𝛘ᵧ> Dᵦᵧᵢⱼ

        :param density_matrices_ao: state (i==j) and transition (i != j) density matrices in
            in the AO basis summed over spin, density_matrices_ao[b,g,i,j] = Dᵦᵧᵢⱼ = Dᵅᵅᵦᵧᵢⱼ+Dᵝᵝᵦᵧᵢⱼ
        :type density_matrices_ao: Tensor of shape (Nbasis,Nbasis,Nstate,Nstate)

        :return nuclear_matrix: nuclear attraction operator in basis
            of electronic states, Vᵢⱼ
        :rtype nuclear_matrix: Tensor of shape (Nstate,Nstate)
        """
        return OneElectronOperatorAO.apply(density_matrices_ao, self.mol, 'int1e_kin')
