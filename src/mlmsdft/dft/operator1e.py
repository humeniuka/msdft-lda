# -*- coding: utf-8 -*-
import pyscf.gto

import torch
from torch import Tensor
from torch.autograd import Function


class OneElectronOperatorAO(Function):
    @staticmethod
    def forward(ctx, density_matrices_ao: Tensor, mol: pyscf.gto.Mole, intor: str) -> Tensor:
        """
        The matrix of one-electron operators in the basis of the electronic states
        is calculated by contracting the (transition) density matrices in the AO basis
        with the AO integrals of the operator:

          Oᵢⱼ = <Ψᵢ|∑ₙ oₙ|Ψⱼ>

              = ∑ᵦ ∑ᵧ <𝛘ᵦ|o(r)|𝛘ᵧ> Dᵦᵧᵢⱼ

        where i,j enumerate many-electron states, 𝛘ᵦ, 𝛘ᵧ are AOs and Dᵦᵧᵢⱼ
        is the (transition) density between the states i and j in the AO basis.

        :param density_matrices_ao: state (i==j) and transition (i != j) density matrices in
            in the AO basis summed over spin, density_matrices_ao[b,g,i,j] = Dᵦᵧᵢⱼ = Dᵅᵦᵧᵢⱼ+Dᵝᵦᵧᵢⱼ
        :type density_matrices_ao: Tensor of shape (Nbasis,Nbasis,Nstate,Nstate)

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole

        :param intor: Name of the 1-electron integrals, e.g. 'int1e_nuc' for
           the nuclear potential energy.
        :type intor: str

        :return matrix_elements: The matrix elements of the operator in the
           basis of the many-electron states in the subspace.
        :rtype matrix_elements: Tensor of shape (Nstate,Nstate)
        """
        # density matrices Dᵦᵧᵢⱼ in AO basis
        dm = density_matrices_ao
        if isinstance(mol, pyscf.pbc.gto.cell.Cell):
            # `mol` is a periodic crystal
            integrals_1e_ao = mol.pbc_intor(intor)
        else:
            # `mol` is a molecule
            integrals_1e_ao = mol.intor_symmetric(intor)
        # convert to torch tensors
        integrals_1e_ao = torch.from_numpy(integrals_1e_ao).to(
            dtype=dm.dtype, device=dm.device)

        # Integrals are needed for backpropagation.
        ctx.save_for_backward(integrals_1e_ao)

        matrix_elements = torch.einsum(
            'bg,bgij->ij',
            integrals_1e_ao,
            dm)

        return matrix_elements

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        """
        Compute gradients w/r/t inputs using chain rule

        :param grad_output: vector xᵢⱼ = ∂g/∂Vᵢⱼ, gradient with respect to outputs of V[D(r)]ᵢⱼ
        :type grad_output: Tensor of size (Nstate,Nstate)

        :return grad_input: ∂g/∂Dᵦᵧₘₙ gradient with respect to D[b,g,m,n], i.e. the inputs of V
        :rtype grad_input: Tensor of size (Nbasis,Nbasis,Nstate,Nstate)
        """
        integrals_1e_ao, = ctx.saved_tensors
        # chain rule
        # ∑ᵢ∑ⱼ ∂Vᵢⱼ/∂Dᵦᵧₘₙ xᵢⱼ = <𝛘ᵦ|o(r)|𝛘ᵧ> xₘₙ
        grad_input = torch.einsum('bg,mn->bgmn', integrals_1e_ao, grad_output)

        # There is no gradient w/r/t arguments which are not tensors, therefore None is returned.
        return grad_input, None, None
