# -*- coding: utf-8 -*-
import numpy

import pyscf.gto
import pyscf.scf

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class _ExactExchangeFunctionalAO(Function):
    """
    The multistate analogue of exact Hartree-Fock exchange (EXX) is defined as:

        K[D(r,r')]ᵢⱼ = -1/2 ∑ₖ ∫∫' Dᵢₖ(r,r') Dₖⱼ(r',r)/|r-r'|

    where Dᵢⱼ(r,r') is the density field, the diagonal part (r=r') of which is
    the electronic density of the state Ψᵢ, Dᵢᵢ(r,r) = ρᵢ(r),
    or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r,r).

    The density field is represented in a basis of atomic orbitals as

        Dᵢⱼ(r,r') = ∑ᵣ ∑ₛ Dᵣₛ,ᵢⱼ 𝛘ᵣ(r) 𝛘ₛ(r')
    """
    @staticmethod
    def forward(ctx, density_matrices_ao: Tensor, mol: pyscf.gto.Mole) -> Tensor:
        """
        Compute the exact exchange-like part of the electron-electronc repulsion
        using the AO representation of the density field,

            K[D(r,r')]ᵢⱼ = -1/2 ∑ₖ ∑ᵣ ∑ₛ Dᵣₛ,ᵢₖ ∑ₜ ∑ᵤ (ru|st) Dₜᵤ,ₖⱼ

        where

            (ru|st) = ∫∫' 𝛘ᵣ(r) 𝛘ᵤ(r) 1/|r-r'| 𝛘ₛ(r') 𝛘ₜ(r')

        are the two-electron integrals between atomic orbital in chemists' notation.

        :param density_matrices_ao: state (i==j) and transition (i != j) density matrices in
            in the AO basis, density_matrices_ao[r,s,i,j] = Dᵣₛ,ᵢⱼ
        :type density_matrices_ao: Tensor of shape (Nbasis,Nbasis,Nstate,Nstate)

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole

        :return exact_exchange_matrix: The exact exchange-like matrix Kᵢⱼ in the subspace
           of the electronic states i,j=1,...,nstate
        :rtype exact_exchange_matrix: Tensor of shape (Nstate,Nstate)
        """
        # Convert matrix density to a list of numpy AO density matrices
        # [D00, D01, D02, ...]
        nbasis, nbasis, nstate, nstate = density_matrices_ao.size()
        dms = []
        for i in range(0, nstate):
            for j in range(0, nstate):
                dms.append(density_matrices_ao[:,:,i,j].cpu().numpy())

        # Compute the exact exchange potentials for all (transition) densities in the
        # matrix density using pyscf (K-build).
        #
        #    Vk_{r,s}[Dᵢⱼ(r,r')] = ∑_{t,u} (ru|st) D_{t,u,i,j}
        #
        # where (ru|st) are the electron repulsion integrals for the basis functions
        # r,s,t,u and i,j are state labels.
        vks = pyscf.scf.jk.get_jk(
            mol, dms,
            # 'rust,tu->rs' is equivalent to 'ruts,tu->rs' because of the symmetry (ru|st) = (ru|ts)
            # of 2e integrals. Since `get_jk(...)` only understands the indices i,j,k,l, we have to
            # rename r->i, u->j, t->k, s->l so that the Einstein summation string becomes 'ijkl,kj->il'.
            len(dms)*['ijkl,kj->il']
        )

        # Convert non-local exchange potentials Vk[D(r,r')] back to pytorch tensor.
        exchange_potentials = numpy.zeros((nbasis, nbasis, nstate, nstate))
        # Iterate over list of potentials and put them
        # back into matrix form.
        vks = iter(vks)
        for i in range(0, nstate):
            for j in range(0, nstate):
                vk = next(vks)
                exchange_potentials[:,:,i,j] = vk

        # exchange_potentials_ao[r,s,i,j] = Vkᵣₛ[Dᵢⱼ(r,r')] = ∑ₜ ∑ᵤ (ru|st) Dₜᵤ,ₖⱼ
        exchange_potentials_ao = torch.from_numpy(exchange_potentials).to(
            dtype=density_matrices_ao.dtype, device=density_matrices_ao.device)

        # Exact exchange potentials are needed for backpropagation.
        ctx.save_for_backward(exchange_potentials_ao)

        # Compute K[D(r,r')]ᵢⱼ = -1/2 ∑ₖ ∑ᵣ ∑ₛ Dᵣₛ,ᵢₖ Vkᵣₛ[Dₖⱼ]
        exact_exchange_matrix = -0.5 * torch.einsum('rsik,rskj->ij', density_matrices_ao, exchange_potentials_ao)

        return exact_exchange_matrix

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor) -> Tensor:
        """
        Compute gradients w/r/t inputs using chain rule

        :param grad_output: vector xᵢⱼ = ∂g/∂Kᵢⱼ, gradient with respect to outputs of K[D(r,r')]ᵢⱼ
        :type grad_output: Tensor of size (Nstate,Nstate)

        :return grad_input: ∂g/∂Dᵣₛ,ₘₙ gradient with respect to D[r,s,m,n], i.e. the inputs of K
        :rtype grad_input: Tensor of size (Nbasis,Nbasis,Nstate,Nstate)
        """
        exchange_potentials_ao, = ctx.saved_tensors
        # chain rule
        # ∑ᵢ∑ⱼ ∂Kᵢⱼ/∂Dᵣₛ,ₘₙ xᵢⱼ = -1/2 ∑ⱼ (xₘⱼ Vkᵣₛₙⱼ + Vkᵣₛⱼₘ xⱼₙ)
        grad_input = -0.5 * (
            torch.einsum('mj,rsnj->rsmn', grad_output, exchange_potentials_ao) +
            torch.einsum('rsjm,jn->rsmn', exchange_potentials_ao, grad_output)
        )
        # There is no gradient w/r/t the second argument `mol`, therefore None is returned.
        return grad_input, None


class ExactExchangeFunctionalAO(torch.nn.Module):
    """ see doc-string for _ExactExchangeFunctionalAO """
    def __init__(self, mol: pyscf.gto.Mole):
        super().__init__()
        if isinstance(mol, pyscf.pbc.gto.cell.Cell):
            raise NotImplementedError("Exact exchange not implemented for periodic cells")
        self.mol = mol
    def forward(self, density_matrices_ao: Tensor) -> Tensor:
        return _ExactExchangeFunctionalAO.apply(density_matrices_ao, self.mol)
