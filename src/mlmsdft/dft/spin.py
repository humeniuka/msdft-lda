#!/usr/bin/env python
# coding: utf-8
"""
Functions for handling electronic spin in the matrix density.
"""
from enum import Enum
import numpy

import torch
from torch import Size, Tensor


class SpinType(str, Enum):
    """
    Treatment      Variable     Rotationally invariant
    ---------------------------------------------------
    UNPOLARIZED    D=Dᵅ+Dᵝ             yes
    POLARIZED      Dᵅ, Dᵝ              no
    INVARIANT      (Dᵅᵅ Dᵅᵝ)           yes
                   (Dᵝᵅ  Dᵝᵝ)
    INVARIANT_MIX  D and (Dᵅᵅ Dᵅᵝ)     yes
                         (Dᵝᵅ  Dᵝᵝ)
    """
    # Hamiltonian depends only on the total charge density.
    UNPOLARIZED = "spin_unpolarized"
    # Hamiltonian depends on spin-up and spin-down densities separately.
    POLARIZED = "spin_polarized"
    # Hamiltonian depends on spin-up, spin-down and mixed spin densities.
    INVARIANT = "spin_invariant"
    # Hamiltonian depends both on total charge density and on spin matrix density.
    INVARIANT_MIX = "spin_invariant_mix"


def concat_spin_blocks(matrix_density: Tensor) -> Tensor:
    """
    stack the four NxN-dimensional spin blocks Dᵅᵅᵢⱼ(r),Dᵅᵝᵢⱼ(r),Dᵝᵅᵢⱼ(r)
    and Dᵝᵝᵢⱼ(r) along the matrix dimensions (the last two dimensions) in
    order to construct the (2N)x(2N)-dimensional spin matrix density:

                          (Dᵅᵅᵢⱼ Dᵅᵝᵢⱼ)
        Dˢᵗᵢⱼ(r)    --->  (           )
                          (Dᵝᵅᵢⱼ  Dᵝᵝᵢⱼ)

    The input matrix is reshaped as

        (2,2,...,N,N) --->  (...,2*N,2*N)

    The spin indices s,t=α,β and the state indices i,j=1,...,N are combined
    into multiindices is,jt=1,...,2*N

        (s,t,...,i,j) --->  (..., is, jt)

    :param matrix_density: matrix density with separate NxN spin blocks
        D[s,t,...,i,j]=Dˢᵗᵢⱼ(r[...]), where ... stands for any other grid
        coordinates
    :type matrix_density: Tensor of shape (2,2,...,N,N)

    :return spin_matrix_density: (2N)x(2N) spin block matrix
    :rtype: Tensor of shape (...,2*N,2*N)
    """
    if matrix_density.size()[:2] != Size([2,2]):
        raise ValueError(
            "The first 2 dimensions of the input tensor must index the spin blocks "
            "aa, ab, ba and bb."
        )
    Daa = matrix_density[0,0,...]
    Dab = matrix_density[0,1,...]
    Dba = matrix_density[1,0,...]
    Dbb = matrix_density[1,1,...]

    spin_matrix_density = torch.cat(
        (
            torch.cat((Daa, Dab), dim=-1),
            torch.cat((Dba, Dbb), dim=-1)
        ),
        dim=-2
    )
    return spin_matrix_density


def split_spin_blocks(spin_matrix_density: Tensor) -> Tensor:
    """
    `split_spin_blocks` is the inverse operation to `concat_spin_blocks`.

    It splits the matrix dimensions into the 4 spin blocks Dᵅᵅ,Dᵅᵝ,Dᵝᵅ,Dᵝᵝ
    so that each spin block can be indexed separately.

    :param spin_matrix_density: (2N)x(2N) spin block matrix
    :param spin_matrix_density: Tensor of shape (...,2*N,2*N)

    :return matrix_density: matrix density with separate NxN spin blocks
        D[s,t,...,i,j]=Dˢᵗᵢⱼ, where ... stands for any other dimensions.
    :rtype matrix_density: Tensor of shape (2,2,...,N,N)
    """
    # Check input dimensions
    matrix_dims = spin_matrix_density.size()[-2:]
    if not (
        # (2*N),2*N)
        (matrix_dims[0] == matrix_dims[1]) and
        (matrix_dims[0] % 2 == 0)
    ):
        raise ValueError(f"Matrix dimensions must be (2*N,2*N), got {matrix_dims}")

    spin_blocks = [
        # split each row into columns
        # (Dᵅᵅ Dᵅᵝ) -> (Dᵅᵅ,Dᵅᵝ) and (Dᵝᵅ Dᵝᵝ) -> (Dᵝᵅ,Dᵝᵝ)
        torch.chunk(row, chunks=2, dim=-1) for
            # split into rows  (Dᵅᵅ Dᵅᵝ) and (Dᵝᵅ Dᵝᵝ)
            row in torch.chunk(spin_matrix_density, chunks=2, dim=-2)
    ]
    Daa = spin_blocks[0][0]
    Dab = spin_blocks[0][1]
    Dba = spin_blocks[1][0]
    Dbb = spin_blocks[1][1]
    matrix_density = torch.stack(
        (
            torch.stack((Daa, Dab), dim=0),
            torch.stack((Dba, Dbb), dim=0)
        ),
        dim=0
    )
    return matrix_density


def spin_trace(X: Tensor) -> Tensor:
    """
    Sum the same-spin blocks

                     (Xᵅᵅᵢⱼ Xᵅᵝᵢⱼ)
        spin_trace ( (           ) ) = Xᵅᵅᵢⱼ + Xᵝᵝᵢⱼ
                     (Xᵝᵅᵢⱼ Xᵝᵝᵢⱼ )

    :param X: (2N)x(2N) spin block matrix
    :type X: Tensor of shape (...,2*N,2*N)

    :return spin_trace(X): sum of the same-spin blocks of X
    :rtype: Tensor of shape (...,N,N)
    """
    spin_trX = torch.einsum(
        'ss...->...', split_spin_blocks(X)
    )
    return spin_trX


def merge_multiplet_energies(energies, spin_multiplicities):
    """
    Keep only the energy of the first component of each degenerate spin multiplet

    :param energies: energies of states, energies of a spin multiplet
        are repeated according to the multiplicity
    :type energies: numpy array of float

    :param spin_multiplicites: spin multiplicities 2*S+1
    :type spin_multiplicities: numpy array of int, same length as `energies`

    :return energies, spin_multiplicities: numpy arrays with repeated elements
        belonging to the same multiplet removed

    Example
    -------
    >>> energies = numpy.array([1.0, 2.0, 2.0, 2.0])
    >>> spin_multiplicities = numpy.array([1,3,3,3])
    # The degenerate energies of the three triplet components are merged.
    >>> energies, spin_multiplicities = merge_multiplet_energies(energies, spin_multiplicities)
    >>> energies
    array([1., 2.])
    >>> spin_multiplicities
    array([1, 3])
    """
    energies_merged = []
    spin_multiplicities_merged = []
    i = 0
    while i < len(energies):
        m = spin_multiplicities[i]
        energies_merged.append(energies[i])
        spin_multiplicities_merged.append(m)
        # There should be m degenerate states belonging to this spin multiplet.
        numpy.testing.assert_allclose(energies[i:i+m] - energies[i], numpy.zeros(m), atol=1.0e-5)
        # Skip the next m-1 states and move to next multiplet.
        i += (m-1)+1
    return numpy.array(energies_merged), numpy.array(spin_multiplicities_merged)


def index_within_multiplicity(spin_multiplicities):
    """
    Counting of electronic states starts from 1 in each spin manifold.
    Given of a list of spin multiplicies, the index of each state within
    its spin multiplicity is returned.

    :param spin_multiplicites: spin multiplicities 2*S+1
    :type spin_multiplicities: array of int

    :return state_indices: 1-based state indices
    :rtype state_indices: array of int

    Example
    -------
    >>> spin_multiplicities = [1,3,1,1,3]
    >>> index_within_multiplicity(spin_multiplicities)
    [1, 1, 2, 3, 2]
    """
    state_indices = numpy.zeros_like(spin_multiplicities, dtype=int)
    for spin_multiplicity in numpy.unique(spin_multiplicities):
        counter = 1
        for i in range(0, len(spin_multiplicities)):
            if spin_multiplicities[i] == spin_multiplicity:
                state_indices[i] = counter
                counter += 1
    return state_indices
