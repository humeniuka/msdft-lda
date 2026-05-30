# -*- coding: utf-8 -*-
import numpy
import torch

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


def evaluate_fci_matrix_density(
    msmd: MultistateMatrixDensityFCI, coords: numpy.ndarray
):
    """
    The packages `msdft` and `mlmsdft` store the matrix densities in different
    array layouts. This functions converts the layout of `msdft` to the
    layout of `mlmsdft`. Also `msdft` works with numpy arrays and `mlmsdft` with
    torch tensors.
    """
    spin_D, grad_spin_D, lapl_spin_D = msmd.evaluate(coords)
    # move coordinate axis to the first position
    # ... for spin densities
    # (spin,bra,ket,coordinate) -> (spin,coordinate,bra,ket)
    spin_D = numpy.transpose(spin_D, axes=[0,3,1,2])
    # (spin,bra,ket,gradient,coordinate) -> (spin,coordinate,gradient,bra,ket)
    grad_spin_D = numpy.transpose(grad_spin_D, axes=[0,4,3,1,2])
    # (spin,bra,ket,coordinate) -> (spin,coordinate,bra,ket)
    lapl_spin_D = numpy.transpose(lapl_spin_D, axes=[0,3,1,2])

    # convert numpy arrays to torch tensors
    spin_D = torch.from_numpy(spin_D)
    grad_spin_D = torch.from_numpy(grad_spin_D)
    lapl_spin_D = torch.from_numpy(lapl_spin_D)

    return spin_D, grad_spin_D, lapl_spin_D
