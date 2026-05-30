#!/usr/bin/env python
# coding: utf-8
import numpy
import unittest

import torch
from torch import Size
import torch.testing

from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import concat_spin_blocks, split_spin_blocks
from mlmsdft.dft.spin import spin_trace
from mlmsdft.dft.spin import merge_multiplet_energies
from mlmsdft.dft.spin import index_within_multiplicity


class TestSpinTypes(unittest.TestCase):
    def test_concat_spin_blocks(self):
        # spin blocks of 3x3 matrix density.
        nstate = 3
        D = torch.zeros((2,2,nstate,nstate))
        unsymmetric_matrix = torch.tensor([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ])
        # aa
        D[0,0,...] = unsymmetric_matrix + 10.0
        # ab
        D[0,1,...] = unsymmetric_matrix + 20.0
        # ba
        D[1,0,...] = unsymmetric_matrix + 30.0
        # bb
        D[1,1,...] = unsymmetric_matrix + 40.0

        spin_D = concat_spin_blocks(D)
        # expected (2N)x(2N) spin matrix
        spin_D_ref = torch.tensor([
            #     Daa             Dab
            [11., 12., 13.,  21., 22., 23.],
            [14., 15., 16.,  24., 25., 26.],
            [17., 18., 19.,  27., 28., 29.],
            #     Dba             Dbb
            [31., 32., 33.,  41., 42., 43.],
            [34., 35., 36.,  44., 45., 46.],
            [37., 38., 39.,  47., 48., 49.],
        ])
        torch.testing.assert_close(spin_D, spin_D_ref)

        # Check that `split_spin_blocks` reverts the effect of `concat_spin_blocks`
        D_test = split_spin_blocks(spin_D)
        torch.testing.assert_close(D_test, D)

    def test_split_spin_blocks(self):
        """
        Check that `split_spin_blocks` reverts the effect of `concat_spin_blocks`
        for different extra dimensions between the spin and the matrix dimensions.
        """
        nstate = 3
        for extra_dims in [(), (1,), (4,), (5,6)]:
            #      spin dims.    extra dims.        matrix dimensions
            size = Size((2,2)) + Size(extra_dims) + Size((nstate,nstate))
            D_ref = torch.randn(size, dtype=torch.double)
            spin_D = concat_spin_blocks(D_ref)
            D = split_spin_blocks(spin_D)
            torch.testing.assert_close(D, D_ref)

    def test_spin_trace(self):
        nstate = 3
        for extra_dims in [(), (1,), (4,), (5,6)]:
            #      spin dims.    extra dims.        matrix dimensions
            size = Size((2,2)) + Size(extra_dims) + Size((nstate,nstate))
            D = torch.randn(size, dtype=torch.double)
            spin_D = concat_spin_blocks(D)
            # compute spin trace from expanded (2N)*(2N) spin matrix density
            spin_trD = spin_trace(spin_D)
            # Sum Dᵅᵅ+Dᵝᵝ directly
            spin_trD_ref = D[0,0,...] + D[1,1,...]
            torch.testing.assert_close(spin_trD, spin_trD_ref)

    def test_concat_spin_blocks_exception(self):
        """
        Verify that an exception is raised if the input to
        `concat_spin_blocks` lacks the spin dimensions
        """
        with self.assertRaises(ValueError) as err:
            D_wrong = torch.ones(1,2,3,3)
            concat_spin_blocks(D_wrong)
        # Check error message
        self.assertIn(
            'The first 2 dimensions of the input tensor must index the spin blocks',
            str(err.exception)
        )

    def test_split_spin_blocks_exception(self):
        """
        Verify that an exception is raised, if the matrix dimensions
        of the input cannot be split into a 2x2 block matrix.
        """
        with self.assertRaises(ValueError) as err:
            spin_D_wrong = torch.ones(1,3,3)
            split_spin_blocks(spin_D_wrong)
        # Check error message
        self.assertIn(
            'Matrix dimensions must be (2*N,2*N)',
            str(err.exception)
        )

    def test_spin_type_enum(self):
        # Not a useful test
        self.assertEqual(len(SpinType), 4)

    def test_merge_multiplet_energies(self):
        # Energies of triplet are repeated
        energies = numpy.array([1.0, 2.0, 2.0, 2.0])
        spin_multiplicities = numpy.array([1,3,3,3])
        # The degenerate energies of the three triplet components are merged.
        energies, spin_multiplicities = merge_multiplet_energies(energies, spin_multiplicities)
        numpy.testing.assert_allclose(energies, numpy.array([1.0, 2.0]))
        numpy.testing.assert_allclose(spin_multiplicities, numpy.array([1, 3]))

    def test_index_within_multiplicity(self):
        spin_multiplicities = [1,3,1,1,3]
        state_indices = index_within_multiplicity(spin_multiplicities)
        numpy.testing.assert_allclose(state_indices, numpy.array([1, 1, 2, 3, 2]))


if __name__ == "__main__":
    unittest.main()
