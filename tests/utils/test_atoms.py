#!/usr/bin/env python
# coding: utf-8
import numpy
import numpy.testing
import os.path
import unittest
import tempfile

from mlmsdft.dft.pure import LDA
from mlmsdft.dft.spin import SpinType
from mlmsdft.utils.atoms import unique_counts
from mlmsdft.utils.atoms import group_atomic_energy_levels
from mlmsdft.utils.atoms import Helium
from mlmsdft.utils.atoms import atomic_benchmark_calculations
from mlmsdft.utils.atoms import atomic_pivot_table


class TestUtilsAtoms(unittest.TestCase):
    def test_unique_counts(self):
        array = numpy.array([1.0, 1.0, 3.0, 2.0, 2.0, 4.0])
        unique_ref = numpy.array([1.0, 2.0, 3.0, 4.0])
        counts_ref = numpy.array([2, 2, 1, 1])
        unique, counts = unique_counts(array)
        numpy.testing.assert_equal(unique, unique_ref)
        numpy.testing.assert_equal(counts, counts_ref)

    def test_group_atomic_energy_levels(self):
        # Find the terms for the lowest 3 levels of the carbon atom
        # 1³P, 1¹D, 1¹S
        energies = numpy.array([
            # ³P
            -38.051325525492494, -38.051325525492494, -38.051325525492494,
            # ¹D
            -37.95844472198104, -37.95844472198103, -37.95844472198102,
            -37.95844472198101, -37.95844472198099,
            # ¹S
            -37.9352121273941
        ])
        spin_multiplicities = numpy.array([3]*3 + [1]*5 + [1]*1)

        (
            energies, term_symbols,
            level_spin_multiplicities, level_spatial_multiplicities,
            state_indices
        ) = group_atomic_energy_levels(energies, spin_multiplicities)

        numpy.testing.assert_allclose(
            energies,
            numpy.array([-38.051325525492494, -37.95844472198104, -37.9352121273941])
        )
        self.assertEqual(term_symbols.tolist(), [r"1³P", r"1¹D", r"1¹S"])
        numpy.testing.assert_equal(level_spin_multiplicities, [3,1,1])
        numpy.testing.assert_equal(level_spatial_multiplicities, [3,5,1])
        numpy.testing.assert_equal(state_indices, [1,1,1])

    def test_atomic_benchmark_calculations_spin_unpolarized(self):
        """
        Check that we can create a pivot table from an atomic calculation.
        The results are not checked, only that the code does not crash.
        """
        atoms = [Helium(basis="cc-pvdz")]
        df = atomic_benchmark_calculations(
            atoms,
            xc_functionals_list=[("LDA", LDA)],
            spin_types=[SpinType.UNPOLARIZED]
        )
        pivot_table = atomic_pivot_table(df)
        print(pivot_table)
        with tempfile.TemporaryDirectory() as tmpdir:
            pivot_table.to_latex(os.path.join(tmpdir, "atomic_calculations.latex"))
            pivot_table.to_pickle(os.path.join(tmpdir, "atomic_calculations.pkl"))

    def test_atomic_benchmark_calculations_spin_polarized(self):
        """
        Check that we can create a pivot table from a spin-polarized atomic calculation.
        The results are not checked, only that the code does not crash.
        """
        atoms = [Helium(basis="aug-cc-pvqz")]
        df = atomic_benchmark_calculations(
            atoms,
            xc_functionals_list=[("LDA", LDA)],
            spin_types=[SpinType.POLARIZED]
        )
        pivot_table = atomic_pivot_table(df)
        print(pivot_table)
        with tempfile.TemporaryDirectory() as tmpdir:
            pivot_table.to_latex(os.path.join(tmpdir, "atomic_calculations.latex"))
            pivot_table.to_pickle(os.path.join(tmpdir, "atomic_calculations.pkl"))

    def test_atomic_benchmark_calculations_spin_invariant(self):
        """
        Check that we can create a pivot table from a spin-invariant atomic calculation.
        The results are not checked, only that the code does not crash.
        """
        atoms = [Helium(basis="cc-pvdz")]
        df = atomic_benchmark_calculations(
            atoms,
            xc_functionals_list=[("LDA", LDA)],
            spin_types=[SpinType.INVARIANT, SpinType.INVARIANT_MIX]
        )
        pivot_table = atomic_pivot_table(df)
        print(pivot_table)
        with tempfile.TemporaryDirectory() as tmpdir:
            pivot_table.to_latex(os.path.join(tmpdir, "atomic_calculations.latex"))
            pivot_table.to_pickle(os.path.join(tmpdir, "atomic_calculations.pkl"))
