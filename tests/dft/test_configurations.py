#!/usr/bin/env python
# coding: utf-8
import numpy
import numpy.testing
import unittest

from mlmsdft.dft.configurations import excitation_levels
from mlmsdft.dft.configurations import configurations_by_excitation_levels
from mlmsdft.dft.configurations import occupation_labels
from mlmsdft.dft.configurations import total_spin_matrix
from mlmsdft.dft.configurations import matrix_density_mo


class TestConfigurations(unittest.TestCase):
    def test_excitation_levels(self):
        # 0 electrons in 2 orbitals
        self.assertListEqual(
            excitation_levels(norb=2, nelec=0),
            # For 0 electrons, the HF determinant is 0, too.
            [0]
        )
        # 1 electron in 3 orbitals
        self.assertListEqual(
            excitation_levels(norb=3, nelec=1),
            [0,1,1]
        )
        # 2 spin-up electrons in 2 orbitals => there is only the HF determinant
        self.assertListEqual(
            excitation_levels(norb=2, nelec=2),
            [0]
        )
        # 2 spin-up electrons in 4 orbitals
        self.assertListEqual(
            excitation_levels(norb=4, nelec=2),
            [0, 1, 1, 1, 1, 2]
        )

    def test_configurations_by_excitation_levels(self):
        # 2 electrons (up and down) in 2 orbitals
        self.assertListEqual(
            configurations_by_excitation_levels(norb=2,nelec=2),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        )
        # same but specifying spin up and spin down separately
        self.assertListEqual(
            configurations_by_excitation_levels(norb=2,nelec=(1,1)),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        )
        # same but with max_level=2, since there are only two electrons
        # the highest excitation level is 2 anyway.
        self.assertListEqual(
            configurations_by_excitation_levels(norb=2,nelec=(1,1), max_level=2),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        )
        # HF and single excitations (skip doubly excited determinants)
        self.assertListEqual(
            configurations_by_excitation_levels(norb=2,nelec=(1,1), max_level=1),
            [(0, 0), (0, 1), (1, 0)]
        )
        # HF only (skip singly and doubly excited determinants)
        self.assertListEqual(
            configurations_by_excitation_levels(norb=2,nelec=(1,1), max_level=0),
            [(0, 0)]
        )

        # HF and singly excited states for 3 electrons in 4 orbitals
        self.assertListEqual(
            configurations_by_excitation_levels(norb=6,nelec=3, max_level=1),
            [
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (7, 0), (10, 0), (11, 0)
            ]
        )
        # same but specify number of up and down electrons explicitly
        self.assertListEqual(
            configurations_by_excitation_levels(norb=6,nelec=(2,1), max_level=1),
            [
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (7, 0), (10, 0), (11, 0)
            ]
        )

    def test_occupation_labels(self):
        # 2 electrons in 2 orbitals
        self.assertListEqual(
            occupation_labels(norb=2, nelec=2),
            ['2.', 'ab', 'ba', '.2']
        )
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1)),
            ['2.', 'ab', 'ba', '.2']
        )
        # HF, singly and doubly excited determinants
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1), max_level=2),
            ['2.', 'ab', 'ba', '.2']
        )
        # Only HF and singly excited determinants
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1), max_level=1),
            ['2.', 'ab', 'ba']
        )
        # Only HF state
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(1,1), max_level=0),
            ['2.']
        )
        # 2 electrons in 3 orbitals
        self.assertListEqual(
            occupation_labels(norb=3, nelec=2),
            ['2..', 'ab.', 'a.b', 'ba.', '.2.', '.ab', 'b.a', '.ba', '..2']
        )
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(1,1)),
            ['2..', 'ab.', 'a.b', 'ba.', '.2.', '.ab', 'b.a', '.ba', '..2']
        )
        # Only HF and singly excited determinants
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(1,1), max_level=1),
            ['2..', 'ab.', 'a.b', 'ba.', 'b.a']
        )
        # 2 spin-up electrons in 2 orbitals
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(2,0)),
            ['aa']
        )
        # 2 spin-down electrons in 2 orbitals
        self.assertListEqual(
            occupation_labels(norb=2, nelec=(0,2)),
            ['bb']
        )
        # Single spin-up electron in 3 orbitals
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(1,0)),
            ['a..', '.a.', '..a']
        )
        # 2 spin-up and 1 spin-down electrons in 3 orbitals
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(2,1)),
            ['2a.', 'a2.', 'aab', '2.a', 'aba', 'a.2', 'baa', '.2a', '.a2']
        )
        # Only singly excited determinants
        self.assertListEqual(
            occupation_labels(norb=3, nelec=(2,1), max_level=1),
            ['2a.', 'a2.', 'aab', '2.a', 'baa']
        )

    def test_total_spin_matrix(self):
        # 2 electrons in 2 orbitals
        s2 = total_spin_matrix(norb=2, nelec=2)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # same with max_level=2
        s2 = total_spin_matrix(norb=2, nelec=2, max_level=2)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # only HF and singly excited determinants -> 2 singlets, 1 triplet (Sz=0)
        s2 = total_spin_matrix(norb=2, nelec=2, max_level=1)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 2.])
        )
        # only HF determinant -> 1 singlet
        s2 = total_spin_matrix(norb=2, nelec=2, max_level=0)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.])
        )
        # 2 electrons in 2 orbitals -> 3 singlets, 1 triplet (Sz=0)
        s2 = total_spin_matrix(norb=2, nelec=(1,1))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # same with max_level=2
        s2 = total_spin_matrix(norb=2, nelec=(1,1), max_level=2)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 0., 2.])
        )
        # only HF and singly excited determinants -> 2 singlets, 1 triplet (Sz=0)
        s2 = total_spin_matrix(norb=2, nelec=(1,1), max_level=1)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0., 0., 2.])
        )
        # only HF determinant
        s2 = total_spin_matrix(norb=2, nelec=(1,1), max_level=0)
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.])
        )

        # 1 electron in 2 orbitals
        s2 = total_spin_matrix(norb=2, nelec=(1,0))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.75, 0.75])
        )
        # 2 spin-up electrons in 2 orbitals -> 1 triplet state
        s2 = total_spin_matrix(norb=2, nelec=(2,0))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([2.0])
        )
        # 2 spin-up electrons and 1 spin-down electron in 2 orbitals -> 2 doublet states
        s2 = total_spin_matrix(norb=2, nelec=(2,1))
        eigvals, _ = numpy.linalg.eigh(s2)
        numpy.testing.assert_allclose(
            eigvals, numpy.array([0.75, 0.75])
        )

        # Possible expectation values of S² for even numbers of electrons.
        even_num_electrons_s2 = []
        for s in [0,1,2,3]:
            s2 = s*(s+1)
            even_num_electrons_s2.append(s2)
        # Possible expectation values of S² for odd numbers of electrons.
        odd_num_electrons_s2 = []
        for s in [0.5, 1.5, 2.5]:
            s2 = s*(s+1)
            odd_num_electrons_s2.append(s2)

        # Check that eigenvalues of S² are in the expected ranges.
        for norb in range(2,6):
            for neleca in range(1, norb):
                for nelecb in range(0, norb-neleca):
                    if abs(neleca-nelecb) == 0:
                        max_levels = [0,1,2,3,4]
                    else:
                        max_levels = [0,numpy.inf]
                    for max_level in max_levels:
                        with self.subTest(norb=norb, neleca=neleca, nelecb=nelecb, max_level=max_level):
                            s2 = total_spin_matrix(norb=norb, nelec=(neleca,nelecb), max_level=max_level)
                            eigvals, _ = numpy.linalg.eigh(s2)
                            if (neleca + nelecb) % 2 == 0:
                                # even number of electrons
                                for s2_ in eigvals:
                                    self.assertIn(numpy.round(s2_, 2), even_num_electrons_s2)
                            else:
                                # odd number of electrons
                                for s2_ in eigvals:
                                    self.assertIn(numpy.round(s2_, 2), odd_num_electrons_s2)

    def test_matrix_density_mo(self):
        """
        Check that state spin densities integrate to the number of electrons and
        transition densities integrate to zero.
        """
        for norb in range(2,6):
            for neleca in range(1, norb):
                for nelecb in range(0, norb-neleca):
                    for max_level in [0,1,2,3,numpy.inf]:
                        with self.subTest(
                            norb=norb, neleca=neleca, nelecb=nelecb, max_level=max_level
                        ):
                            D_mo = matrix_density_mo(norb=norb, nelec=(neleca,nelecb), max_level=max_level)
                            # Check that
                            # ∫ Dₛ(r)_{I,J} dr = ∑_{p} Dmo[s,p,p,I,J] = nelec[s] δ_{I,J}
                            nelec = (neleca, nelecb)
                            nspin = D_mo.shape[0]
                            nstate = D_mo.shape[-1]
                            for s in range(0, nspin):
                                numpy.testing.assert_allclose(
                                    numpy.einsum('ppIJ->IJ', D_mo[s,...]), nelec[s] * numpy.eye(nstate)
                                )


if __name__ == "__main__":
    unittest.main()
