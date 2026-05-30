#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy
import numpy.testing
import pyscf.fci
import pyscf.gto
import pyscf.scf
import unittest

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

from mlmsdft.utils.xc_hole import plot_spherically_averaged_xc_holes

class TestUtilsXCHole(unittest.TestCase):
    def test_plot_spherically_averaged_xc_holes_helium(self):
        """
        Check plotting of xc-hole using matplotlib.
        """
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'He'
        mol.basis = 'cc-pvdz'
        # Do not use spatial symmetry.
        mol.symmetry = False
        # The electronic configuration of helium is 1s² which is spherically symmetric
        mol.spin = 0
        mol.charge = 0
        mol.build()

        rohf = pyscf.scf.ROHF(mol)
        rohf.kernel()

        # full CI for lowest two singlet states, 1¹S and 2¹S, and triplet ³S
        norb = rohf.mo_energy.size
        nelec = (1,1)
        fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
        # Multiplicity of ¹S is 1, of ³S (Sz=0) is also 1.
        fci.nroots = 2*1+1
        _, fcivecs = fci.kernel(nelec=nelec)
        # spin multiplicities 2*S+1
        multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
        multiplicities = numpy.round(multiplicities, decimals=2)
        # Check that we got the expected states.
        numpy.testing.assert_allclose(multiplicities, [1.0, 3.0, 1.0])
        state_labels = [r"1¹S", r"1³S", r"2¹S"]

        # Compute exact FCI matrix density and the pair density
        msmd = MultistateMatrixDensityFCI(
            mol, rohf, fci, fcivecs,
            compute_pair_density=True)

        # First electron is put somewhat close to the origin r=0
        center_r = numpy.array([0.0, 0.0, 2.0])

        # xc-hole as a function of the distance from the electron.
        distances_u = numpy.linspace(1.0e-6, 5.0, 500)
        # The Taylor expansion is only accurate where u is small, for large u it diverges.
        distances_u_small = numpy.linspace(1.0e-6, 2.5, 500)

        # Compute and plot xc-hole
        fig = plot_spherically_averaged_xc_holes(
            msmd,
            state_labels,
            multiplicities,
            center_r,
            distances_u,
            distances_u_small,
        )
        plt.close(fig)
