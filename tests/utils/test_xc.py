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

from mlmsdft.utils.xc import plot_xc_energy_density
from mlmsdft.dft.xc import lda_x_dirac, lda_c_chachiyo


class TestUtilsXC(unittest.TestCase):
    def test_plot_xc_energy_density_helium(self):
        """
        Check plotting of xc energy density using matplotlib.
        """
        mol = pyscf.gto.Mole()
        mol.verbose = 0
        mol.atom = 'He'
        mol.basis = '6-31g'
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

        ncoord = 20
        r = numpy.linspace(0.0, 5.0, ncoord)
        coords = numpy.zeros((ncoord,3))
        coords[:,0] = r

        # --- plot along x-axis ---
        fig = plot_xc_energy_density(
            msmd,
            state_labels,
            multiplicities,
            coords,
            r,
            xc_functionals = [
                (lda_x_dirac, lda_c_chachiyo),
            ],
            xc_names = [
                ("Dirac", "Chachiyo"),
            ],
            xc_linestyles = ["-.", "--", ":"],
            spin_polarized = False
        )

        plt.close(fig)
