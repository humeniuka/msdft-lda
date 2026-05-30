#!/usr/bin/env python
"""
Compute the energies of the lowest triplet (³P) and singlet states (¹D and ¹S)
of the carbon atom with full configuration interaction, exploting spin symmetry
but without using spatial symmetry.

The exact exchange-correlation energy density is computed from the pair-density
matrix and is plotted on a radial grid.
"""
import matplotlib.pyplot as plt
import numpy
import numpy.testing

from pyscf.data.nist import HARTREE2EV
import pyscf.fci
import pyscf.gto
import pyscf.scf

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

from mlmsdft.utils.xc import plot_xc_energy_density
from mlmsdft.dft.xc import lda_x_dirac, lda_c_chachiyo

plt.style.use('./latex.mplstyle')

#
# taken from https://github.com/pyscf/pyscf/blob/master/examples/scf/31-v_atom_rohf.py
#
# Spherical symmetry needs to be carefully treated in the atomic calculation.
# The default initial guess may break the spherical symmetry.  To preserve the
# spherical symmetry in the atomic calculation, it is often needed to tune the
# initial guess and SCF model.
#
# Construct the atomic initial guess from cation.
#
mol = pyscf.gto.Mole()
mol.verbose = 4
mol.atom = 'C'
mol.basis = 'cc-pvdz'
# Do not use spatial symmetry.
mol.symmetry = False
# The electronic configuration of neutral carbon is 1s²2s²2p²
# The dication C^{2+} has the closed-shell configuration 1s²2s², which is spherically symmetric.
mol.spin = 0
mol.charge = 2
mol.build()

rohf = pyscf.scf.ROHF(mol)
rohf.kernel()
rohf.analyze()

# Restore the neutral atom, the ground state is a ³P
# spin = Sz
mol.spin = 2
mol.charge = 0
mol.build()

# full CI for ³P, ¹D and ¹S
norb = rohf.mo_energy.size
nelec = (3,3)
mol.nelec = nelec
fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
# Multiplicity of ³P is 3 (2*L+1=3),
# multiplicity of ¹D is (2*L+1)=5 and multiplicity of ¹S is 1.
fci.nroots = 3+5+1
fci_energies, fcivecs = fci.kernel(nelec=nelec)
e = fci_energies
# spin multiplicities 2*S+1
multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
multiplicities = numpy.round(multiplicities, decimals=2)
for i in range(0, len(fcivecs)):
    print(
        'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
        (i, e[i], (e[i]-e[0])*HARTREE2EV, multiplicities[i])
    )
# Check that we got the expected states.
numpy.testing.assert_allclose(multiplicities, 3*[3.0] + 5*[1.0] + 1*[1.0])
state_labels = (
    [r"³P(1)", "³P(2)", "³P(3)"] +
    [r"¹D(1)","¹D(2)","¹D(3)","¹D(4)","¹D(5)"] +
    ["¹S"])

# Compute xc-hole
msmd = MultistateMatrixDensityFCI(
    mol, rohf, fci, fcivecs,
    compute_pair_density=True)

ncoord = 1000
r = numpy.linspace(0.0, 5.0, ncoord)
coords = numpy.zeros((ncoord,3))
coords[:,0] = r

# Only plot some spin states to avoid very cluttered plots
plot_multiplicities = [3]

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
    xc_linestyles = ["-."], # "--", ":"],
    spin_polarized = False,
    plot_multiplicities = plot_multiplicities
)
fig.set_size_inches(10, 5)

plt.savefig("carbon_fci_xc_energy_density.svg")
plt.savefig("carbon_fci_xc_energy_density.png", dpi=300)

plt.show()

# --- zoom in around minimum ---
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
    xc_linestyles = ["-."], # "--", ":"],
    spin_polarized = False,
    plot_multiplicities = plot_multiplicities
)
fig.set_size_inches(10, 5)

axes = fig.get_axes()
rmin, rmax = 0.075, 0.215
axes[0].set_xlim((rmin, rmax))
axes[0].set_ylim((-13.75, -12.2))

axes[1].set_xlim((rmin, rmax))
axes[1].set_ylim((0.0008, 0.021))

plt.savefig("carbon_fci_xc_energy_density_zoom_minimum.svg")
plt.savefig("carbon_fci_xc_energy_density_zoom_minimum.png", dpi=300)

plt.show()

# --- zoom in around valence region ---
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
    xc_linestyles = ["-."], # "--", ":"],
    spin_polarized = False,
    plot_multiplicities = plot_multiplicities
)
fig.set_size_inches(10, 5)

axes = fig.get_axes()
rmin, rmax = 0.4, 3.0
axes[0].set_xlim((rmin, rmax))
axes[0].set_ylim((-1.6, 0.0))

axes[1].set_xlim((rmin, rmax))
axes[1].set_ylim((0.0008, 0.021))

plt.savefig("carbon_fci_xc_energy_density_zoom_valence-region.svg")
plt.savefig("carbon_fci_xc_energy_density_zoom_valence-region.png", dpi=300)

plt.show()
