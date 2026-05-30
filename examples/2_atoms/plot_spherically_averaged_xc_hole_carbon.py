#!/usr/bin/env python
"""
Compute the energies of the lowest triplet (³P) and singlet states (¹D and ¹S)
of the carbon atom with full configuration interaction, exploting spin symmetry
but without using spatial symmetry.

The spherically averaged exchange-correlation hole around some point r,

is defined as

    H^{xc}ᵢⱼ(r,|u|) = 1/(4 π) ∫ dΩ H^{xc}ᵢⱼ(r,r+|u|*e(Ω))

where e(Ω) is a unit vector in the direction Ω, such that u = r'-r = |u|*e(Ω).

H^{xc}ᵢⱼ(r,|u|) is plotted as a function of the distance |u| from r
(similarly to figures in [Becke/Roussel1989]).

The exact spherically-average exchange-correlation hole is compared with

    1) the Taylor expansion of the exchange hole around u=0
    2) the exchange-hole of the homogeneous electron grad (HEG)
    3) the Becke-Roussel exchange hole

References
----------
[Becke1983] Becke, A. D.
    "Hartree-Fock exchange energy of an inhomogeneous electron gas."
    International journal of quantum chemistry 23.6 (1983): 1915-1922.
    doi:10.1002/qua.560230605
[Becke/Roussel1989] A. Becke, A, M. Roussel.
    "Exchange holes in inhomogeneous systems: A coordinate-space model."
    Phys. Rev. A, 39(8), 3761-3767.
    doi:10.1103/PhysRevA.39.3761
"""
import matplotlib.pyplot as plt
import numpy
from pyscf.data.nist import HARTREE2EV
import pyscf.fci
import pyscf.gto
import pyscf.scf

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

from mlmsdft.utils.xc_hole import plot_spherically_averaged_xc_holes

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

# Which hole approximations should be shown in the plot?
#plot_methods = ["Taylor", "HEG", "Becke-Roussel"]
plot_methods = ["Taylor", "HEG"]

# Which spin states should be included in the plot (None means to include all)
plot_multiplicities = [3]

# --- Reference point is at the nucleus ---
center_r = numpy.array([0.0, 0.0, 0.0])

# Plot xc-hole as a function of the distance from the electron.
distances_u = numpy.linspace(1.0e-6, 2.0, 2000)
# The Taylor expansion is only accurate where u is small, for large u it diverges.
distances_u_small = numpy.linspace(1.0e-6, 0.026, 500)

# Compute and plot xc-hole
fig = plot_spherically_averaged_xc_holes(
    msmd,
    state_labels,
    multiplicities,
    center_r,
    distances_u,
    distances_u_small,
    plot_methods = plot_methods,
    plot_multiplicities = plot_multiplicities,
    xlim = (0.001, 0.5)
)
fig.set_size_inches(10, 5)

plt.savefig("carbon_spherical_averaged_xc_hole_nucleus.svg")
plt.savefig("carbon_spherical_averaged_xc_hole_nucleus.png", dpi=300)

plt.show()

# --- Reference point is shifted along the z-axis ---
center_r = numpy.array([0.0, 0.0, 0.2])

# Plot xc-hole as a function of the distance from the electron.
distances_u = numpy.linspace(1.0e-6, 2.0, 2000)
# The Taylor expansion is only accurate where u is small, for large u it diverges.
distances_u_small = numpy.linspace(1.0e-6, 0.026, 500)

# Compute and plot xc-hole
fig = plot_spherically_averaged_xc_holes(
    msmd,
    state_labels,
    multiplicities,
    center_r,
    distances_u,
    distances_u_small,
    plot_methods = plot_methods,
    plot_multiplicities = plot_multiplicities,
    xlim = (0.001, 1.2)
)
fig.set_size_inches(10, 5)

plt.savefig("carbon_spherical_averaged_xc_hole_on-z-axis.svg")
plt.savefig("carbon_spherical_averaged_xc_hole_on-z-axis.png", dpi=300)

plt.show()
