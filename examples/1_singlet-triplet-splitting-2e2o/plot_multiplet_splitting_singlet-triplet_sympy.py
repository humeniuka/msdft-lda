import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sympy import (
    Matrix, Rational, Symbol,
    eye, init_printing, lambdify, pprint, shape, simplify, sqrt, zeros
)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

# enable best pretty printing available
init_printing()

def spin_trace(spin_matrix: Matrix) -> Matrix:
    """ Trace out electron spins by summing diagonal of 2x2 blocks. """
    nrows, ncols = shape(spin_matrix)
    assert nrows == ncols
    nstate = nrows//2
    assert 2*nstate == nrows

    # spin trace of M
    matrix = eye(nstate)
    for i in range(0, nstate):
        for j in range(0, nstate):
            # D_ij = Dᵅᵅ_ij + Dᵝᵝ_ij
            matrix[i,j] = spin_matrix[2*i,2*j] + spin_matrix[2*i+1,2*j+1]
    return matrix

def spin_diagonal_matrix(spin_matrix: Matrix) -> Matrix:
    """
    Set the spin off-diagonals in all 2x2 subblocks to 0.

    (Dᵅᵅ  Dᵅᵝ)        (Dᵅᵅ   0)
    (        )  -->  (       )
    (Dᵝᵅ   Dᵝᵝ)        (0    Dᵝᵝ)
    """
    nrows, ncols = shape(spin_matrix)
    assert nrows == ncols
    nstate = nrows//2
    assert 2*nstate == nrows

    spin_diag_matrix = zeros(2*nstate, 2*nstate)
    for i in range(0, nstate):
        for j in range(0, nstate):
            # keep Dᵅᵅ_ij
            spin_diag_matrix[2*i,2*j] = spin_matrix[2*i,2*j]
            # keep Dᵝᵝ_ij
            spin_diag_matrix[2*i+1,2*j+1] = spin_matrix[2*i+1,2*j+1]
            # Dᵅᵝ_ij and Dᵝᵅ_ij are zero
    return spin_diag_matrix

# Two electrons (n=2) with spin in two orbitals (HOMO and LUMO).
# Considering only the states where each spatial orbital is occupied by
# exactly one electron, we can construct N=4 states
# - an open-shell singlet (S=0, Sz=0)
# - three components of a triplet (S=1, Sz=-1,0,+1)

# total charge density,
# ρ(r) = n/2 (|ϕH(r)|² + |ϕL(r)|²)
rho = Symbol('rho', positive=True)
# Difference between HOMO and LUMO charge densities,
# Δ(r) = n/2 ((|ϕH(r)|² - |ϕL(r)|²)
Delta = Symbol('Delta')

# After factoring out the total density the matrix density only
# depends on the ratio of the difference density to the total density.
xi = Symbol('xi')
# ξ(r) = Δ(r)/ρ(r)

# The spin matrix density consists of NxN block matrix of
# 2x2 spin blocks
#  (Dᵅᵅ  Dᵅᵝ)
#  (Dᵝᵅ   Dᵝᵝ)
spin_matrix_density = Matrix(
    [
        [Rational(1,2), 0,               0, xi/sqrt(2),   xi/2,          0,              0,           0],
        [0,             Rational(1,2),   0, 0,            0,             -xi/2,          -xi/sqrt(2), 0],

        [0,             0,               0, 0,            0,             0,              0,           0],
        [xi/sqrt(2),    0,               0, 1,            1/sqrt(2),     0,              0,           0],

        [xi/2,          0,               0, 1/sqrt(2),    Rational(1,2), 0,              0,           0],
        [0,             -xi/2,           0, 0,            0,             Rational(1,2),  1/sqrt(2),   0],

        [0,             -xi/sqrt(2),     0, 0,            0,             1/sqrt(2),      1,           0],
        [0,             0,               0, 0,            0,             0,              0,           0]
    ]
)

def dirac_exchange(rho):
    # exchange energy density (without the prefactor Cx * ρ(r)⁴ᐟ³)
    xed = -rho**Rational(4,3)
    return xed


def plot_exchange_energy_density_eigenvalues(xed_eigenvals, axis, loc="best"):
    # Plot eigenvalues of exchange energy density as a function of ξ(r) = Δ(r)/ρ(r)
    xi_ = np.linspace(-1.0, 1.0, 100)

    linestyles = ["-", "--"]

    axis.set_xlabel(r"$\xi(\mathbf{r})$", fontsize=18)
    axis.set_ylabel(r"$\epsilon_x(\mathbf{r})$ / $(C_x \rho^{4/3})$", fontsize=16)
    for i,(xed_eigenval, multiplicity) in enumerate(xed_eigenvals.items()):
        # Convert symbolic expression into lambda function
        xed_eigenval_func = lambdify(xi, xed_eigenval)
        # Evaluate lambda function on numpy array
        xed_eigenval_ = xed_eigenval_func(xi_)
        # If the eigenvalue is a constant independent of xi, only a single value is returned.
        # We need to convert it to an array with the same shape as xi.
        xed_eigenval_ = np.resize(np.asarray(xed_eigenval_), xi_.shape)

        axis.plot(xi_, xed_eigenval_, ls=linestyles[i], label=str(multiplicity))

    axis.legend(title="Multiplicity", fontsize="large", title_fontsize="large", loc=loc, frameon=False)


fig, axes = plt.subplots(
    nrows=2, ncols=2,
    #sharex=True,
    sharey=True,
    height_ratios=[0.5, 0.5],
    figsize=(6.5, 7.0)
)

LEGEND_OFFSET = 0.10

# --- spin-invariant Hamiltonian ---
print("Diagonalize spin matrix density")
U, D_eigvals = spin_matrix_density.diagonalize()

# Apply the local exchange functional to the eigenvalues of the spin matrix density.
nstate = 1+3
spin_XED_diag = zeros(2*nstate,2*nstate)
for i in range(0, 2*nstate):
    spin_XED_diag[i,i] = dirac_exchange(D_eigvals[i,i])

# Transform with eigenvectors to get xed(D)
spin_XED = simplify(U*spin_XED_diag*U**-1)
# matrix of exchange energy density
XED_inv = simplify(spin_trace(spin_XED))
print("Exchange energy matrix")
print(XED_inv)
print("Eigenvalues of exchange energy matrix")
xed_inv_eigenvals = XED_inv.eigenvals()
print(pprint(xed_inv_eigenvals))

plot_exchange_energy_density_eigenvalues(xed_inv_eigenvals, axes[0,0], loc="center")
axes[0,0].get_xaxis().set_visible(False)
axes[0,0].text(0.0, -0.75+LEGEND_OFFSET,
    "(a) invariant\n"
    r'$\begin{pmatrix} D^{\alpha\alpha} & D^{\alpha\beta} \\  D^{\beta\alpha} & D^{\beta\beta} \end{pmatrix}$',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize="x-large"
)
axes[0,0].tick_params(axis='y', which='major', labelsize=15)

# --- spin-polarized Hamiltonian ---
# remove off-diagonal parts of spin matrix (2x2)
spin_matrix_density = spin_diagonal_matrix(spin_matrix_density)

U, D_eigvals = spin_matrix_density.diagonalize()

# Apply the local exchange functional to the eigenvalues of the spin matrix density.
nstate = 1+3
spin_XED_diag = zeros(2*nstate,2*nstate)
for i in range(0, 2*nstate):
    spin_XED_diag[i,i] = dirac_exchange(D_eigvals[i,i])

# Transform with eigenvectors to get xed(D)
spin_XED = simplify(U*spin_XED_diag*U**-1)
# matrix of exchange energy density
XED_pol = simplify(spin_trace(spin_XED))
xed_pol_eigenvals = XED_pol.eigenvals()
print(pprint(xed_pol_eigenvals))

plot_exchange_energy_density_eigenvalues(xed_pol_eigenvals, axes[0,1])
axes[0,1].get_xaxis().set_visible(False)
axes[0,1].get_yaxis().set_visible(False)
axes[0,1].text(0.0, -0.75+LEGEND_OFFSET,
    "(b) polarized\n"
    r'$\begin{pmatrix} D^{\alpha\alpha} & 0 \\  0 & D^{\beta\beta} \end{pmatrix}$',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize="x-large"
)

# --- spin-unpolarized Hamiltonian ---
matrix_density = spin_trace(spin_matrix_density)
U, D_eigvals = matrix_density.diagonalize()
XED_diag = zeros(nstate,nstate)
for i in range(0, nstate):
    XED_diag[i,i] = 2.0 * dirac_exchange(D_eigvals[i,i]/2.0)

# Transform with eigenvectors to get xed(D)
XED_unpol = simplify(U*XED_diag*U**-1)
# eigenvalues of exchange energy matrix density
xed_unpol_eigenvals = XED_unpol.eigenvals()
print(pprint(xed_unpol_eigenvals))

plot_exchange_energy_density_eigenvalues(xed_unpol_eigenvals, axes[1,0])
axes[1,0].text(0.0, -0.75+LEGEND_OFFSET,
    "(c) unpolarized\n"
    r'$D = D^{\alpha\alpha} + D^{\beta\beta}$',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize="x-large"
)
axes[1,0].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
axes[1,0].set_xticklabels(["1", "-0.5", "0", "0.5", "1"], fontsize=15)
axes[1,0].tick_params(axis='y', which='major', labelsize=15)

# --- 50% spin-unpolarized + 50% spin-invariant Hamiltonian ---
# Average of spin-unpolarized and spin-invariant exchange energy density
XED_inv_mix = (XED_unpol + XED_inv)/2
# eigenvalues of exchange energy matrix density
xed_inv_mix_eigenvals = XED_inv_mix.eigenvals()
print(pprint(xed_inv_mix_eigenvals))

plot_exchange_energy_density_eigenvalues(xed_inv_mix_eigenvals, axes[1,1])
axes[1,1].get_yaxis().set_visible(False)
axes[1,1].text(0.0, -0.75+LEGEND_OFFSET,
    "(d) equal-weight\n"
    r"50\%invariant+50\%unpolarized",
    horizontalalignment='center',
    verticalalignment='center',
    fontsize="x-large"
)
axes[1,1].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
axes[1,1].set_xticklabels(["$1$", "$-0.5$", "$0$", "$0.5$", "$1$"], fontsize=15)

axes[0,0].set_ylim((-1.3, -0.7+0.155))
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0.0)

plt.savefig("singlet-tripplet_splitting_2x2.png", dpi=300)
plt.savefig("singlet-tripplet_splitting_2x2.svg")

plt.show()

# Observations: Singlet-triplet splitting, eigenvalues of the exchange energy density
# - (a) The off-diagonal spin blocks are necessary to ensure invariance under rotation in spin space and the correct
#   multiplicity of the eigenvalues (1 singlet, 3 triplets).
#       Ex(³(↑↑)) = Ex(³(↓↓)) = Ex(³(↑↓)) ≠ Ex(¹(↑↓))
#   (b) Spin-polarized calculations only take the spin-up
#   (Daa) and spin-down densities (Dbb) into account but neglect the off-diagonal spin blocks Dab and Dba leading
#   to the wrong multiplicities and breaking the degeneracy of the three components of the triplet.
#       Ex(³(↑↑)) = Ex(³(↓↓)) ≠ Ex(³(↑↓)) = Ex(¹(↑↓))
#   (c) Spin-unpolarized exchange energy density only depends on the total charge density D = Daa+Dbb and cannot
#   distinguish between the open-shell singlet and the triplet states, since all have the same charge density.
#   The spin-degeneracy of the triplet states is preserved at the cost of assigning the same energy to the open-shell
#   singlet as well.
#       Ex(³(↑↑)) = Ex(³(↓↓)) = Ex(³(↑↓)) = Ex(¹(↑↓))
#   (d) The spin-invariant calculation overestimates the exchange energy (in magnitude), while the spin-unpolarized
#   calculation underestimates it. The average of the invariant and unpolarized exchange energy still preserves spin symmetry,
#   and can distinguish different spin states (with the correct multiplicity), but the energy is much closer to the
#   polarized calculation.
# - While spin-unpolarized and spin-polarized exchange-correlation functionals reduce to ground state functionals
#   for a single state (N=1), the spin-invariant functionals does not. In fact a rotationally invariant ground state #   functional has to be a matrix functional of the 2 x 2 spin density matrix. The equivariance condition defining
#   a analytic matrix functional and the requirement of rotational invariance are one and the same thing
#   for a single state. If one is willing to accept matrix functionals, the reward is that the symmetry dilemma
#   disappears.