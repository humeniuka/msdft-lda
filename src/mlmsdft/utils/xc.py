# -*- coding: utf-8 -*-
"""
Functions for plotting the exact exchange-correlation energy density and comparing
it with the energy density from LDA (Dirac exchange + Chachiyo correlation)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

from mlmsdft.dft.xc import lda_x_dirac, lda_c_chachiyo
from mlmsdft.utils.fci import evaluate_fci_matrix_density


def plot_xc_energy_density(
    msmd: MultistateMatrixDensityFCI,
    state_labels,
    multiplicities,
    coords: numpy.ndarray,
    r: numpy.ndarray,
    xc_functionals = [
        (lda_x_dirac, lda_c_chachiyo),
    ],
    xc_names = [
        ("Dirac", "Chachiyo"),
    ],
    xc_linestyles = ["-.", "--", ":"],
    spin_polarized = False,
    plot_multiplicities = None
):
    """
    Plot exact exchange-correlation energy density computed from the pair density matrix
    of a FCI calculation and compare it with different approximate functionals.

    :param msmd: multistate matrix density from a full CI calculation
        with pair density, from which the exact exchange-correlation
        energy density is calculated
    :type msmd: MultistateMatrixDensityFCI

    :param state_labels: list of strings for each of the electronic states
        in matrix density that will be used to label the diagonal and
        off-diagonal elements of the xc energy density matrix
    :type state_labels: list of str
        e.g. [r"1¹S", r"1³S", r"2¹S"] for the lowest 3 states of the He atom

    :param multiplicities: list of spin multiplicity for each state.
        Only states with Sz=0 are calculated. The spin multiplicities are used
        to distinguish different spin states and only plot the off-diagonal
        elements of the xc energy density between states with the same multiplicity,
        since those between different ones are always zero.
    :type multiplicities: list of int
        e.g. [1,3,1] for the lowest 3 states of the He atom

    :param coords: 3D coordinates (in Bohr) where xc energy density is evaluated
    :type coords: numpy.ndarray of shape (npts,3)

    :param r: coordinates for plotting (x-axis) (in Bohr)
    :type r: numpy.ndarray of shape (npts)

    :param xc_functionals: exchange and correlation functionals
    :type xc_functionals: list of `nfunc` tuples of Callables

    :param xc_names: names of exchange and correlation functionals
    :type xc_names: list of `nfunc` tuples of strings

    :param xc_linestyles: linestyles for plotting different approximate functionals
    :type xc_linestyles: list of `nfunc` strings

    :param spin_polarized: Whether to compute the exchange-energy density from
        the total matrix density or spin the spin-up and spin-down matrix densities
    :type spin_polarized: bool

    :param plot_multiplicities: If different from None, only states with the selected
        multiplicities are plotted
    :type plot_multiplicities: list of int (e.g. [1,3]) or None

    :return fig: plot with xc-energy densities
    :rtype fig: matplotlib Figure
    """
    assert r.shape[0] == coords.shape[0]
    assert len(xc_functionals) == len(xc_names) == len(xc_linestyles)

    # excat xc-energy density from pair density matrix
    xced_exact = msmd.exchange_correlation_energy_density(coords)

    # D(r), ∇D(r) and ∇²D at the reference point r
    spin_D, grad_spin_D, lapl_spin_D = evaluate_fci_matrix_density(msmd, coords)
    # sum over spin
    D, grad_D, lapl_D = spin_D.sum(axis=0), grad_spin_D.sum(axis=0), lapl_spin_D.sum(axis=0)

    xced_approximations = []
    for (exchange_functional, correlation_functional) in xc_functionals:
        # --- Evaluate approximate functionals for exchange ---
        # The exchange interaction exists only between electrons with the same spin.
        if not spin_polarized:
            # The spin-restricted/unpolarized version of the exchange functional assumes
            # that the spin-up and spin-down parts of the matrix density are the same,
            # D(r) = Dᵅ(r)+Dᵝ(r) = 2 Dᵅ(r)
            # Therefore, one can reconstruct the spin matrix density from the charge matrix density,
            #   Dᵅ(r) = D(r)/2
            # and similarly for the gradient
            #   ∇Dᵅ(r) = ∇D(r)/2
            # and the Laplacian
            #   ∇²Dᵅ(r) = ∇²D(r)/2
            # Since, the spin matrix densities for up and down are the same,
            # the exchange energy is only calculated once and then multiplied by two.
            # xed[Dᵅ(r),Dᵝ(r)] = xed[Dᵅ(r)] + xed[Dᵝ(r)] = 2 xed[Dᵅ(r)]
            if lapl_D is None:
                # Functional does not depend on ∇²D(r)/2
                xed = 2.0 * exchange_functional(D/2.0, grad_D/2.0, None)
            else:
                xed = 2.0 * exchange_functional(D/2.0, grad_D/2.0, lapl_D/2.0)
        else:
            # Compute the exchange energy for spin-up and spin-down electrons separately
            # and add it.
            # xed[Dᵅ(r),Dᵝ(r)] = xed[Dᵅ(r)] + xed[Dᵝ(r)]
            xed = exchange_functional(spin_D, grad_spin_D, lapl_spin_D).sum(axis=0)

        # --- Evaluate functional for correlation ---
        # Some correlation functions operate on the total charge density, while
        # others expect separate matrix densities for spin-up and spin-down.
        if getattr(correlation_functional, "need_spin_density", False):
            ced = correlation_functional(
                spin_D, grad_spin_D, lapl_spin_D,
                spin_polarized=spin_polarized
            )
        else:
            ced = correlation_functional(D, grad_D, lapl_D)

        # convert torch tensors into numpy arrays
        xced_approx = xed + ced
        xced_approx = numpy.transpose(xced_approx.numpy(), axes=[1,2,0])

        xced_approximations.append(xced_approx)

    # number of electronic states
    nstate = msmd.number_of_states

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,8))

    # Diagonal elements of electron repulsion operator
    # (~ classical Coulomb energies of electronic states)
    axes[0].set_ylabel(r"XC energy density $4 \pi r^2 xc_{II}(r)$ / $E_h a_0^{-1}$")
    axes[0].set_xlabel(r"r / $a_0$")
    ## Diagonal elements of exchange matrix are plotted on a log-scale.
    #axes[0].set_yscale("symlog")

    for i in range(0, nstate):

        if plot_multiplicities is not None:
            # Skip multiplicities that are not selected for plotting
            # to avoid very cluttered plots.
            if multiplicities[i] not in plot_multiplicities:
                continue

        line, = axes[0].plot(
            r, 4*numpy.pi*r**2 * xced_exact[i,i,:],
            lw=2, alpha=0.5,
            label=state_labels[i]
        )
        for xced_approx, linestyle in zip(xced_approximations, xc_linestyles):
            axes[0].plot(
                r, 4*numpy.pi*r**2 * xced_approx[i,i,:],
                ls=linestyle, color=line.get_color()
            )

    axes[0].legend(title=r"$\mathbf{(a)}$ diagonal")

    # Off-diagonal elements of electron repulsion operator
    axes[1].set_ylabel(r"XC energy density $4 \pi r^2 xc_{IJ}(r)$ / $E_h a_0^{-1}$")
    axes[1].set_xlabel(r"r / $a_0$")

    for i in range(0, nstate):

        if plot_multiplicities is not None:
            # Skip multiplicities that are not selected for plotting
            # to avoid very cluttered plots.
            if multiplicities[i] not in plot_multiplicities:
                continue

        for j in range(i+1, nstate):
            if multiplicities[i] == multiplicities[j]:
                line, = axes[1].plot(
                    r, 4*numpy.pi*r**2 * xced_exact[i,j,:],
                    lw=2, alpha=0.5,
                    label=state_labels[i]+","+state_labels[j]
                )
                for xced_approx, linestyle in zip(xced_approximations, xc_linestyles):
                    axes[1].plot(
                        r, 4*numpy.pi*r**2 * xced_approx[i,j,:],
                        ls=linestyle, color=line.get_color())
            else:
                # Transition matrix elements between states with different spin are zero.
                numpy.testing.assert_allclose(
                    xced_exact[i,j,:], numpy.zeros_like(xced_exact[i,j,:]),
                    atol=1.0e-5)
                for xced_approx in xced_approximations:
                    numpy.testing.assert_allclose(
                        xced_approx[i,j,:], numpy.zeros_like(xced_exact[i,j,:]),
                        atol=1.0e-5)

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title=r"$\mathbf{(b)}$ off-diagonal")

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    xc_lines = [
        matplotlib.lines.Line2D([], [], ls="-", color="black")
    ]
    for linestyle in xc_linestyles:
        line = matplotlib.lines.Line2D([], [], ls=linestyle, color="black")
        xc_lines.append(line)

    xc_labels = [
        # exact xc energy density
        r"$\text{xc}^{\text{exact}}_{IJ}(r) = \frac{1}{2} \int \frac{ D^{(2)}_{IJ}(r,r') - \sum_k D_{IK}(r) D_{KJ}(r')}{|r-r'|} d^3r'$",
    ]
    # labels for approximate xc energy densities
    for (exchange_name, correlation_name) in xc_names:
        if not spin_polarized:
            x_label = r"2 \epsilon_x^{\text{%s}}[\mathbf{D}/2]_{IJ}(r)" % exchange_name
        else:
            x_label = (
                (r"\epsilon_x^{\text{%s}}[\mathbf{D}^{\alpha}]_{IJ}(r) + " % exchange_name) +
                (r"\epsilon_x^{\text{%s}}[\mathbf{D}^{\beta}]_{IJ}(r)" % exchange_name)
            )
        c_label = r"\epsilon_c^{\text{%s}}[\mathbf{D}]_{IJ}(r)$" % correlation_name
        xc_label = r"$\text{xc}^{\text{MSDFT}}_{IJ}(r) = %s + %s" % (x_label, c_label)
        xc_labels.append(xc_label)

    fig.legend(
        xc_lines,
        xc_labels,
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=2
    )

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

    return fig
