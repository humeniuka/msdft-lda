# -*- coding: utf-8 -*-
"""
Functions for plotting the exact xc-hole and the LDA (HEG) approximation to it.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy
import scipy.special

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


def xc_hole_taylor_expansion(
    msmd: MultistateMatrixDensityFCI,
    center_r: numpy.ndarray,
    distances_u: numpy.ndarray
    ) -> numpy.ndarray:
    """
    Compute the Taylor expansion of the spherically averaged exchange-correlation hole
    as a function of the distance from the reference point.

    For a system with a single electron, the Taylor expansion up to second order
    of the spherical average of the exchange-correlation hole H^{xc}ᵢⱼ(r,r+u)
    over the coordinate u about the reference point r is

        <H^{xc,sr}ᵢⱼ(r,r+u)> = -Dᵢⱼ(r) - 1/6 ∇²Dᵢⱼ(r) u² + ...

    The exact exchange-correlation hole for a one-electron system however is different
    from the Taylor expansion. It is independent of the reference point and is given by

        H^{xc,sr}ᵢⱼ(r,r') = -Dᵢⱼ(r')     (note that the argument of Dᵢⱼ is r', not r!)

    :param msmd: matrix density and its derivatives at the reference point
    :type msmd: instance of MultistateMatrixDensityFCI

    :param center_r: reference point
    :type center_r: numpy.ndarray of shape (3,)

    :param distances_u: distances u from reference point
    :type distances_u: numpy.ndarray of shape (Ndist,)

    :return spherical_xc_hole_sr: short-range Taylor expansion of spherical
        exchange-correlation hole as a function of the distance from the reference point,
        xc_hole_sr[u,:,:] = <H^{xc,sr}ᵢⱼ(r)>(distances_u[u])
    :rtype spherical_xc_hole_sr: numpy.ndarray of shape (Ndist,Nstate,Nstate)
    """
    # Evaluate matrix density and its derivatives at the reference point r
    coords = numpy.reshape(center_r, (1,3))
    # Only Dᵢⱼ(r) and its Laplacian ∇²Dᵢⱼ(r) are needed
    D, _, lapl_D = msmd.evaluate(coords)
    # sum over spin and select reference point
    D = numpy.sum(D, axis=0)[:,:,0]
    lapl_D = numpy.sum(lapl_D, axis=0)[:,:,0]

    # Taylor expansion around u=0 up to quadratic order
    spherical_xc_hole_sr = (
        # -Dᵢⱼ(r)
        -numpy.expand_dims(D, 0)
        # - 1/6 ∇²Dᵢⱼ(r) u²
        -1.0/6.0 * numpy.einsum('ij,u->uij', lapl_D, distances_u**2)
    )

    # The exchange energy should be calculated for spin up and spin down
    # separately. Since only the total charge density is given, we have to
    # divide it by two.
    spherical_xc_hole_sr *= 0.5

    return spherical_xc_hole_sr


def exchange_hole_homogeneous(
    msmd: MultistateMatrixDensityFCI,
    center_r: numpy.ndarray,
    distances_u: numpy.ndarray
) -> numpy.ndarray:
    """
    Compute the (spherically symmetric) exchange hole of the homogeneous electron gas (HEG)
    as a function of the distance from the reference point.

    For the ground state density, the exchange hole is given by (see [Becke1983], without minus sign)

        ρₓ^{HEG}(u) = -9 ρ [j1(k u)/(k u)]²

    with the Fermi momentum

        k = (6π²ρ)¹ᐟ³

    and the spherical Bessel function of 1st order

        j1(x) = sin(x)/x² - cos(x)/x

    Since the hole only depends on the density it can be turned into a matrix function
    by diagonalizing the matrix density, applying ρₓ^{HEG}(u) to each eigenvalue and
    transforming back.

    :param msmd: matrix density at the reference point
    :type msmd: instance of MultistateMatrixDensityFCI

    :param center_r: reference point
    :type center_r: numpy.ndarray of shape (3,)

    :param distances_u: distances u from reference point
    :type distances_u: numpy.ndarray of shape (Ndist,)

    :return x_hole_heg: exchange hole of the homogeneous electron gas
        as a function of the distance from the reference point,
    :rtype x_hole_heg: numpy.ndarray of shape (Ndist,Nstate,Nstate)

    References
    ----------
    [Becke1983] Becke, A. D.
        "Hartree-Fock exchange energy of an inhomogeneous electron gas."
        International journal of quantum chemistry 23.6 (1983): 1915-1922.
        doi:10.1002/qua.560230605
    """
    # Evaluate matrix density at the reference point r
    coords = numpy.reshape(center_r, (1,3))
    # Only Dᵢⱼ(r) is needed
    D, _, _ = msmd.evaluate(coords)
    # sum over spin and select reference point
    D = numpy.sum(D, axis=0)[:,:,0]

    # The exchange energy should be calculated for spin up and spin down
    # separately. Since only the total charge density is given, we have to
    # divide it by two.
    D *= 0.5

    def exchange_hole_heg(density, u):
        # Fermi momentum
        kF = pow(6.0 * numpy.pi**2 * density, 1.0/3.0)
        x = kF*u
        # spherical Bessel function of 1st order
        j1 = scipy.special.spherical_jn(1, x)
        rho_x_heg = -9.0 * density * pow(j1/x, 2)
        return rho_x_heg

    # Compute eigenvalues Λ and eigenvectors U of the symmetric matrix D.
    L, U = numpy.linalg.eigh(D)

    # number of electronic states
    nstate = msmd.number_of_states
    # number of distances u
    ndist = distances_u.shape[0]

    # Empty output array
    x_hole_heg = numpy.zeros((ndist,nstate,nstate))

    # Loop over distances u from reference point
    for i,u in enumerate(distances_u):
        x_hole_heg[i,:,:] = numpy.einsum(
            'ia,a,ja->ij',
            # X = U ρₓ(Λ,u) U⁻¹
            U,
            # apply ρₓ(Λ,u) to eigenvalues Λ
            exchange_hole_heg(L, u),
            U
        )

    return x_hole_heg


def check_hole_normalization(x_hole, distances_u):
    """
    verify that the exchange hole is normalized

        4 π ∫ Dₓ(u) u² du = -Id
    """
    du = numpy.ediff1d(distances_u, to_end=distances_u[-1]-distances_u[-2])
    integral = numpy.einsum(
        'uij,u->ij', x_hole, 4.0 * numpy.pi * distances_u**2 * du
    )
    print(r"4 π ∫ Dₓ(u) u² du")
    with numpy.printoptions(precision=3, suppress=True):
        print(integral)

    # number of electronic states
    #nstate = x_hole.shape[-1]
    #Id = numpy.eye(nstate)
    # check normalization condition
    #numpy.testing.assert_allclose(integral, -Id)


def plot_spherically_averaged_xc_holes(
    msmd: MultistateMatrixDensityFCI,
    state_labels,
    multiplicities,
    center_r = numpy.array([0.0, 0.0, 0.5]),
    distances_u = numpy.linspace(1.0e-6, 2.0, 500),
    distances_u_small = numpy.linspace(1.0e-6, 0.1, 500),
    plot_methods = ["Taylor", "HEG"],
    xlim = None,
    plot_multiplicities = None
):
    """
    The exact spherically-average exchange-correlation hole is plotted and
    compared with

        1) the Taylor expansion of the exchange hole around u=0
        2) the exchange-hole of the homogeneous electron grad (HEG)

    The spherically averaged exchange-correlation hole around some point r,

    is defined as

        H^{xc}ᵢⱼ(r,|u|) = 1/(4 π) ∫ dΩ H^{xc}ᵢⱼ(r,r+|u|*e(Ω))

    where e(Ω) is a unit vector in the direction Ω, such that u = r'-r = |u|*e(Ω).

    H^{xc}ᵢⱼ(r,|u|) is plotted as a function of the distance |u| from r
    (similarly to figures in [Becke/Roussel1989]).


    :param msmd: multistate matrix density from a full CI calculation
        with pair density, from which the exact xc-hole is calculated
    :type msmd: MultistateMatrixDensityFCI

    :param state_labels: list of strings for each of the electronic states
        in matrix density that will be used to label the diagonal and
        off-diagonal elements of the xc-hole
    :type state_labels: list of str
        e.g. [r"1¹S", r"1³S", r"2¹S"] for the lowest 3 states of the He atom

    :param multiplicities: list of spin multiplicity for each state.
        Only states with Sz=0 are calculated. The spin multiplicities are used
        to distinguish different spin states and only plot the off-diagonal
        elements of the xc-hole between states with the same multiplicity,
        since those between different ones are always zero.
    :type multiplicities: list of int
        e.g. [1,3,1] for the lowest 3 states of the He atom

    :param center_r: reference point r where matrix density D(r) and its
        derivatives are evaluated
    :type center_r: ndarray of shape (3,)

    :param distances_u: distances u = |r'-r| to the reference point
        The xc-hole is plotted as function of u
    :type distances_u: numpy.ndarray of shape (Nu,)

    :param distances_u_small: distances u = |r'-r| to the reference point
        for which the quadratic Taylor expansion of the xc-hole should be plotted.
        For large values the quadratic approximation diverges.
    :param distance_u_small: numpy.ndarray of shape (Nsmall,)

    :param plot_methods: list of hole approximations that should be shown in the plot
        "Taylor", "HEG"
    :type plot_methods: list of str

    :param plot_multiplicities: If different from None, only states with the selected
        multiplicities are plotted
    :type plot_multiplicities: list of int (e.g. [1,3]) or None

    :param xlim: range of x-axis for plotting, unless None
    :type xlim: tuple (xmin,xmax)

    :return fig: The matplotlib figure which has the diagonal elements of
        the xc-hole and its approximations as functions of u in the left axis
        and the off-diagonal elements in the right axis.
    :rtype fig: matplotlib figure


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
    # --- Exact spherical xc-hole ---
    spherical_xc_hole = msmd.spherically_averaged_xc_hole(center_r, distances_u)
    print("Exact spherically averaged xc-hole")
    check_hole_normalization(spherical_xc_hole, distances_u)

    if "Taylor" in plot_methods:
        # --- Quadratic Taylor expansion around u=0 derived for one-electron system ---
        spherical_xc_hole_taylor = xc_hole_taylor_expansion(msmd, center_r, distances_u_small)

    if "HEG" in plot_methods:
        # --- exchange hole of homogeneous electron gas ---
        x_hole_heg = exchange_hole_homogeneous(msmd, center_r, distances_u)
        print("Exchange hole of homogeneous electron gas")
        check_hole_normalization(x_hole_heg, distances_u)

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,8))

    axes[0].text(
        0.7, 0.1, "reference point \n"+r" $\vec{r}$=( %2.1f, %2.1f, %2.1f )" % tuple(center_r),
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=14,
        bbox = dict(boxstyle='round', facecolor='white', alpha=0.5),
        transform = axes[0].transAxes
    )

    # Diagonal elements of xc-hole
    axes[0].set_ylabel(r"spherically averaged xc-hole / $e a_0^{-3}$")
    axes[0].set_xlabel(r"u / $a_0$")
    ## Diagonal elements of xc-hole are plotted on a log-scale.
    #axes[0].set_yscale("symlog")

    nstate = msmd.number_of_states
    for i in range(0, nstate):

        if plot_multiplicities is not None:
            # Skip multiplicities that are not selected for plotting
            # to avoid very cluttered plots.
            if multiplicities[i] not in plot_multiplicities:
                continue

        line, = axes[0].plot(
            distances_u, spherical_xc_hole[:,i,i],
            lw=2, alpha=0.5,
            label=state_labels[i]
        )
        if "Taylor" in plot_methods:
            axes[0].plot(
                distances_u_small, spherical_xc_hole_taylor[:,i,i],
                lw=2,
                ls=":", color=line.get_color()
            )
        if "HEG" in plot_methods:
            axes[0].plot(
                distances_u, x_hole_heg[:,i,i],
                lw=2,
                ls="-.", color=line.get_color()
            )

    axes[0].legend(title="$\mathbf{(a)}$ diagonal")

    # Off-diagonal elements of xc-hole
    axes[1].set_ylabel(r"spherically averaged xc-hole / $e a_0^{-3}$")
    axes[1].set_xlabel(r"u / $a_0$")

    for i in range(0, nstate):

        if plot_multiplicities is not None:
            # Skip multiplicities that are not selected for plotting
            # to avoid very cluttered plots.
            if multiplicities[i] not in plot_multiplicities:
                continue

        for j in range(i+1, nstate):
            if multiplicities[i] == multiplicities[j]:
                line, = axes[1].plot(
                    distances_u, spherical_xc_hole[:,i,j],
                    lw=2, alpha=0.5,
                    label=state_labels[i]+","+state_labels[j]
                )
                if "Taylor" in plot_methods:
                    axes[1].plot(
                        distances_u_small, spherical_xc_hole_taylor[:,i,j],
                        lw=2,
                        ls=":", color=line.get_color()
                    )
                if "HEG" in plot_methods:
                    axes[1].plot(
                        distances_u, x_hole_heg[:,i,j],
                        ls="-.", color=line.get_color()
                    )
            else:
                # Transition matrix elements between states with different spin are zero.
                numpy.testing.assert_allclose(
                    spherical_xc_hole[:,i,j], numpy.zeros_like(spherical_xc_hole[:,i,j]),
                    atol=1.0e-5)

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(b)}$ off-diagonal")

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", lw=2, color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls=":", lw=2, color="black")
    dashdot_line = matplotlib.lines.Line2D([], [], ls="-.", lw=2, color="black")

    handles = []
    labels = []

    handles.append(solid_line)
    labels.append(
        r"$<H^{\text{xc}}_{IJ}>(u) = \frac{1}{4 \pi} \int H^{\text{xc}}_{IJ}(\vec{r},\vec{u}) d\Omega_u$"
    )
    if "Taylor" in plot_methods:
        handles.append(dashed_line)
        labels.append(
            r"$<H^{\text{xc,Taylor expansion}}_{IJ}>(u) = -D_{IJ}(\vec{r}) -\frac{1}{6} \nabla^2 D_{IJ}(\vec{r}) u^2$"
        )
    if "HEG" in plot_methods:
        handles.append(dashdot_line)
        labels.append(r"$H^{\text{x,HEG}}_{IJ}(u)$")

    fig.legend(handles, labels,
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=2
    )

    if xlim is not None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.025, left=0.2, right=0.86)

    return fig
