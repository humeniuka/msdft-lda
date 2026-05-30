#!/usr/bin/env python
# coding: utf-8
import matplotlib
import matplotlib.pyplot as plt
import pandas

from mlmsdft.dft.spin import SpinType

# Unit conversion, Hartree -> kcal/mol
HARTREE2KCALMOL = 627.509469

def subplot_automerization_curve(
    df, fig, axis,
    method = "LDA",
    linestyle = "--"
):
    """
    Plot the automerization curve of cyclobutadiene
    """
    df = df.drop("spin type", axis=1)
    pivot_table = pandas.pivot_table(df,
        values=["energy (Hartree)",],
        index=["basis", "scan coordinate"],
        columns=["method", "spin multiplicity", "state index"]
    )

    axis.set_ylim((-5.0, 45.0))

    # Remove unnecessary indices
    pivot_table = pivot_table.droplevel(level="basis")

    spin_multiplicities = [1,1,1,3]
    state_indices = [1,2,3,1]
    colors = ["black", "green", "blue", "red"]

    scan_coordinate = list(pivot_table.index)

    for (spin_multiplicity, state_index, color) in zip(
        spin_multiplicities, state_indices, colors
    ):
        try:
            energy = pivot_table[("energy (Hartree)", method, spin_multiplicity, state_index)]
        except KeyError as err:
            print(f"The following column does not exist in the pivot table: {err}")
            continue

        # We compare the energies relative to the S0 minimum
        # (spin_multiplicity=1, state_index=1).
        minimum = pivot_table[("energy (Hartree)", method, 1, 1)].min()
        # Energy relative to minimum in kcal/mol
        energy_eV = (energy - minimum) * HARTREE2KCALMOL

        axis.plot(
            scan_coordinate, energy_eV,
            color=color,
            linestyle=linestyle
        )

    # Show potential energy curves from correlated wavefunction method (Eckert-Maksic 2006)
    plot_reference_curves(fig, axis)


def show_legends(fig, axis):
    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    method_lines = []
    for linestyle, alpha, lw in zip(["-", "-.", "--"], [0.5, 1.0, 1.0], [2, 1, 1]):
        line = matplotlib.lines.Line2D([], [], ls=linestyle, lw=lw, alpha=alpha, color="black")
        method_lines.append(line)
    methods = ["MR-AQCC", "LMDA", "KS-LDA"]

    fig.legend(
        method_lines,
        methods,
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=3,
    )

    # Create the invisible colored lines for the state labels.
    #states = [r"$1^1A_{g}$", r"$1^1B_{1g}$", r"$2^1A_{g}$", r"$1^3B_{1g}$"]
    #colors = ["black", "green", "blue", "red"]
    # Show only S0 and T1
    states = [r"$S_0$", r"$T_1$"]
    colors = ["black", "red"]
    state_lines = []
    for color in colors:
        line = matplotlib.lines.Line2D([], [], color=color)
        state_lines.append(line)

    axis.legend(
        state_lines,
        states,
        fontsize='large',
        frameon=False,
        reverse=True,
        loc='lower center',
        ncol=1
    )


def plot_reference_curves(fig, axis):
    """
    Plot MR-AQCC/SA-4-CASSCF/cc-pVTZ potential energy curves digitized from
    Fig. 2 of Eckert-Maksic (2006).
    """
    # Singlets
    df = pandas.read_csv(
        "reference/cyclobutadiene_11Ag.txt", names=["s", "dE"], skiprows=1, sep=r'\s+'
    )
    axis.plot(df["s"], df["dE"], color="black", ls="-", lw=2, alpha=0.5, label=r"$1^1A_{g}$")

    df = pandas.read_csv(
        "reference/cyclobutadiene_11B1g.txt", names=["s", "dE"], skiprows=1, sep=r'\s+'
    )
    axis.plot(df["s"], df["dE"], color="green", ls="-", lw=2, alpha=0.5, label=r"$1^1B_{1g}$")

    df = pandas.read_csv(
        "reference/cyclobutadiene_21Ag.txt", names=["s", "dE"], skiprows=1, sep=r'\s+'
    )
    axis.plot(df["s"], df["dE"], color="blue", ls="-", lw=2, alpha=0.5, label=r"$2^1A_{g}$")

    # Triplet
    df = pandas.read_csv(
        "reference/cyclobutadiene_13B1g.txt", names=["s", "dE"], skiprows=1, sep=r'\s+'
    )
    axis.plot(df["s"], df["dE"], color="red", ls="-", lw=2, alpha=0.5, label=r"$1^3B_{1g}$")


def plot_dissociation_curve_vertical(df, methods=["LDA"]):
    # Figure, axes, labels
    fig, axes = plt.subplots(5,1, figsize=(4,13.2), sharex=True)

    for axis in axes:
        axis.set_ylim((-5.0, 40.0))

    axes[2].set_ylabel(r"$\Delta E$ / kcal mol$^{-1}$")

    axes[4].set_xlabel(r"Reaction Coordinate $\sigma$")
    axes[4].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[4].set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_invariant = df[
        (df["spin type"] == SpinType.INVARIANT.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_invariant, fig, axes[0], method="LDA", linestyle="-.")
    axes[0].text(
        0.95, 0.95, "(a) invariant",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[0].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_polarized = df[
        (df["spin type"] == SpinType.POLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_polarized, fig, axes[1], method="LDA", linestyle="-.")
    axes[1].text(
        0.95, 0.95, "(b) polarized",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[1].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_unpolarized = df[
        (df["spin type"] == SpinType.UNPOLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_unpolarized, fig, axes[2], method="LDA", linestyle="-.")
    axes[2].text(
        0.95, 0.95, "(c) unpolarized",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[2].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_invariant_mix = df[
        (df["spin type"] == SpinType.INVARIANT_MIX.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_invariant_mix, fig, axes[3], method="LDA", linestyle="-.")
    axes[3].text(
        0.97, 0.92, "(d) 50% invariant    \n50% unpolarized",
        fontsize=16,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[3].transAxes
    )

    # Kohn-Sham calculation of S0 and T1
    df_ks = df[df["spin type"].isna()]
    subplot_automerization_curve(df_ks, fig, axes[4], method="KS-LDA", linestyle="--")
    axes[4].text(
        0.95, 0.95, "(e) KS-LDA",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[4].transAxes
    )

    show_legends(fig, axes[4])

    plt.subplots_adjust(hspace=0.0)
    return fig

def plot_dissociation_curve_horizontal(df, methods=["LDA"]):
    # Figure, axes, labels
    fig, axes = plt.subplots(1,5, figsize=(13.2, 5))

    axes[0].set_ylabel(r"$\Delta E$ / kcal mol$^{-1}$")

    axes[2].set_xlabel(r"Reaction Coordinate $\sigma$")
    for axis in axes:
        axis.set_xticks([-1, -0.5, 0, 0.5, 1])
        axis.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    axes[3].set_yticks([])

    axes[4].yaxis.set_label_position("right")
    axes[4].yaxis.tick_right()
    axes[4].set_ylabel(r"$\Delta E$ / kcal mol$^{-1}$")

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_invariant = df[
        (df["spin type"] == SpinType.INVARIANT.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_invariant, fig, axes[0], method="LDA", linestyle="-.")
    axes[0].text(
        0.05, 0.95, "(a) invariant",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[0].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_polarized = df[
        (df["spin type"] == SpinType.POLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_polarized, fig, axes[1], method="LDA", linestyle="-.")
    axes[1].text(
        0.05, 0.95, "(b) polarized",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[1].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_unpolarized = df[
        (df["spin type"] == SpinType.UNPOLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_unpolarized, fig, axes[2], method="LDA", linestyle="-.")
    axes[2].text(
        0.05, 0.95, "(c) unpolarized",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[2].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_invariant_mix = df[
        (df["spin type"] == SpinType.INVARIANT_MIX.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_invariant_mix, fig, axes[3], method="LDA", linestyle="-.")
    axes[3].text(
        0.07, 0.92, "(d) 50% invariant    \n50% unpolarized",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[3].transAxes
    )

    # Kohn-Sham calculation of S0 and T1
    df_ks = df[df["spin type"].isna()]
    subplot_automerization_curve(df_ks, fig, axes[4], method="KS-LDA", linestyle="--")
    axes[4].text(
        0.05, 0.95, "(e) KS-LDA",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[4].transAxes
    )

    show_legends(fig, axes[4])

    plt.subplots_adjust(wspace=0.0)
    return fig


def plot_dissociation_curve_2x2(df, methods=["LDA"]):
    # Figure, axes, labels
    fig, axes = plt.subplots(2,2, figsize=(8, 8))

    axes[0,0].set_ylabel(r"$\Delta E$ / kcal mol$^{-1}$")
    axes[0,0].set_xticks([])

    axes[0,1].set_yticks([])
    axes[0,1].set_xticks([])

    axes[1,0].set_xlabel(r"Automerization Coordinate", fontsize=14)
    axes[1,0].set_ylabel(r"$\Delta E$ / kcal mol$^{-1}$")
    axes[1,0].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[1,0].set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])

    axes[1,1].set_xlabel(r"Automerization Coordinate", fontsize=14)
    axes[1,1].set_yticks([])
    axes[1,1].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[1,1].set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_invariant = df[
        (df["spin type"] == SpinType.INVARIANT.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_invariant, fig, axes[0,0], method="LDA", linestyle="-.")
    # Kohn-Sham calculation of S0 and T1
    df_ks = df[df["spin type"].isna()]
    subplot_automerization_curve(df_ks, fig, axes[0,0], method="KS-LDA", linestyle="--")

    axes[0,0].text(
        0.05, 0.95, "(a) invariant",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[0,0].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_polarized = df[
        (df["spin type"] == SpinType.POLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_polarized, fig, axes[0,1], method="LDA", linestyle="-.")
    axes[0,1].text(
        0.05, 0.95, "(b) polarized",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[0,1].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_unpolarized = df[
        (df["spin type"] == SpinType.UNPOLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_unpolarized, fig, axes[1,0], method="LDA", linestyle="-.")
    axes[1,0].text(
        0.05, 0.95, "(c) unpolarized",
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[1,0].transAxes
    )

    # Select spin type of calculation, for MRCI reference calculations the column is NaN
    df_invariant_mix = df[
        (df["spin type"] == SpinType.INVARIANT_MIX.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_automerization_curve(df_invariant_mix, fig, axes[1,1], method="LDA", linestyle="-.")
    axes[1,1].text(
        0.07, 0.92, "(d) equal-weight",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axes[1,1].transAxes
    )

    show_legends(fig, axes[0,1])
    show_legends(fig, axes[1,1])

    plt.subplots_adjust(hspace=0.0)
    plt.subplots_adjust(wspace=0.0)

    return fig


if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')

    # Plot automerization curve with LDA functional
    # and compare with MRCI
    df_msdft = pandas.read_csv("cyclobutadiene_automerization.msdft.csv")
    df_rks = pandas.read_csv("cyclobutadiene_automerization.rks.csv")
    df = pandas.concat([df_msdft, df_rks])
    print(df)

    #fig = plot_dissociation_curve_horizontal(df, methods=["LDA"])
    fig = plot_dissociation_curve_2x2(df, methods=["LDA"])

    fig.savefig("cyclobutadiene_automerization.png", dpi=300)
    fig.savefig("cyclobutadiene_automerization.svg", dpi=300)
    plt.show()
