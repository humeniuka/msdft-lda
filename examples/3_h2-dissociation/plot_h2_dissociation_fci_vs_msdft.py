#!/usr/bin/env python
# coding: utf-8
import matplotlib
import matplotlib.pyplot as plt
import pandas

from mlmsdft.dft.spin import SpinType


def subplot_dissociation_curve(
    df, fig, axis,
    methods=["FCI", " LDA", " B88+LYP", " BR89+LYP"],
    show_legend=True,
    show_method_legend=True):
    """
    Plot the dissociation curve of H2
    """
    df = df.drop("spin type", axis=1)
    pivot_table = pandas.pivot_table(df,
        values=["energy (Hartree)",],
        index=["basis", "bond length (Angstrom)"],
        columns=["method", "spin multiplicity", "state index"]
    )

    # Remove unnecessary indices
    pivot_table = pivot_table.droplevel(level="basis")

    spin_multiplicities = [1,1,1,3]
    state_indices = [1,2,3,1]
    colors = ["black", "green", "blue", "red"]

    bond_lengths = list(pivot_table.index)
    linestyles = ["-", "-.", "--", ":"]
    assert len(methods) <= len(linestyles), "Need one linestyle per method"

    for method,linestyle in zip(methods, linestyles):
        for (spin_multiplicity, state_index, color) in zip(
            spin_multiplicities, state_indices, colors
        ):
            try:
                energy = pivot_table[("energy (Hartree)", method, spin_multiplicity, state_index)]
            except KeyError as err:
                print(f"Missing entry in pivot table: {err}")
                continue
            axis.plot(
                bond_lengths, energy,
                color=color,
                linestyle=linestyle,
                #label=(method.strip(), spin_multiplicity, state_index)
            )

    axis.set_xlim((0.23, 5.0))
    axis.set_ylim((-1.2, -0.38))

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    method_lines = []
    for linestyle in linestyles:
        line = matplotlib.lines.Line2D([], [], ls=linestyle, color="black")
        method_lines.append(line)

    if show_method_legend:
        fig.legend(
            method_lines,
            methods,
            fontsize='large',
            frameon=False,
            loc='outside upper center',
            ncol=4
        )

    # Create the invisible colored lines for the axis legend.
    states = [r"$S_0$", r"$S_1$", r"$S_2$", r"$T_1$"]
    state_lines = []
    for color in colors:
        line = matplotlib.lines.Line2D([], [], color=color)
        state_lines.append(line)

    if show_legend:
        axis.legend(
            state_lines,
            states,
            #title="States",
            fontsize='large',
            frameon=False,
            loc='best',
            reverse=True,
            ncol=1
        )

def plot_dissociation_curve_horizontal(df, methods=["FCI", " LDA", " B88+LYP", " BR89+LYP"]):
    # Figure, axes, labels
    fig, axes = plt.subplots(1,4, figsize=(13.2,4))

    axes[0].set_xlabel(r"r / $\AA$")
    axes[0].set_ylabel(r"Energy / $E_h$")

    axes[1].set_xlabel(r"r / $\AA$")
    #axes[1].yaxis.set_label_position("right")
    axes[1].set_yticks([])
    #axes[1].set_ylabel(r"Energy / $E_h$")

    axes[2].set_xlabel(r"r / $\AA$")
    #axes[2].yaxis.set_label_position("right")
    axes[2].set_yticks([])
    #axes[2].set_ylabel(r"Energy / $E_h$")

    axes[3].set_xlabel(r"r / $\AA$")
    axes[3].yaxis.set_label_position("right")
    axes[3].yaxis.tick_right()
    axes[3].set_ylabel(r"Energy / $E_h$")

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_invariant = df[
        (df["spin type"] == SpinType.INVARIANT.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_invariant, fig, axes[0], show_legend=True, show_method_legend=True, methods=methods)
    axes[0].text(
        0.95, 0.95, "(a) invariant",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[0].transAxes
    )

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_polarized = df[
        (df["spin type"] == SpinType.POLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_polarized, fig, axes[1], show_legend=False, show_method_legend=False, methods=methods)
    axes[1].text(
        0.95, 0.95, "(b) polarized",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[1].transAxes
    )

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_unpolarized = df[
        (df["spin type"] == SpinType.UNPOLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_unpolarized, fig, axes[2], show_legend=False, show_method_legend=False, methods=methods)
    axes[2].text(
        0.95, 0.95, "(c) unpolarized",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[2].transAxes
    )

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_invariant_mix = df[
        (df["spin type"] == SpinType.INVARIANT_MIX.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_invariant_mix, fig, axes[3], show_legend=False, show_method_legend=False, methods=methods)
    axes[3].text(
        0.97, 0.92, "(d) 50% invariant    \n50% unpolarized",
        fontsize=16,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[3].transAxes
    )
    
    plt.subplots_adjust(wspace=0.05)
    return fig


def plot_dissociation_curve_2x2(df, methods=["FCI", " LDA", " B88+LYP", " BR89+LYP"]):
    # Figure, axes, labels
    fig, axes = plt.subplots(2,2, figsize=(8,9))#, sharex=True, sharey=True)

    axes[0,0].set_ylabel(r"Energy / $E_h$")
    axes[0,0].set_yticks([-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5])
    axes[0,0].set_xticks([])

    #axes[0,1].yaxis.set_label_position("right")
    #axes[0,1].yaxis.tick_right()
    #axes[0,1].set_ylabel(r"Energy / $E_h$")
    axes[0,1].set_yticks([])
    axes[0,1].set_xticks([])

    axes[1,0].set_xlabel(r"r / $\AA$")
    axes[1,0].set_ylabel(r"Energy / $E_h$")
    axes[1,0].set_yticks([-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5])
    axes[1,0].set_xticks([0, 1, 2, 3, 4, 5])

    axes[1,1].set_xlabel(r"r / $\AA$")
    #axes[1,1].yaxis.set_label_position("right")
    #axes[1,1].yaxis.tick_right()
    #axes[1,1].set_ylabel(r"Energy / $E_h$")
    axes[1,1].set_yticks([])
    axes[1,1].set_xticks([0, 1, 2, 3, 4, 5])

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_invariant = df[
        (df["spin type"] == SpinType.INVARIANT.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_invariant, fig, axes[0,0], show_legend=False, show_method_legend=False, methods=methods + ["KS-LDA"])
    axes[0,0].text(
        0.95, 0.95, "(a) invariant",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[0,0].transAxes
    )

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_polarized = df[
        (df["spin type"] == SpinType.POLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_polarized, fig, axes[0,1], show_legend=True, show_method_legend=False, methods=methods)
    axes[0,1].text(
        0.95, 0.95, "(b) polarized",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[0,1].transAxes
    )

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_unpolarized = df[
        (df["spin type"] == SpinType.UNPOLARIZED.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_unpolarized, fig, axes[1,0], show_legend=False, show_method_legend=False, methods=methods)
    axes[1,0].text(
        0.95, 0.95, "(c) unpolarized",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[1,0].transAxes
    )

    # Select spin type of calculation, for FCI calculations the column is NaN
    df_invariant_mix = df[
        (df["spin type"] == SpinType.INVARIANT_MIX.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(
        df_invariant_mix, fig, axes[1,1], show_legend=True, show_method_legend=False, methods=methods)
    axes[1,1].text(
        0.97, 0.92, "(d) equal-weight",
        fontsize=18,
        horizontalalignment="right",
        verticalalignment="center",
        transform=axes[1,1].transAxes
    )

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    methods = ["FCI", "LMDA", "KS-LDA"]
    linestyles = ["-", "-.", "--"]
    method_lines = []
    for linestyle in linestyles:
        line = matplotlib.lines.Line2D([], [], ls=linestyle, color="black")
        method_lines.append(line)

    fig.legend(
        method_lines,
        methods,
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=4
    )

    plt.subplots_adjust(hspace=0.0)
    plt.subplots_adjust(wspace=0.0)
    return fig


if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')

    df_msdft = pandas.read_csv("h2_dissociation.csv")
    df_rks = pandas.read_csv("h2_dissociation_lda.rks.csv")
    df = pandas.concat([df_msdft, df_rks])
    print(df)

    # Plot dissociation curve with LDA vs. FCI
    #fig = plot_dissociation_curve_horizontal(df, methods=["FCI", " LDA"])
    fig = plot_dissociation_curve_2x2(df, methods=["FCI", " LDA"])
    fig.savefig("h2_dissociation_lda.png", dpi=300, bbox_inches='tight')
    fig.savefig("h2_dissociation_lda.svg", dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)
