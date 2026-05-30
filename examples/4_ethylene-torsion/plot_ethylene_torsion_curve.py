#!/usr/bin/env python
# coding: utf-8
import matplotlib
import matplotlib.pyplot as plt
import pandas

from mlmsdft.dft.spin import SpinType

# same value as pyscf.data.nist.HARTREE2EV
HARTREE2EV=27.21138602


def subplot_dissociation_curve(
    df, fig, axis,
    methods=["LDA"],
    show_legend=True,
):
    """
    Plot the torsion curve of ethene
    """
    df = df.drop("spin type", axis=1)
    pivot_table = pandas.pivot_table(df,
        values=["energy (Hartree)",],
        index=["basis", "torsion angle (deg)"],
        columns=["method", "spin multiplicity", "state index"]
    )

    # Remove unnecessary indices
    pivot_table = pivot_table.droplevel(level="basis")

    spin_multiplicities = [1,1,1]
    state_indices = [1,2,3]
    colors = ["black", "green", "red"]

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
                print(err)
                continue

            # We compare the energies relative to the S0 minimum
            # (spin_multiplicity=1, state_index=1).
            minimum = pivot_table[("energy (Hartree)", method, 1, 1)].min()
            # Energy relative to minimum in eV
            energy = (energy - minimum) * HARTREE2EV

            axis.plot(
                bond_lengths, energy,
                color=color,
                linestyle=linestyle
                #label=(method.strip(), spin_multiplicity, state_index)
            )

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    method_lines = []
    for linestyle in linestyles:
        line = matplotlib.lines.Line2D([], [], ls=linestyle, color="black")
        method_lines.append(line)

    fig.legend(
        method_lines,
        methods,
        fontsize='x-large',
        frameon=False,
        loc='outside upper center',
        ncol=2
    )

    # Create the invisible colored lines for the axis legend.
    states = [r"$S_0$", r"$S_1$", r"$S_2$"]
    state_lines = []
    for color in colors:
        line = matplotlib.lines.Line2D([], [], color=color)
        state_lines.append(line)

    axis.legend(
        state_lines,
        states,
        fontsize='large',
        frameon=False,
        loc='lower center',
        reverse=True,
        ncol=1
    )


def plot_dissociation_curve(df, methods=["LDA"]):
    fig, axis = plt.subplots(1,1, figsize=(6,6))

    axis.set_xlabel(r"torsion angle / degrees")
    axis.set_ylabel(r"$\Delta E$ / eV")
    axis.set_xticks([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])

    # Select spin type of calculation, for XMS-CASPT2 calculations the column is NaN
    # POLARIZED, UNPOLARIZED and INVARIANT calculations give identical results,
    # since only singlet states are included in the subspace.
    df_invariant = df[
        (df["spin type"] == SpinType.INVARIANT.name) |       # noqa: E712
        df["spin type"].isna()]
    subplot_dissociation_curve(df_invariant, fig, axis, methods=methods)
    return fig


if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')

    # Plot dissociation curve with LDA functional
    # and compare with XMS-CASPT2
    df_msdft = pandas.read_csv("ethylene_torsion.msdft.csv")
    df_xms_caspt2 = pandas.read_csv("ethylene_torsion.xms-caspt2.csv")
    df = pandas.concat([df_msdft, df_xms_caspt2])
    # Rename LDA into LMDA
    df.replace(to_replace="LDA", value="LMDA", inplace=True)
    print(df)

    fig = plot_dissociation_curve(df, methods=["XMS-CASPT2", "LMDA"])
    fig.savefig("ethylene_torsion.png", dpi=300)
    fig.savefig("ethylene_torsion.svg", dpi=300)
    plt.show()
