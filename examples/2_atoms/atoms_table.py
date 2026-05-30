#!/usr/bin/env python
from mlmsdft.utils.atoms import (
    atomic_benchmark_calculations,
    atomic_pivot_table
)
from mlmsdft.utils.atoms import (
    Hydrogen,                                                      Helium,
    Lithium, Beryllium, Boron, Carbon, Nitrogen, Oxygen,
)
from mlmsdft.dft.pure import LDA
from mlmsdft.dft.spin import SpinType


if __name__ == "__main__":
    atoms = [
        Hydrogen(),                                                      Helium(),
        Lithium(), Beryllium(), Boron(), Carbon(), Nitrogen(), Oxygen(),
    ]
    xc_functionals = [
        # (xc_name, xc_functional_class)
        # --- pure functionals
        ('LDA', LDA),
    ]
    # Table with atomic excitation energies using MSDFT with pure functional.
    df = atomic_benchmark_calculations(
        atoms, xc_functionals_list=xc_functionals, spin_types=SpinType)
    pivot_table = atomic_pivot_table(df)
    # Save tables
    df.to_csv("atoms_msdft.csv", index=False)
    pivot_table.to_latex("atoms_msdft.tex", float_format=str)
