#!/usr/bin/env python
"""
Compute the dissociation curve of H2 in the singlet and triplet ground states
with restricted Hartree-Fock or Kohn-Sham theory using the LDA xc-functional.
"""
# coding: utf-8
import numpy
import pandas
import pyscf.gto
import pyscf.dft
from tqdm import tqdm



def h2_rks_single_point(
    bond_length = 0.0,
    basis = 'cc-pvdz',
    xc_functional_string = 'HF,',
    spin = 0
) -> pandas.DataFrame:
    """
    Perform an RKS or ROHF calculation on stretch H2.

    :param bond_length: H-H bond length in Angstrom
    :param basis: basis set
    :param xc_functional_string: Functional string passed to pyscf ('exchange,correlation')
    :param spin: 2*S (0 - singlet, 2 - triplet)

    :return df: pandas dataframe with energies of S0 and T1
    """
    print(f"* Bond length r= {bond_length}")
    # H2
    mol = pyscf.gto.M(
        atom = f"H 0 0 {-r/2}; H 0 0 {r/2}",
        basis = basis,
        charge = 0,
        spin = spin
    )

    rks = pyscf.dft.RKS(mol)
    rks.xc = xc_functional_string
    rks.verbose = 4
    # Use second order solver with quadratic convergence
    energy = rks.newton().kernel()

    data = {
        "basis": [mol.basis],
        "bond length (Angstrom)": [bond_length],
        "state index": [1],
        # 2*S+1
        "spin multiplicity": [spin+1],
        "spin type": [numpy.nan],
        "energy (Hartree)": [energy],
    }
    df_rks = pandas.DataFrame.from_dict(data)

    return df_rks


if __name__ == "__main__":
    # number of bond lengths
    ncoord = 90

    basis = 'cc-pvdz'

    bond_lengths = numpy.linspace(0.2, 5.0, ncoord)
    dataframes = []
    for xc_functional_name, xc_functional_string in [
#        ("ROHF", "hf,"),
        ("KS-LDA", "LDA_X,LDA_C_CHACHIYO"),
#        ("KS-BLYP", "b88,lyp")
    ]:
        # Compute potential energy curves
        for i, r in enumerate(tqdm(bond_lengths)):
            for spin in [0, 2]:
                df_rks = h2_rks_single_point(
                    bond_length=r,
                    xc_functional_string=xc_functional_string,
                    spin=spin
                )
                df_rks["method"] = xc_functional_name

                dataframes.append(df_rks)
                print(df_rks)

                # Save intermediates after each step.
                df = pandas.concat(dataframes)
                df.to_csv("h2_dissociation_lda.rks.csv", index=False)

    # Show complete table
    print(df)
