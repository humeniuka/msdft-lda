#!/usr/bin/env python
"""
Compute the potential energy curve of the singlet and triplet ground states
for the automerization of cyclobutadiene with the restricted open-shell Kohn-Sham
(RKS) or Hartree-Fock (ROHF) method.

    rectangle <-> square <-> rectangle

Reference
---------
[Eckert-Maksic2006] M. Eckert-Maksic et al.
    "Automerization reaction of cyclobutadiene and its barrier
    height: An ab initio benchmark multireference average-
    quadratic coupled cluster study"
    J. Chem. Phys. 125, 064310 (2006)
    https://doi.org/10.1063/1.2222366
"""
# coding: utf-8
import numpy
import pandas
import pyscf.gto
import pyscf.dft
from tqdm import tqdm


def cyclobutadiene(
    rCCx = 1.562,
    rCCy = 1.349,
    rCH = 1.077,
    angleCHx = 134.9,
    basis = 'cc-pvdz',
    spin = 0,
) -> pyscf.gto.Mole:
    """
    Geometry of D2h or D4h cyclobutadiene

    rCCx: length of horizontal CC bonds (in Ang)
    rCCy: length of vertical CC bonds (in Ang)
    rCH: length of CH bond (in Ang)
    angleCHx: angle between CH bond and x-axis (in degrees)
    spin: 2*S (0 - singlet, 2 - triplet)
    """
    # convert angle to radians
    angleCHx = numpy.deg2rad(angleCHx)
    # position of hydrogen in the top right corner
    xH = rCCx/2.0 + numpy.cos(numpy.pi-angleCHx) * rCH
    yH = rCCy/2.0 + numpy.sin(numpy.pi-angleCHx) * rCH

    # cyclobutadiene
    mol = pyscf.gto.M(
        atom = f"""
        C { rCCx/2.0}    { rCCy/2.0}        0.0;
        C { rCCx/2.0}    {-rCCy/2.0}        0.0;
        C {-rCCx/2.0}    {-rCCy/2.0}        0.0;
        C {-rCCx/2.0}    { rCCy/2.0}        0.0;
        H { xH}          { yH}              0.0;
        H { xH}          {-yH}              0.0;
        H {-xH}          {-yH}              0.0;
        H {-xH}          { yH}              0.0;
        """,
        basis = basis,
        spin = spin
    )
    return mol


def cyclobutadiene_rks_single_point(
    scan_coordinate = 0.0,
    basis = 'cc-pvdz',
    xc_functional_string = 'HF,',
    spin = 0
) -> pandas.DataFrame:
    """
    Perform an RKS or ROHF calculation along the automerization coordinate of cyclobutadiene.

    :param scan_coordinate: parameter s that interpolates linearly
        between the rectangular (s=-1), square (s=0) and the other rectangular (s=+1)
        geometries of cyclobutadiene
    :param basis: basis set
    :param xc_functional_string: Functional string passed to pyscf ('exchange,correlation')
    :param spin: 2*S (0 - singlet, 2 - triplet)

    :return df: pandas dataframe with energies of S0 and T1
    """
    print(f"* Scan coordinate s= {scan_coordinate}")
    # The geometric parameters come from table I row "cc-pVDZ" of [Eckert-Maksic].
    # - D2h square: C=C bonds parallel to y-axis
    #   rCCx=1.573, rCCy=1.367, rCH=1.093, angleCHx=134.9°
    # - D2h square C=C bonds parallel to x-axis
    #   rCCx=1.367, rCCy=1.573, rCH=1.093, angleCHx=270.0°-134.9°
    # - square transition state
    #   rCCx=1.461, rCCy=1.461, rCH=1.092, angleCHx=135.0°

    assert -1.0 <= scan_coordinate <= 1.0
    s = abs(scan_coordinate)
    if scan_coordinate <= 0.0:
        # Interpolate linearly in internal coordinates between
        # the left rectangle minimum and the square transition state.
        mol = cyclobutadiene(
            rCCx=s*1.573+(1-s)*1.461,
            rCCy=s*1.367+(1-s)*1.461,
            rCH=s*1.093+(1-s)*1.092,
            angleCHx=s*134.9+(1-s)*135.0,
            basis=basis,
            spin=spin
        )
    else:
        # Interpolate linearly between the square transition state
        # and the right rectangular minimum.
        mol = cyclobutadiene(
            rCCx=s*1.367+(1-s)*1.461,
            rCCy=s*1.573+(1-s)*1.461,
            rCH=s*1.093+(1-s)*1.092,
            angleCHx=s*(270.0-134.9)+(1-s)*135.0,
            basis=basis,
            spin=spin
        )

    # mol.tofile(f"tmp/cyclobutadiene_{s}.xyz", format="xyz")

    rks = pyscf.dft.RKS(mol)
    rks.xc = xc_functional_string
    rks.verbose = 4
    # Use second order solver with quadratic convergence
    energy = rks.newton().kernel()

    data = {
        "basis": [mol.basis],
        "scan coordinate": [scan_coordinate],
        "state index": [1],
        # 2*S+1
        "spin multiplicity": [spin+1],
        "spin type": [numpy.nan],
        "energy (Hartree)": [energy],
    }
    df_rks = pandas.DataFrame.from_dict(data)

    return df_rks


if __name__ == "__main__":
    # number of scan geometries
    ncoord = 21

    scan_coordinates = numpy.linspace(-1.0, 1.0, ncoord)
    # Break the D4h symmetry slightly.
    scan_coordinates[scan_coordinates == 0.0] += 0.001

    dataframes = []
    for xc_functional_name, xc_functional_string in [
#        ("ROHF", "hf,"),
        ("KS-LDA", "LDA_X,LDA_C_CHACHIYO"),
#        ("KS-BLYP", "b88,lyp")
    ]:
        # Compute potential energy curves
        for i, scan_coordinate in enumerate(tqdm(scan_coordinates)):
            for spin in [0, 2]:
                df_rks = cyclobutadiene_rks_single_point(
                    scan_coordinate=scan_coordinate,
                    xc_functional_string=xc_functional_string,
                    spin=spin
                )
                df_rks["method"] = xc_functional_name

                dataframes.append(df_rks)
                print(df_rks)

                # Save intermediates after each step.
                df = pandas.concat(dataframes)
                df.to_csv("cyclobutadiene_automerization.rks.csv", index=False)

    # Show complete table
    print(df)
