#!/usr/bin/env python
"""
Compute the potential energy curves of the lowest three singlet states
of twisted ethylene with XMS-CASPT2 theory using BAGEL (https://nubakery.org/)

The active space consists of 2 electrons in the 2 orbitals.
"""
import json
import numpy
import pandas
import subprocess
import tempfile


def bagel_input(torsion_angle=0.0, basis="cc-pvdz") -> dict:
    """
    Create dictionary with BAGEL commands for a XMS-CASPT2 calculation. 
    """
    # The geometry of ethylene is taken from A. I. Krylov, Chemical Physics Letters 338, 375 (2001).
    rCC = 1.330
    rCH = 1.076
    # H-C-H angle in radians
    angleHCH = numpy.deg2rad(116.6)
    #
    xH = numpy.cos(angleHCH/2.0) * rCH
    yH = numpy.sin(angleHCH/2.0) * rCH
    # torsion angle
    alpha = numpy.deg2rad(torsion_angle)
    c = numpy.cos(alpha)
    s = numpy.sin(alpha)

    return {
        "bagel" : [
        {
            "title" : "molecule",
            "basis" : basis,
            "df_basis" : f"{basis}-jkfit",
            "geometry" : [
                { "atom" : "C", "xyz" : [      -rCC/2.0,      0.0,      0.0 ] },
                { "atom" : "C", "xyz" : [       rCC/2.0,      0.0,      0.0 ] },
                { "atom" : "H", "xyz" : [      -rCC/2.0-xH,  -yH,       0.0 ] },
                { "atom" : "H", "xyz" : [      -rCC/2.0-xH,   yH,       0.0 ] },
                { "atom" : "H", "xyz" : [       rCC/2.0+xH,   c*yH,    s*yH ] },
                { "atom" : "H", "xyz" : [       rCC/2.0+xH,  -c*yH,   -s*yH ] }
            ],
            "angstrom": True
        },
        {
            "title" : "hf"
        },
        {
            "title": "print",
            "file": f"/tmp/orbitals_{torsion_angle}.hf.molden",
            "orbitals": True
        },
        {
            "title" : "casscf",
            "nstate" : 3,
            "nspin": 0,
            # Ethylene has 2*6+4 = 16 electrons, 14 electrons occupy the 7 closed orbitals,
            # leaving 2 correlated electrons in 2 active orbitals.
            "nact" : 2,
            "nclosed" : 7,
            "maxiter" : 100,
            # continue running even if the maximum iterations is reached without convergence
            "conv_ignore": True
        },
        {
            "title" : "smith",
            "method" : "caspt2",
            "ms" : True,
            "xms" : True,
            "sssr" : True,
            "shift" : 0.3
        }
        ]
    }


def parse_bagel_output(output: str) -> pandas.DataFrame:
    """
    Extract XMS-CASPT2 energies from single point calculation.
    """
    energies = []
    state_indices = []
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith("ERROR: EXCEPTION RAISED:"):
            print(output)
            raise RuntimeError(line)

        if line.startswith("* MS-CASPT2 energy : state"):
            words = line.split()
            # state indices are 1-based
            state_index = int(words[5])+1
            state_indices.append(state_index)
            energy = float(words[6])
            energies.append(energy)

    data = {
        "state index": state_indices,
        "energy (Hartree)": energies,
    }
    df_xms_caspt2 = pandas.DataFrame.from_dict(data)
    return df_xms_caspt2


if __name__ == "__main__":
    # number of torsion angles
    ncoord = 91

    # basis set
    basis = "cc-pvdz"

    torsion_angles = numpy.linspace(0.0, 180.0, ncoord)
    dataframes = []

    # Loop over angles
    for i, torsion_angle in enumerate(torsion_angles):
        # Prepare .json input file for BAGEL
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            inp = bagel_input(torsion_angle=torsion_angle, basis=basis)
            f.write(json.dumps(inp))
            f.close()

            # Run BAGEL
            json_file = f.name
            args = ["BAGEL", json_file]
            completed_process = subprocess.run(args, capture_output=True, check=True)

        # Extract energies from BAGEL's output
        output = completed_process.stdout.decode('utf-8')
        df_xms_caspt2 = parse_bagel_output(output)

        # Additional fields that are the same for all states.
        df_xms_caspt2["torsion angle (deg)"] = torsion_angle
        df_xms_caspt2["method"] = "XMS-CASPT2"
        df_xms_caspt2["basis"] = basis
        df_xms_caspt2["spin multiplicity"] = 1
        df_xms_caspt2["spin type"] = numpy.nan

        dataframes.append(df_xms_caspt2)
        print(df_xms_caspt2)

    df = pandas.concat(dataframes)
    df.to_csv("ethylene_torsion.xms-caspt2.csv", index=False)
    print(df)
