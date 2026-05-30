# coding: utf-8
import numpy
import pyscf.gto
import pyscf.pbc.gto

from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.density import MultistateMatrixDensityKohnSham


class FixtureMixin:
    @classmethod
    def create_test_molecules(cls):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            'hydrogen atom (large basis set)': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = 'aug-cc-pvtz',
                # doublet
                spin = 1),
            'hydrogen molecular ion': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 1,
                spin = 1),
            # 2-electron systems, paired spins
            'hydrogen molecule': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 0),
            # 2-electron systems, parallel spins
            'hydrogen molecule (triplet)': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 2),
            # 3-electron systems, one unpaired spin
            'lithium atom': pyscf.gto.M(
                atom = 'Li 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            # 4-electron system, closed shell
            'lithium hydride': pyscf.gto.M(
                atom = 'Li 0 0 0; H 0 0 1.60',
                basis = '6-31g',
                # singlet
                spin = 0),
            # many electrons
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0),
            # pseudo potential which removes the 1s orbital of oxygen
            'water (pseudo)': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'gth-szv',
                pseudo = 'gthbp',
                # singlet
                spin = 0),
            # effective core potential which removes the 1s orbital of oxygen
            'oxygen (ECP)': pyscf.gto.M(
                atom = 'O  0 0 0',
                basis = {'O': 'crenbl'},
                ecp = {'O': 'crenbl'},
                # triplet
                spin = 2),
            # Closed shell molecules with heavy atom whose core electrons are
            # replaced by effective core potentials
            'hydrogen sulfide (ECP)': pyscf.gto.M(
                atom = 'S 0.0000 0.0000 0.1030; H 0.0000 0.9616 -0.8239; H 0.0000 -0.9616 -0.8239',
                basis = {'H': '6-31g', 'S': 'lanl08'},
                ecp = {'S': 'lanl08'},
                # singlet
                spin = 0
            ),
        }
        return molecules

    @classmethod
    def create_test_crystals(cls):
        """ dictionary with different crystals to run the tests on """

        def Cell_without_diffuse(*args, **kwargs):
            # Wrapper removes diffuse basis functions (with exponents < 0.1)
            # to avoid singular overlap matrix.
            cell = pyscf.pbc.gto.Cell()
            cell.exp_to_discard = 0.1
            cell.build(*args, **kwargs)
            return cell

        crystals = {
            'hydrogen crystal': pyscf.pbc.gto.M(
                atom = 'H  0 0 0; H 1 1 1',
                basis = 'sto-3g',
                a = numpy.eye(3) * 2),
            'silicon crystal': Cell_without_diffuse(
                atom = 'Si 0 0 0; Si 1.3575 1.3575 1.3575',
                basis = 'gth-szv',
                pseudo = 'gthbp',
                a = numpy.array([
                    [0.0, 2.715, 2.715],
                    [2.715, 0.0, 2.715],
                    [2.715, 2.715, 0.0]])
                ),
        }
        return crystals

    @classmethod
    def create_test_systems(cls):
        """ combined molecule and crystal test systems """
        return {**cls.create_test_molecules(), **cls.create_test_crystals()}

    @classmethod
    def create_test_molecules_minimal(cls):
        """ smaller set of test molecules for long-running tests """
        molecules = cls.create_test_molecules()
        minimal = {key: molecules[key] for key in ['hydrogen molecular ion', 'hydrogen molecule', 'water']}
        return minimal

    @classmethod
    def create_test_molecules_closed_shell(cls):
        """ smaller set of closed-shell molecules for long-running tests """
        molecules = cls.create_test_molecules()
        minimal = {key: molecules[key] for key in ['hydrogen molecule', 'water', 'hydrogen sulfide (ECP)']}
        return minimal

    @classmethod
    def create_test_molecules_single_electron(cls):
        """ set of test molecules with only a single electron """
        one_electron = {}
        for name, mol in cls.create_test_molecules().items():
            if mol.tot_electrons() == 1:
                one_electron[name] = mol
        return one_electron

    @classmethod
    def create_test_systems_minimal(cls):
        """ smaller set of test molecules/crystals for long-running tests """
        systems = cls.create_test_systems()
        minimal = {key: systems[key] for key in ['hydrogen molecular ion', 'hydrogen molecule', 'water', 'hydrogen crystal']}
        return minimal

    @classmethod
    def create_random_matrix_densities(cls, mol):
        """
        Iterator of multistate matrix densities with random parameters.
        """
        # Kohn-Sham (single state)
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random", seed=567)
        yield msmd

        # CAS (multiple states)
        if mol.tot_electrons() % 2 == 0:
            norb = 2
            nelec = 2
        else:
            norb = 2
            nelec = 1

        for spin_symmetry in [True, False]:
            msmd = MultistateMatrixDensityCAS.from_guess(
                mol, norb, nelec,
                spin_symmetry=spin_symmetry,
                guess="random", seed=986
            )
            yield msmd
