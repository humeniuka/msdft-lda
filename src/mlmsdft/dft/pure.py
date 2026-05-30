# -*- coding: utf-8 -*-
""" Composite pure functionals """
from abc import ABC
from torch import Tensor

from mlmsdft.dft.xc import lda_x_dirac, lda_c_chachiyo


class PureXCFunctional(ABC):
    """
    Base class for composite functionals
    """
    # Pure functional have no exact (HF) exchange part.
    exact_exchange = None

    def __init__(self, mol_dummy):
        # Pure functionals do not need access to the basis set.
        pass

    def exchange(
        self,
        matrix_density: Tensor,
        grad_D: Tensor,
        # `lapl_dummy` argument is ignored.
        lapl_dummy: Tensor = None
    ) -> Tensor:
        """ Compute part of exchange energy density from matrix density and its gradient. """
        # This method has to be implemented by the hybrid functional.
        return 0.0 * matrix_density

    def correlation(
        self,
        matrix_density: Tensor,
        grad_D: Tensor,
        # `lapl_dummy` argument is ignored.
        lapl_dummy: Tensor = None
    ) -> Tensor:
        """ Compute correlation energy density from matrix density and its gradient. """
        # This method has to be implemented by the hybrid functional.
        return 0.0 * matrix_density


class LDA(PureXCFunctional):
    def exchange(
        self,
        matrix_density: Tensor,
        grad_D: Tensor,
        # `lapl_dummy` argument is ignored.
        lapl_dummy: Tensor = None
    ) -> Tensor:
        """ Dirac exchange """
        # exchange energy density
        xed = lda_x_dirac(matrix_density)
        return xed

    def correlation(
        self,
        matrix_density: Tensor,
        grad_D: Tensor,
        # `lapl_dummy` argument is ignored.
        lapl_dummy: Tensor = None
    ) -> Tensor:
        """ Chachiyo's correlation """
        # correlation energy density
        ced = lda_c_chachiyo(matrix_density)
        return ced
