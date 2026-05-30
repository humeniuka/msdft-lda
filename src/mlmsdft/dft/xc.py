# -*- coding: utf-8 -*-
import math
import torch
from torch import Tensor
import torch.nn
import torch.nn.functional
import torch.linalg

from mlmsdft.nn.functional import ScalarFunction, MatrixFunction


__all__ = [
    "lda_x_dirac",
    "lda_c_chachiyo",
]

# Cₓ = (3/4) (3/pi)¹ᐟ³ = 0.7386 from Dirac's exchange-energy, Eqn. (6.1.20) in [Parr&Yang]
Cx_Dirac = 3.0/4.0 * pow(3.0/math.pi, 1.0/3.0)
# Cₓ from the "Gaussian" approximation in Eqn. (6.5.25) of [Parr&Yang]
Cx_Gaussian = 0.7937

class _LDAExchangeDirac(ScalarFunction):
    # The prefactor Cₓ for the exchange energy.
    # Depending on whether the exchange energy is calculated from the spin density or the
    # total density, the prefactor is different. Cx is used for spin densities and Cx_Dirac
    # for the total density. For a closed shell, where ρᵅ=ρᵝ=ρ/2 such that ρ=ρᵅ+ρᵝ=2 ρᵅ, we
    # have
    #   Cx_Dirac (2 ρᵅ)⁴ᐟ³ = Cx [(ρᵅ)⁴ᐟ³ + (ρᵝ)⁴ᐟ³]
    # which means that
    #   Cx = 2¹ᐟ³ Cx_Dirac
    Cx = pow(2.0, 1.0/3.0) * Cx_Dirac

    @staticmethod
    def value(scalar_density: Tensor) -> Tensor:
        """
        compute the energy density for the exchange-like part of the electron-electron
        repulsion for a scalar density,

            XED[ρ](r) = -Cₓ ρ(r)⁴ᐟ³

        NOTE: At odds with the usual definition of the exchange energy density,
        (εₓ,ᵢⱼ(r) ∝ ρ(r)¹ᐟ³), XED contains an additional factor of ρ(r)
        (XED(r) ∝ ρ(r)⁴ᐟ³), since the exchange energy is calculated
        as X[D] = ∫ XED(r) dr rather than X[ρ] = ∫ ρ(r) εₓ(r) dr.

        :param scalar_density: density ρ
        :type scalar_density: arbitrary Tensor

        :return xed: exchange energy density
        :rtype xed: Tensor with same shape as input
        """
        # Since the matrix density is positive definite, the argument
        # `scalar_density` should always be positive. However, due to
        # finite numerical precision negative values close to 0 might occur.
        # To avoid NaN's in pow(rho,4/3.0) we take the absolute value.
        rho = torch.abs(scalar_density)
        # exchange energy is always negative
        xed = (-1) * _LDAExchangeDirac.Cx * torch.pow(rho, 4.0/3.0)
        return xed

    @staticmethod
    def derivative1(scalar_density: Tensor) -> Tensor:
        """
        Derivative of exchange energy density w/r/t density

            XED'[ρ] = d(XED[ρ])/dρ = -4/3 Cₓ ρ¹ᐟ³
        """
        rho = torch.abs(scalar_density)
        xed_deriv1 = -4.0/3.0 * _LDAExchangeDirac.Cx * torch.pow(rho, 1.0/3.0)
        return xed_deriv1


def lda_x_dirac(
        matrix_density: Tensor,
        # `grad_dummy` and `lapl_dummy` arguments are ignored.
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None
    ) -> Tensor:
    """
    Multi-state exchange energy according to the local density approximation
    (eqn. 6.5.29 in Ref. [Yang&Parr]),

    X[Dᵅ(r)] = Cₓ ∫ Dᵅ(r)⁴ᐟ³ dr = ∫ XED(r) dr

    Dᵅ(r)⁴ᐟ³ is a fractional matrix-power of Dᵅ(r), which is calculated by diagonalizing Dᵅ.

    The value of the prefactor Cₓ = 2¹ᐟ³ 0.7386 is taken from Dirac's approximation in
    Eqn. (6.1.20) of chapter 6 in Ref. [Yang&Parr]

    References
    ----------
    [Yang&Parr] Parr & Yang (1989), "Density Functional Theory of Atoms and Molecules".

    :param matrix_density: matrix density, Dᵅᵢⱼ(r), Dᵝᵢⱼ(r) or Dᵢⱼ/2
            D[...,i,j] = Dᵅᵢⱼ
    :type matrix_density: Tensor of shape (...,n,n)

    :return xed: exchange energy density
        xed[...,i,j] = Cₓ (Dᵅ(r)⁴ᐟ³)ᵢⱼ
    :rtype xed: Tensor of shape (...,n,n)

    where the indices i,j=1,...,n run over the number of electronic states.
    """
    return MatrixFunction.apply(_LDAExchangeDirac, matrix_density)


class _LDACorrelationChachiyo(ScalarFunction):
    @staticmethod
    def functional_parameters(spin: int):
        """ parameters in Chachiyo's functional """
        assert spin in [0,1]
        # Parameters of Chachiyo's functional from Eqn.(3) of [Chachiyo]
        a = (math.log(2.0)-1.0)/(2*math.pi**2)
        # b from Eqn.(3) for the paramagnetic part εᶜ₀
        b_paramagnetic = 20.4562557
        # b from Eqn.(12) for the ferromagnetic part εᶜ₁
        b_ferromagnetic = 27.4203609

        # The parameter b is different from paramagnetic or ferromagnetic densities.
        if spin == 1:
            a = 0.5 * a
            b = b_ferromagnetic
        else:
            b = b_paramagnetic
        b1 = pow(4.0/3.0*math.pi, 1.0/3.0) * b
        b2 = pow(4.0/3.0*math.pi, 2.0/3.0) * b
        return (a, b1, b2)

    @staticmethod
    def value(scalar_density: Tensor, spin: int) -> Tensor:
        """
        compute the energy density for the correlation-like part of the electron-electron
        repulsion for a scalar density,

            CED[ρ](r) = a log(1 + b₁ ρ(r)¹ᐟ³ + b₂ ρ(r)²ᐟ³ ) ρ(r)

        NOTE: At odds with the usual definition of the correlation energy density,
        CED contains an additional factor of ρ(r), since the correlation energy is calculated as
        C[D] = ∫ CED(r) dr rather than C[ρ] = ∫ ρ(r) εᶜ(r) dr.

        :param scalar_density: density ρ
        :type scalar_density: arbitrary Tensor

        :return ced: correlation energy density
        :rtype ced: Tensor with same shape as input
        """
        a, b1, b2 = _LDACorrelationChachiyo.functional_parameters(spin)
        # Since the matrix density is positive definite, the argument
        # `scalar_density` should always be positive. However, due to
        # finite numerical precision negative values close to 0 might occur.
        # To avoid NaN's in pow(rho,4/3.0) we take the absolute value
        # and add a tiny positive number.
        rho = torch.abs(scalar_density) + 1.0e-15
        # In terms of the density the correlation energy becomes
        #  εᶜ(ρ) = a log( 1 + b1 ρ¹ᐟ³ + b2 ρ²ᐟ³ )
        arg = 1.0 + b1 * torch.pow(rho, 1.0/3.0) + b2 * torch.pow(rho, 2.0/3.0)
        epsilon_c = a * torch.log(arg)
        # Multiply the correlation energy per particle by the particle density
        # to get the correlation energy density (CED(r))
        ced = epsilon_c * rho
        return ced

    @staticmethod
    def derivative1(scalar_density: Tensor, spin: int) -> Tensor:
        """
        Derivative of correlation energy density w/r/t density

            CED'[ρ] = d(CED[ρ])/dρ = d(εᶜ(ρ))/dρ ρ + εᶜ(ρ)
        """
        a, b1, b2 = _LDACorrelationChachiyo.functional_parameters(spin)
        rho = torch.abs(scalar_density) + 1.0e-15
        # d(ced)/dρ = d(εᶜ(ρ))/dρ ρ + εᶜ(ρ)
        arg = 1.0 + b1 * torch.pow(rho, 1.0/3.0) + b2 * torch.pow(rho, 2.0/3.0)
        ced_deriv1 = a * (
            (1.0/3.0 * b1 * torch.pow(rho, 1.0/3.0) + 2.0/3.0 * b2 * torch.pow(rho, 2.0/3.0)
            ) / arg + torch.log(arg))
        return ced_deriv1


def lda_c_chachiyo(
        matrix_density: Tensor,
        # `grad_dummy` and `lapl_dummy` arguments are ignored.
        grad_dummy: Tensor = None,
        lapl_dummy: Tensor = None,
        spin=0
    ) -> Tensor:
    """
    Multi-state correlation energy according to the local density approximation.

    For a single electronic state it reduces to the correlation energy of the uniform
    electron gas. The functional form from [Chachiyo] is a simple and elegant parameterization
    of the correlation energy per electron of the uniform electron gas.
    It recovers the exact high density limit and fits the quantum Monte-Carlo results of
    [Ceperley&Alder] in the medium density range rather well.

    Taking the paramagnetic part of the correlation energy (spin polarization = 0) and
    replacing the electron density ρ(r) with the density matrix D(r), the multistate extension
    of the Chachiyo functional (Eqn.8 in [Chachiyo]) can be written in the following form:

        C[D(r)] = a ∫ log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³ ) D(r) dr

                = ∫ CED(r) dr

    with

        a = (log(2)-1)/(2 π²) = -0.01554534543482745
        b = 20.4562557 (paramagnetic)
        b₁ = (4π/3)¹ᐟ³ b = 32.975319597703546
        b₂ = (4π/3)²ᐟ³ b = 53.155949872619715

    `f[D] = a log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³) D(r)` is a matrix funtional,
    which is calculated by diagonalizing D and applying the function f to the eigenvalues.

    References
    ----------
    [Chachiyo] T. Chachiyo (2016), J. Chem. Phys. 145, 2
        "Communication: Simple and accurate uniform electron gas correlation energy for the full range of densities"
    [Ceperley&Alder] D. Ceperley, B. Alder (1980), Phys. Rev. Lett., 45, 7, 566.
        "Ground state of the electron gas by a stochastic method"

    :param matrix_density: matrix density summed over spins, Dᵢⱼ = Dᵅᵢⱼ(r) + Dᵝᵢⱼ(r)
        matrix_density[...,i,j] = Dᵢⱼ
    :type matrix_density: Tensor of shape (...,n,n)

    :param spin: The spin parameter determines whether the paramagnetic (spin=0)
        or ferromagnetic (spin=1) correlation energy is calculated.
    :type spin: int

    :return: Electron correlation energy density CED
    :rtype: Tensor of same shape as input `matrix_density`.
    """
    return MatrixFunction.apply(_LDACorrelationChachiyo, matrix_density, spin)
