# coding: utf-8
"""Matrix functionals."""
from abc import ABC, abstractmethod

import torch
import torch.linalg
from torch import Size, Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

__all__ = [
    "trace_average"
]


class ScalarFunction(ABC):
    """
    Abstract base class.

    All derived matrix functions have to implement f(x) and f'(x).
    """
    @staticmethod
    @abstractmethod
    def value(x: Tensor, *other_args) -> Tensor:
        """
        f(x)
        Implementation of the scalar function f(x).
        The matrix function F(X) has the same Taylor expansion as f.
        """
        pass

    @staticmethod
    @abstractmethod
    def derivative1(x: Tensor, *other_args) -> Tensor:
        """
        f'(x)
        Implementation of the first derivative f'(x) = df/dx of the scalar function f(x).
        """
        pass


def check_dimensions_(X: Tensor, name: str) -> None:
    """
    Check that the input tensor `X` can be interpreted as a batch of square
    matrices. The last 2 dimensions have to be the same.
    The `name` is used to refer to the tensor in the error message.
    """
    size = X.size()
    # 1) X has to have at least 2 dimensions
    # 2) The last two dimensions have to be the same so that X[...,i,j]
    # can be interpreted as the (i,j) element of a batch of n x n
    # matrices.
    if not (len(size) >= 2 and size[-1] == size[-2]):
        raise ValueError(
            f"The tensor {name} has to be of size (...,n,n) "
            f"so that it can be intepreted as a batch of n x n matrices.\n"
            f"The given input of size {size} does not fit that pattern."
        )


class MatrixFunction(Function):
    @staticmethod
    def forward(ctx, function: ScalarFunction, X: Tensor, *other_args) -> Tensor:
        """
        Evaluate the analytic matrix function F(X).

        The analytic matrix function F(X) is defined by the scalar function f(x), which
        operates on the eigenvalues of X,

            F(t) = F(X(t)) = U.f(Λ).U(t)ᵀ

        NOTE: The input X is expected to be a batch of *symmetric* matrices,
        although this is not checked. If the input is not symmetric
        with respect to the exchange of the last two dimensions, results
        will be wrong and the consistency between forward and backward propagation
        is also broken.

        :param function: object for evaluating f(x) and f'(x)
        :type function: :class:`~.ScalarFunction`

        :param X: batch of symmetric matrices
        :type X: Tensor of size (...,n,n)

        :param other_args: other positional arguments that are passed to f(x)
        :type other_args: anything except Tensor

        :return: F
            F=f(X) is the value of the matrix function f at the argument X.
        :rtype: Tensor of size (...,n,n)
        """
        check_dimensions_(X, 'X')
        # Compute eigenvalues Λ and eigenvectors U of the symmetric
        # matrix X.
        with torch.no_grad():
            # We do not need the gradients of L and U. Anyway, if there are repeated eigenvalues
            # computing continuous eigenvalue derivatives would require higher order derivatives.
            L, U = torch.linalg.eigh(X)
            # Apply the scalar function to the eigenvalues, f(λₐ)
            fL = function.value(L, *other_args)
            # Compute the matrix function F(X)ᵢⱼ = ∑ₐ Uᵢₐ f(λₐ) Uⱼₐ
            F = torch.einsum('...ia,...a,...ja->...ij', U, fL, U)

        # Eigenvalues Λ, f(Λ) and eigenvectors are needed for evaluating the Jacobian
        # dF/dX during backpropagation.
        ctx.save_for_backward(L, fL, U)
        # Save function and other arguments for evaluating f'(x,**args).
        ctx.function = function
        ctx.other_args = other_args

        return F

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor) -> Tensor:
        """
        Compute the vector-Jacobian product Jᵀ·v needed for back-propagation.

        The analytic matrix function F(X) is defined by the scalar function f(x), which
        operates on the eigenvalues of X,

            F = F(X) = U.f(Λ).Uᵀ

        The gradient of a scalar function g = g(F(X)) with respect to X is given by

            ∂g/∂Xₖₗ = ∑ᵢ ∑ⱼ ∂Fᵢⱼ/∂Xₖₗ ∂g/∂Fᵢⱼ
                    = ∑ᵢ ∑ⱼ  Jᵢⱼ,ₖₗ vᵢⱼ                 (~ Jᵀ·v)
                    = ∑ₐ ∑ᵦ Uₖₐ (Yₐᵦ (∑ᵢ ∑ⱼ Uᵢₐ vᵢⱼ Uⱼᵦ)) Uₗᵦ

        where Y contains the eigenvalue derivatives

                    f'(λₐ)                if λₐ=λᵦ
            Yₐᵦ = {
                    [f(λₐ)-f(λᵦ)]/(λₐ-λᵦ)  if λₐ≠λᵦ


        :param grad_output: vector vᵢⱼ = ∂g/∂Fᵢⱼ, gradient with respect to outputs of F
        :type grad_output: Tensor of size (...,n,n)

        :return grad_input: ∂g/∂Xₖₗ gradient with respect to inputs of F
        :rtype grad_input: Tensor of size (....,n,n)

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Matrix_calculus
        """
        check_dimensions_(grad_output, 'grad_output')
        # Recover eigenvalues and eigenvectors computed in the forward pass.
        L, fL, U = ctx.saved_tensors
        # Scalar function f and any other arguments that might be needed by f'(x)
        function = ctx.function
        other_args = ctx.other_args

        # Construct matrix Yₐᵦ of eigenvalue derivatives.
        eigval_derivs = torch.zeros_like(U)
        # Eigenvalues are considered the same, if they differ by less than `epsilon`.
        epsilon = 1.0e-12
        # Loop over matrix dimensions
        n = grad_output.size(dim=-1)
        for a in range(0, n):
            La = L[...,a]
            fLa = fL[...,a]
            for b in range(0, n):
                Lb = L[...,b]
                fLb = fL[...,b]
                # Which eigenvalue pairs are the same?
                same = torch.abs(La - Lb) < epsilon

                # Yₐᵦ has size (...), without the last two dimensions (n,n).
                Yab = torch.zeros(U.size()[:-2], dtype=U.dtype, device=U.device)
                # Eigenvalues λₐ=λᵦ to within numerical precision.
                # To ensure that Y is symmetric, we compute
                # Yₐᵦ = f'(1/2(λₐ+λᵦ))
                # for the average of the two eigenvalues.
                Yab[same] = function.derivative1(0.5 * (La[same] + Lb[same]), *other_args)

                # Eigenvalues are different, λₐ≠λᵦ,
                # Yₐᵦ = [f(λₐ)-f(λᵦ)]/(λₐ-λᵦ)
                Yab[~same] = (fLa[~same]-fLb[~same])/(La[~same]-Lb[~same])

                eigval_derivs[...,a,b] = Yab

        # Transform vᵢⱼ into [UvU]ₐᵦ = ∑ᵢ ∑ⱼ Uᵢₐ vᵢⱼ Uⱼᵦ
        UvU = torch.einsum('...ia,...ij,...jb->...ab', U, grad_output, U)

        # Derivative of g(F(X)) w/r/t inputs of F.
        # ∂g/∂Xₖₗ = ∑ₐ ∑ᵦ Uₖₐ (Yₐᵦ UvUₐᵦ) Uₗᵦ
        grad_input = torch.einsum(
            '...ka,...ab,...lb->...kl', U, eigval_derivs * UvU, U)

        # For each argument we have to return the gradient w/r/t the
        # input or None, if it does not require gradients. For each
        # optional argument None has to be appended
        ret = [None, grad_input] + len(other_args)*[None]

        return tuple(ret)


# Example how to define a matrix function from an analytic function.
class Exp(ScalarFunction):
    @staticmethod
    def value(x: Tensor) -> Tensor:
        r""" f(x) = exp(x) """
        f = torch.exp(x)
        return f

    @staticmethod
    def derivative1(x) -> Tensor:
        r""" f'(x) = exp(x) """
        f_deriv1 = torch.exp(x)
        return f_deriv1

def exp(tensor):
    """ matrix eponential """
    return MatrixFunction.apply(Exp, tensor)


def trace_average(tensor, weights=None, subspace_dim=None):
    """
    compute the average of the diagonal elements in the matrix dimensions

        tr(X) = 1/n ∑ᵢ Xᵢᵢ

    Dividing by the matrix dimension n is important to ensure that the range
    of the output values is independent of the matrix dimension. Otherwise the
    range would increase linearly with n.

    If the subspace dimension m is specified, the tr(X) is the average over the
    lowest m eigenvalues λₐ of X,

        tr(X) = 1/m ∑ₐ λₐ   a=1,...,m

    Different diagonal elements can be weighted differently, if a vector of
    weights is provided. If the optional argument `weights` is not None, the
    weighted trace is calculated as

        trʷ(X) = 1/m ∑ᵢ wᵢ/(∑ⱼ wⱼ) Xᵢᵢ

    To incorporate the weights, we define a weighted matrix WX,

        WXᵢⱼ = wᵢ Xᵢⱼ such that Tr(WX) = ∑ᵢ wᵢ Xᵢᵢ,

    Then the weighted trace of X can be calculated from the eigenvalues ƛₐ of WX,

        trʷ(X) = 1/m ∑ₐ ƛₐ

    Finally it is possible to restrict the trace average to an `m`-dimensional
    subspace (m <= n) spanned by the lowest `m` eigenvectors.

        trʷₘ(X) = 1/m ∑ₐ ƛₐ  with j,a=1,...,m

    If m < n, the diagonal matrix of weights, Wᵢⱼ = wᵢ δᵢⱼ, must commute with X,
    X.W = W.X, so that both X and W have the same eigenvectors, X = U λ U⁻¹, W = U w U⁻¹,
    and

        trʷₘ(X) = tr(WX) = tr(U w U⁻¹ U λ U⁻¹) = ∑ₐ wₐ λₐ  for a = 1,...,m

    :param tensor: input tensor X, batch of matrices
    :type tensor: Tensor of size (...,n,n)

    :param weights: weight vector wᵢ
        The diagonal matrix Wᵢⱼ = wᵢ δᵢⱼ must commute with X
    :type weights: Tensor of size (n,) or None

    :param subspace_dim: Limit trace to the `m` lowest eigenvalues of X
        If `subspace_dim` is None, the trace extends over the whole n-dimensional space.
    :type subspace_dim: int m > 0 or None

    :return trX: batch of 1x1 matrices with the
        matrix dimensions traced out
    :type trX: Tensor of size (...,1,1)
    """
    check_dimensions_(tensor, 'X')
    # matrix dimension
    n = tensor.size()[-1]
    # Number of lowest eigenvalues that are included in the average
    if subspace_dim is None:
        # Include all eigenvalues
        m = n
    else:
        if not (0 < subspace_dim <= n):
            raise ValueError(
                "0 < `subspace_dim` <= n is the number of lowest eigenvalues to include.\n"
                f"n={n}, but subspace_dim={subspace_dim}"
            )
        m = subspace_dim

    # Weights wᵢ
    if weights is None:
        # All dimensions are weighted equally
        weights = torch.ones(n)
    else:
        # Check dimensions
        if weights.size() != Size([n]):
            raise ValueError(
                "Optional input tensor `weights` has to be of size (n,) "
                "with weights for each of the diagonal elements of an (n,n) matrix.\n"
                f"n={n}, but `weights` has size {weights.size()} ."
            )
        # Normalize weights such that ∑ⱼ wⱼ = n.
        weights = n * weights / torch.sum(weights)
    # Move weights to same device as input matrix X.
    weights = weights.to(dtype=tensor.dtype, device=tensor.device)

    # The trace average is computed from the eigenvalues of the weighted
    # input tensor,
    # WXᵢⱼ = wᵢ Xᵢⱼ
    WX = torch.einsum('i,...ij->...ij', weights, tensor)
    # diagonalize WX to get eigenvalues wₐ*λₐ in ascending order
    evals = torch.linalg.eigvalsh(WX)
    # NOTE: If there are degenerate eigenvalue, we would have to include all of them,
    # w(1)λ(1) < w(2)λ(2) < ... < w(m)λ(m) = w(m+1)λ(m+1) = ... = w(m+g)λ(m+g)
    # indices [0,1,...,m] of lowest eigenvalues
    indices = torch.arange(0, m).to(device=tensor.device)
    evals_subspace = torch.index_select(evals, -1, indices)

    # Sum over eigenvalues ∑ₐ wₐ λₐ  with a=1,...,m
    trWX = torch.sum(evals_subspace, dim=-1)

    # Normalize trace so that it does not depend on the number of (subspace) dimensions.
    # Trace average
    # trʷ(X) = 1/m ∑ₐ wₐ λₐ  with a=1,...,m
    trWX = 1.0/m * trWX

    # Insert two dimensions of size 1, (...) -> (...,1,1),
    # so that the output tensor is a batch of 1x1 matrices.
    trWX = torch.unsqueeze(torch.unsqueeze(trWX, -1), -1)

    return trWX
