# -*- coding: utf-8 -*-
import numpy
import numpy.linalg as la

def numerical_hessian_G(grad,x0,h=1.0e-8):
    """
    compute Hessian of a function by numerical differentiation of gradients

    Parameters:
    ===========
    grad: grad(x) computes the gradient at position x
    """
    n = len(x0)
    hessian = numpy.zeros((n,n))
    for i in range(0, n):
        # unit vector in the i-th direction
        ei = numpy.zeros(n)
        ei[i] = 1.0
        # symmetric finite difference
        hessian[i,:] = (grad(x0+h*ei) - grad(x0-h*ei))/(2.0*h)
    # hessian should be symmetric
    hessian = 0.5 * (hessian + hessian.transpose())
    return hessian

def condition_number(A):
    """
    computes the condition number of a matrix A
    """
    Ainv = la.inv(A)
    k = la.norm(Ainv)*la.norm(A)
    return k
