# -*- coding: utf-8 -*-
import numpy
import numpy.testing
import unittest

from mlmsdft.optim.finite_differences import numerical_hessian_G

class TestFiniteDifferences(unittest.TestCase):
    def test_numerical_hessian_G(self):
        """compare numerical Hessian with exact one for test function"""
        # f(x,y) = x^3 + y^2 + x*y
        def func(x):
            return x[0]**3 + x[1]**2 + x[0]*x[1]
        def grad(x):
            return numpy.array([3*x[0]**2 + x[1], 2*x[1] + x[0]])
        def hess(x):
            return numpy.array([
                [6*x[0], 1.0],
                [1.0, 2.0]
            ])
        x0 = numpy.array([1.234, -0.245])
        # compute Hessian of f(x,y) from finite differences of its gradient
        hessian_from_grad = numerical_hessian_G(grad, x0)
        hessian_ref = hess(x0)

        numpy.testing.assert_almost_equal(hessian_from_grad, hessian_ref)


if __name__ == "__main__":
    unittest.main()