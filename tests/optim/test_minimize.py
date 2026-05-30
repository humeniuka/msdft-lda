# -*- coding: utf-8 -*-
import numpy
import numpy.testing
import unittest

from mlmsdft.optim.minimize import minimize

# (constrained) Rosenbrock's function

def rosenbrock(x, a=1.0, b=100.0, requires_grad=True):
    f = (a-x[0])**2 + b*(x[1]-x[0]**2)**2
    if requires_grad:
        dfdx = 0*x
        dfdx[0] = -2*(a-x[0]) - 4*b*x[0]*(x[1]-x[0]**2)
        dfdx[1] = 2*b*(x[1]-x[0]**2)

    if requires_grad:
        return f, dfdx
    else:
        return f

def rosenbrock_constraints(x, a=1.0, b=100.0):
    """
    constraint
       x1^2 + x2^2 < 1
    and its gradients
    """
    c1 = 1.0 - x[0]**2 - x[1]**2
    C = numpy.array([c1])
    A = numpy.array([[-2*x[0], -2*x[1]]])
    return C, A

class TestMinimize(unittest.TestCase):
    def test_minimize_Rosenbrock_function(self):
        # Starting point for optimization
        x0 = numpy.array([0.0, 0.0])

        # minimize Rosenbrock's function without constraints
        res = minimize(rosenbrock, x0)
        # The Rosenbrock function f(x,y) = (a-x)^2 + b (y-x^2)^2
        # has a global minimum at (a,a^2)
        a = 1.0
        # Check if positions of minimum is found correctly.
        numpy.testing.assert_almost_equal(res.x, numpy.array([a,a**2]))
        # Check if function value is correct.
        numpy.testing.assert_almost_equal(res.fun, 0.0)

    def test_minimize_constrained_Rosenbrock_function(self):
        # minimize Rosenbrock's function subject to  c1 = 1 - x1^2 - x2^2 > 0

        # Exact minimum of constrained minimization
        # see https://www.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html
        x_min = numpy.array([0.7864, 0.6177])
        # Exact function value at constrained minimum
        fun_min = 0.0457

        # Starting point for optimization
        x0 = numpy.array([0.0, 0.0])

        # ... with Newton (using exact Hessian)
        with self.subTest("Newton with Armijo line search"):
            res = minimize(
                rosenbrock, x0,
                method="Newton", line_search_method="Armijo", constraints=rosenbrock_constraints)
            numpy.testing.assert_almost_equal(res.x, x_min, decimal=4)
            numpy.testing.assert_almost_equal(res.fun, fun_min, decimal=4)
        with self.subTest("Newton with Wolfe line search"):
            res = minimize(
                rosenbrock, x0,
                method="Newton", line_search_method="Wolfe", constraints=rosenbrock_constraints)
            numpy.testing.assert_almost_equal(res.x, x_min, decimal=4)
            numpy.testing.assert_almost_equal(res.fun, fun_min, decimal=4)

        # ... with BFGS approximation to inverse Hessian
        with self.subTest("BFGS with Armijo line search"):
            res = minimize(
                rosenbrock, x0,
                method="BFGS", line_search_method="Armijo", constraints=rosenbrock_constraints)
            numpy.testing.assert_almost_equal(res.x, x_min, decimal=4)
            numpy.testing.assert_almost_equal(res.fun, fun_min, decimal=4)
        with self.subTest("BFGS with Wolfe line search"):
            res = minimize(
                rosenbrock, x0,
                method="BFGS", line_search_method="Wolfe", constraints=rosenbrock_constraints)
            numpy.testing.assert_almost_equal(res.x, x_min, decimal=4)
            numpy.testing.assert_almost_equal(res.fun, fun_min, decimal=4)

        # ... with Steepest Descent
        with self.subTest("Steepest Descent"):
            res = minimize(
                rosenbrock, x0,
                method="Steepest Descent", constraints=rosenbrock_constraints, gtol=1.0e-7)
            numpy.testing.assert_almost_equal(res.x, x_min, decimal=4)
            numpy.testing.assert_almost_equal(res.fun, fun_min, decimal=4)


if __name__ == "__main__":
    unittest.main()