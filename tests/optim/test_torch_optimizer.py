# -*- coding: utf-8 -*-
import torch.nn
import torch.testing
import unittest

from mlmsdft.optim.torch_optimizer import WrappedOptimizer

class RosenbrockFunction(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.x = torch.nn.Parameter(data=torch.tensor([0.0, 0.0], dtype=torch.double), requires_grad=True)

    def forward(self, a=1.0, b=100.0):
        f = (a-self.x[0])**2 + b*(self.x[1]-self.x[0]**2)**2
        return f


class TestTorchOptimizer(unittest.TestCase):
    def check_minimize_Rosenbrock_function(self, **optimizer_kwds):
        rosenbrock = RosenbrockFunction()

        def closure():
            rosenbrock.zero_grad()
            # evaluate function
            f = rosenbrock.forward()
            return f

        optimizer = WrappedOptimizer(rosenbrock.parameters(), **optimizer_kwds)
        # minimizer function
        optimizer.step(closure)
        f = closure()

        # The Rosenbrock function f(x,y) = (a-x)^2 + b (y-x^2)^2
        # has a global minimum at (a,a^2)
        a = 1.0
        x_min_ref = torch.tensor([a,a**2], dtype=torch.double)
        # Check if positions of minimum is found correctly.
        torch.testing.assert_close(rosenbrock.x, x_min_ref, rtol=1.0e-4, atol=1.0e-4)
        # Check if function value is correct.
        torch.testing.assert_close(f.item(), 0.0, rtol=1.0e-4, atol=1.0e-4)

    def test_minimize_Rosenbrock_function(self):
        # ... with Newton (using exact Hessian)
        with self.subTest("Newton with Armijo line search"):
            self.check_minimize_Rosenbrock_function(method="Newton", line_search_method="Armijo")
        with self.subTest("Newton with Wolfe line search"):
            self.check_minimize_Rosenbrock_function(method="Newton", line_search_method="Wolfe")

        # ... with BFGS approximation to inverse Hessian
        with self.subTest("BFGS with Armijo line search"):
            self.check_minimize_Rosenbrock_function(method="BFGS", line_search_method="Armijo")
        with self.subTest("BFGS with Wolfe line search"):
            self.check_minimize_Rosenbrock_function(method="BFGS", line_search_method="Wolfe")

        # ... with Steepest Descent
        with self.subTest("Steepest Descent"):
            self.check_minimize_Rosenbrock_function(method="Steepest Descent", gtol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
