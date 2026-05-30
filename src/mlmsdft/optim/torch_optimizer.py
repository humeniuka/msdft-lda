
# -*- coding: utf-8 -*-
""" torch wrapper around minimize(...) function"""
import numpy as np

import torch

from mlmsdft.optim.minimize import minimize

def parameters_to_vector(parameters):
    """ concatenate parameter data into a numpy vector """
    x_list = []
    for param in parameters:
        data_flat = torch.reshape(param.data, (-1,)).numpy(force=True)
        x_list.append(data_flat)

    x = np.hstack(x_list)
    return x

def parameter_gradients(parameters):
    """ concatenate gradients of parameters into a numpy vector """
    grad_list = []
    for param in parameters:
        dfdx_flat = torch.reshape(param.grad, (-1,)).numpy(force=True)
        grad_list.append(dfdx_flat)

    dfdx = np.hstack(grad_list)
    return dfdx

def vector_to_parameters(x, parameters):
    """ replace parameter data with the values from the numpy array x """
    offset = 0
    for param in parameters:
        size = param.numel()
        data = torch.reshape(torch.tensor(x[offset:offset+size]), param.size())
        param.data.copy_(data)
        offset += size

class WrappedOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            method="BFGS",
            line_search_method="Wolfe",
            constraints=None, max_steplen=None,
            callback=None, maxiter=100000,
            gtol=1.0e-6, ftol=1.0e-8,
            debug=0):
        defaults = dict(
            method=method,
            line_search_method=line_search_method,
            constraints=constraints,
            max_steplen=max_steplen,
            callback=callback,
            maxiter=maxiter,
            gtol=gtol,
            ftol=ftol,
            debug=debug)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Optimizer doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']

    def step(self, closure):

        # wrapper around pytorch module that has the interface required by
        # `minimize()`
        def objfunc(x, requires_grad=True):
            # set parameters
            vector_to_parameters(x, self._params)
            # compute the objective function and its gradients through
            # backpropagation, if required
            f = closure()
            if requires_grad:
                f.backward()

            if requires_grad:
                dfdx = parameter_gradients(self._params)
                return f.item(), dfdx
            else:
                return f.item()

        # initial point
        x0 = parameters_to_vector(self._params)

        # run the BFGS minimization
        res = minimize(objfunc, x0, **self.defaults)

        vector_to_parameters(res.x, self._params)
