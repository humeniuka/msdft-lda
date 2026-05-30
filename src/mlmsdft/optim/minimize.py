# -*- coding: utf-8 -*-
"""
solve the following optimization problem:

  minimize f(x)      subject to  c_i(x) > 0   for  i=1,...,m

where f(x) is a scalar function, x is a real vector of size n, and c_i(x) are the
m strict inequality constraints. The feasible space {x|C(x) > 0} is assumed to be convex.
The constraints are enforced by minimizing an auxiliary function f(x)+nu*B(x).
B(x) is the log-barrier

  B(x) = - sum_i log(c_i(x))

and `nu` is a small adjustable number.

References
----------
 [1] J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006
"""
import numpy as np
import numpy.linalg as la

from mlmsdft.optim import finite_differences


def line_search_backtracking(xk, fk, grad_fk, pk, func,
                             constraints=None,
                             a0=1.0, rho=0.3, c=0.0001, lmax=100):
    """
    perform a line search along the search direction pk using the Armijo backtracking algorithm

    see Algorithm 3.1 in Ref. [1]

    Parameters
    ----------
    xk: vector, current point
    fk: scalar, current value of objective function
    grad_fk: vector, current gradient
    pk: vector, search direction
    func: callable, computes function value and gradient,
            (fx, dfdx) = func(x, requires_grad=True)
        or only function values
            fx = func(x, requires_grad=False)
        depending on the keyword argument `requires_grad`.

    Optional
    --------
    constraints: callable, constraints(x) should return a vector `C` with the values
              of the constraints at x and matrix `A` whose rows are the gradients
              of the constraints w/r/t to x
    lmax: maximum number of tries before giving up

    The meaning of `a0`, `rho` and `c` can be glanced from the algorithm description in Ref.[1]

    Returns
    -------
    xkp1: next point
    """
    a = a0
    # directional derivative
    df = np.dot(grad_fk, pk)
    # check that pk is a descent direction
    assert df <= 0.0, "pk=%s not a descent direction" % pk

    for l in range(0, lmax):
        x_interp = xk + a*pk
        # choose the step small enough so that no constraints are violated
        if constraints is not None:
            C, A = constraints(x_interp)
            if np.any(C <= 0.0):
                # inequality constraints C > 0 were violated, a call to func() would
                # result in an error, reduce step size
                a *= rho
                continue

        if func(x_interp) <= fk + c*a*df:
            break
        else:
            a *= rho
    else:
        raise RuntimeError("Linesearch failed! Could not find a step length that satisfies the sufficient decrease condition.")
    return x_interp

def line_search_wolfe(xk, fk, grad_fk, pk, func,
                      constraints=None, max_steplen=None,
                      a0=1.0, amax=50.0, c1=0.0001, c2=0.9,
                      lmax=100,
                      debug=0):
    """
    find step size `a`` that satisfies the strong Wolfe conditions:

      1) sufficient decrease condition   f(xk + a pk) <= f(xk) + c1 a ∇f(xk).pk

      2) curvature condition            |∇f(xk + a pk).pk| <= c2 |∇f(xk).pk|

    see Algorithm 3.5 and 3.6 in Ref. [1]

    Parameters
    ----------
    xk: vector, current point
    fk: scalar, current value of objective function
    grad_fk: vector, current gradient
    pk: vector, search direction
    func: callable, computes function value and gradient,
            (fx, dfdx) = func(x, requires_grad=True)
        or only function values
            fx = func(x, requires_grad=False)
        depending on the keyword argument `requires_grad`.

    Optional
    --------
    constraints: callable, constraints(x) should return a vector `C` with the values
              of the constraints at x and matrix `A` whose rows are the gradients
              of the constraints w/r/t to x
    max_steplen: callable, `max_steplen(x,v)` computes the largest step length `amax` such that
              x + amax*v  lies inside the feasible set or on its boundary.
    lmax: maximum number of tries before giving up

    For the meaning of `a0`, `amax`, `c1` and `c2` see the algorithm description in Ref.[1]

    Returns
    -------
    xkp1: next point
    """
    assert 0 < c1 < c2 < 1
    def s(a):
        """computes scalar function s: a -> f(xk + a*pk) and its derivative ds/da"""
        fx,dfdx = func(xk + a*pk)
        dsda = np.dot(dfdx, pk)
        return fx, dsda

    s0 = fk      # s(a=0.0)
    Ds0 = np.dot(grad_fk, pk)  # ds/da(a=0.0)

    def zoom(alo,ahi, slo,shi):
        """
        find a step length a that satisfies Wolfe's conditions by bisection inside in the interval [alo,ali]
        """
        for j in range(0, lmax):
            # evaluate s and s' at the midpoint of the search interval
            aj = 0.5*(alo+ahi)
            sj,Dsj = s(aj)

            if (sj > s0 + c1*aj*Ds0) or (sj >= slo):
                # sufficient decrease condition is not fulfilled
                ahi = aj
                shi = sj
            else:
                if abs(Dsj) <= c2*abs(Ds0):
                    # curvature condition met, we are done
                    aWolfe = aj
                    break
                if Dsj*(ahi-alo) >= 0.0:
                    ahi = alo
                    shi = slo #noqa: F841
                alo = aj
                slo = sj
        else:
            aWolfe = 0.5*(alo+ahi)
            if debug > 0:
                msg = "``zoom`` could not find a point satisfying Wolfe's condition in the interval [%e,%e] in %d iterations!" % (alo,ahi, j)
                print( "WARNING: %s" % msg )
        return aWolfe

    def feasible(a):
        """checks whether the point xk+a*pk is inside the feasible region"""
        if constraints is not None:
            C,A = constraints(xk + a*pk)
            if np.any(C <= 0.0):
                # inequality constraints C > 0 are violated
                return False
        return True

    # Find the largest feasible step length.
    if max_steplen is not None:
        _amax = max_steplen(xk, pk)
        if _amax == np.inf:
            # If there is no limit on the maximum step length, we leave the default value `amax` untouched
            pass
        else:
            # override default value with maximum step length
            amax = _amax
            # any step length < `amax` should lie in the feasible region,
            # while any step length > `amax` should lie outside
            #assert feasible(0.999*amax) == True
            #assert feasible(1.001*amax) == False
    # Since the feasible region is convex, all step lengths `a` < `amax`
    # will be feasible as well, if amax is feasible.
    else:
        # If no function was provided for selecting the maximum step length,
        # we have to ensure that `amax` at least does not lie outside the feasible set.
        while not feasible(amax):
            # decrease `amax` until it lies in the feasible region
            amax *= 0.99
    # The initial guess for the step length should satisfy `a0` < `amax`.
    if a0 >= amax:
        a0 = 0.5*amax

    # Algorithm 3.5, brackets the interval
    aim1 = 0.0
    sim1 = s0
    ai = a0
    for i in range(1, lmax):
        si,Dsi = s(ai)
        if (si > s0 + c1*ai*Ds0) or ((si >= sim1) and i > 1):
            # sufficient decrease condition is not fulfilled => a minimum has to lie
            # in between around, which the Wolfe conditions hold.
            aWolfe = zoom(aim1,ai, sim1,si)
            break
        if abs(Dsi) <= c2*abs(Ds0):
            # curvature condition is fulfilled
            aWolfe = ai
            break
        if Dsi >= 0.0:
            # derivative s'(a) changed sign => a minimum has to lie in between
            aWolfe = zoom(ai,aim1, si,sim1)
            break
        aim1 = ai
        sim1 = si
        # choose a new a_(i+1) from the interval (ai,amax)
        ai = 2*aim1
        if ai >= amax:
            if debug > 0:
                print( "WARNING: end of search interval reached!" )
            # end of interval reached when multiplying ai by 2, approach amax from below
            ai = 0.5*(aim1 + amax)
    else:
        raise RuntimeError("Linesearch failed! Could not find a step length that satisfies Wolfe's conditions.")
    x_Wolfe = xk + aWolfe*pk
    return x_Wolfe

def bfgs_update(invHk, sk, yk, k):
    """
    update the inverse Hessian invH_(k+1) based on Algorithm 6.1 in Ref.[1]

    Parameters:
    -----------
    invHk: inverse Hessian approximation
    yk: gradient difference vector, yk= ∇f_(k+1) - ∇f_k
    sk: step vector,  x_(k+1) - x_k
    k: integer counting the number of iterations

    Returns
    -------
    invH_(k+1): next inverse Hessian approximation
    """
    n = len(sk)
    Id = np.eye(n)
    assert k >= 1
    if k == 1:
        invHkp1 = np.dot(yk,sk)/np.dot(yk,yk) * Id
    else:
        rk = 1.0/np.dot(yk,sk)
        U = Id - rk*np.outer(sk,yk)
        V = Id - rk*np.outer(yk,sk)
        W = rk*np.outer(sk,sk)

        invHkp1 = np.dot(U, np.dot(invHk, V)) + W
    return invHkp1

def log_barrier(C, A):
    """
    The log-barrier prevents the parameter vector from leaving the feasible area, where the constraints
    are fulfilled by adding a repulsive barrier B(x) to the objective function f(x):

      B(x) = - sum_i log(C_i(x))

    The gradient of the barrier becomes

      dB/dxj = - sum_i 1/C_i(x) dC_i/dxj = - sum_i 1/C_i A_ij

    The auxiliary function, that is minimized instead of f(x), is now

      f(x) + nu*B(x)

    where ``nu`` is a small adjustable number.

    Parameters
    ----------
    C: vector with values of constraints C_i(x), i=1,...,m
    A: (m x n) matrix, where A[i,j] contains the derivative of the i-th constraint w/r/t the j-th variable

    Returns
    -------
    B: value of barrier function
    dB: gradient of barrier function
    """
    m = len(C)   # number of constraints
    B = 0.0
    for i in range(0, m):
        B -= np.log(C[i])
    # gradient
    dB = - np.dot(1.0/C, A)

    return B, dB


class OptimizationResult:
    def __init__(self, x, fun, grad, nit):
        self.x = x
        self.fun = fun
        self.grad = grad
        self.nit = nit


def minimize(objfunc, x0,
             method="BFGS",
             line_search_method="Wolfe",
             constraints=None, max_steplen=None,
             callback=None, maxiter=100000,
             gtol=1.0e-6, ftol=1.0e-8,
             debug=0):
    """
    minimize a scalar function ``objfunc``(x) possibly subject to constraints.

    The minimization is converged if
      * |df/dx| < gtol and
      * |f(k+1)-f(k)| < ftol

    Parameters
    ----------
    objfunc: callable, objective function that should be minimized.
        In addition to x it should take a keyword argument `requires_grad`.
        It should return the function value and gradient,

            fx, dfdx = objfunc(x, requires_grad=True)

        or only the function value

            fx = objfunc(x, requires_grad=False)

        depending on the value of `requires_grad`. During line-searches gradients
        are not required.

    x0: initial point, where the optimization starts

    Optional
    --------
    method: choose how the search direction should be determined,
       'Newton'           - the Hessian H is calculated by numerically differentiating the gradient,
                            the search direction is then obtained by solving  H.p = -df/dx
       'Steepest Descent' - the search direction is antiparallel to the gradient, p = -df/dx
       'BFGS'             - an approximation to the inverse Hessian invH is updated after each step
                            so as to track the curvature along the path, then p = -invH.df/dx
    line_search_method: choose how the step length in the search direction should be determined
       'Armijo'           - starting with a=1, the algorithm backtracks until sufficient decrease is obtained
       'Wolfe'            - attempts to find a step length that satisfies Wolfe's conditions
    constraints: callable, constraints(x) should return a vector C with the values
              of the constraints at x and matrix A whose rows are the gradients
              of the constraints w/r/t to x
    max_steplen: callable, `max_steplen(x,v)` computes the largest step length `amax` such that
              x + amax*v  lies inside the feasible set or on its boundary.
    callback: callable, at the end of each iteration this function is called with the current vector x
              as argument
    maxiter: maximum number of iterations
    gtol: tolerance for norm of gradient
    ftol: tolerance for change of function value

    Returns
    -------
    res: instance of OptimizationResult

    Notes
    -----
    The 'BFGS' algorithm should be combined with a 'Wolfe' line search. The curvature condition is important
    in a quasi-Newton method, because it ensures that the approximation to the Hessian remains positive definite
    after each update.
    """
    assert ftol > 0.0, "Tolerance for change of function value has to be > 0.0"
    assert gtol > 0.0, "Tolerance for norm of gradient has to be > 0.0"
    assert method in ["Newton", "Steepest Descent", "BFGS"]
    assert line_search_method in ["Armijo", "Wolfe"]
    n = len(x0)
    def barrier(x):
        nu = 1.0e-5  #0.001 #0.00000001 #0.001
        if constraints is not None:
            C,A = constraints(x)
            # add log-barrier
            Bx,dBdx = log_barrier(C,A)
        else:
            Bx = 0.0
            dBdx = np.zeros(n)
        Bx *= nu
        dBdx *= nu
        return Bx, dBdx
    def func(x):
        fx = objfunc(x, requires_grad=False)
        Bx,dBdx = barrier(x)
        return fx+Bx
    def grad(x):
        fx,dfdx = objfunc(x, requires_grad=True)
        Bx,dBdx = barrier(x)
        return dfdx+dBdx
    def func_grad(x):
        fx,dfdx = objfunc(x, requires_grad=True)
        Bx,dBdx = barrier(x)
        return fx+Bx, dfdx+dBdx
    def hess(x):
        H = finite_differences.numerical_hessian_G(grad, x)
        #print( "condition number k=%e" % finite_differences.condition_number(H) )
        return H

    xk = x0
    # sk and yk is set inside the for loop
    sk = None
    yk = None

    fk, grad_fk = func_grad(xk)
    converged = False
    # smallest representable positive number such that 1.0+eps != 1.0.
    epsilon = np.finfo(float).eps
    for k in range(0, maxiter):
        # determine new search direction
        if method == "Newton":
            # compute exact hessian numerically
            Hk = hess(xk)
            # make Hk sufficiently positive definite
            Bk, tau_k = modified_cholesky(Hk)
            # search direction
            pk = la.solve(Bk, -grad_fk)
        elif method == "Steepest Descent":
            pk = -grad_fk    # steepest descent
        elif method == "BFGS":
            if k == 0:
                invHk = np.eye(n)
            else:
                if np.dot(yk,sk) <= 0.0:
                    if debug > 0:
                        print( "WARNING: positive definiteness of Hessian approximation lost in BFGS update, since yk.sk <= 0!" )
                invHk = bfgs_update(invHk, sk, yk, k)
            pk = np.dot(invHk,-grad_fk)
        # determine next point by a line search
        if line_search_method == "Armijo":
            x_kp1 = line_search_backtracking(xk, fk, grad_fk, pk, func, constraints=constraints)
        elif line_search_method == "Wolfe":
            # Quasi-Newton methods may fail to work well with backtracking line search, since
            # the curvature condition might not be fulfilled. In this case tiny steps are taken
            # along the descent direction, although a single larger step could be taken.
            x_kp1 = line_search_wolfe(xk, fk, grad_fk, pk, func_grad,
                                      constraints=constraints, max_steplen=max_steplen,
                                      debug=debug)
        f_kp1, grad_f_kp1 = func_grad(x_kp1)
        # compute change of function value from step k to the next and norm of the gradient
        f_change = abs(f_kp1 - fk)
        gnorm = la.norm(grad_f_kp1)
        if f_change < ftol and gnorm < gtol:
            converged = True
        if f_change < epsilon:
            # f(k+1) and f(k) cannot be distinguished properly because of finite numerical precision
            if debug > 0:
                print( "WARNING: |f(k+1) - f(k)| < epsilon  (numerical precision) !" )
            converged = True
        # step vector
        sk = x_kp1 - xk
        # gradient difference vector
        yk = grad_f_kp1 - grad_fk
        # new variables for step k become old ones for step k+1
        xk = x_kp1
        fk = f_kp1
        grad_fk = grad_f_kp1
        if callback is not None:
            callback(xk)
        if debug > 0:
            # Show a little tick behind each term if it is converged
            # ... change in function value
            if f_change < ftol:
                ftol_status = r"✓"
            else:
                ftol_status = r" "
            # ... gradient norm
            if gnorm < gtol:
                gtol_status = r"✓"
            else:
                gtol_status = r" "

            print( "k=%10.1d  f(x) = %15.10f  |x(k+1)-x(k)| = %e  |f(k+1)-f(k)| = %e %s |df/dx| = %e %s" % (
                k, fk, la.norm(sk), f_change, ftol_status, gnorm, gtol_status) )
        if converged:
            break
    else:
        raise RuntimeError("No convergence in %s method after %d iterations!" % (method, k+1))
    return OptimizationResult(xk, fk, grad_fk, k)


def modified_cholesky(A, beta=0.01):
    """
    make the matrix A positive definite by adding a small positive multiple of the identity:
      A -> A + tau*Id

    see Algorithm 3.3 in Ref. [1]
    """
    n,n = A.shape

    min_diag = np.diag(A).min()
    if min_diag > 0.0:
        tau_0 = 0.0
    else:
        tau_0 = -min_diag + beta
    tau_k = tau_0
    Id = np.eye(n)
    while True:
        try:
            la.cholesky(A + tau_k * Id)
        except la.LinAlgError:
            tau_k = max(2*tau_k, beta)
            continue
        break
    Apos = A + tau_k*Id
    return Apos, tau_k
