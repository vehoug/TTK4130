import numpy as np

def newtons_method(t, x0, f, Jf, TOL=1e-6, MAX_ITER=100):
    """
    Newton's method for solving nonlinear equations.
    
    Args:
        t (float): Current time
        x0 (np.ndarray): Initial guess for the root
        f (Callable[[float, np.ndarray], np.ndarray]): Function for which we want to find the root
        Jf (Callable[[float, np.ndarray], np.ndarray]): Jacobian of the function f
        TOL (float, optional): Tolerance for convergence. Defaults to 1e-6.
        MAX_ITER (int, optional): Maximum number of iterations. Defaults to 100.
    """
    
    x = x0
    err = np.inf
    i = 0

    while err > TOL and i < MAX_ITER:
        G = f(t, x)
        J = Jf(t, x)
        dx = np.linalg.solve(J, -G).squeeze()
        x += dx
        err = np.linalg.norm(dx)
        i += 1
    
    return x

def implicit_euler_step(tn, xn, h, g, Jg):
    """
    Implicit Euler time stepper.
    
    Args:
        tn (float): Current time
        xn (np.ndarray): Current state
        h (float): Time step size
        g (Callable[[float, np.ndarray], np.ndarray]): Function defining the ODE
        Jg (Callable[[float, np.ndarray], np.ndarray]): Jacobian of the function g
    """

    x = xn.copy()

    res = lambda t, x: x - xn - h*g(t, x).squeeze()
    res_jac = lambda t, x: np.eye(len(x)) - h * Jg(t, x).squeeze()

    return newtons_method(tn+h, x, res, res_jac)

def implicit_midpoint_step(tn, xn, h, g, Jg):
    """
    Implicit Midpoint time stepper.
    
    Args:
        tn (float): Current time
        xn (np.ndarray): Current state
        h (float): Time step size
        g (Callable[[float, np.ndarray], np.ndarray]): Function defining the ODE
        Jg (Callable[[float, np.ndarray], np.ndarray]): Jacobian of the function g
    """
    
    x = xn.copy()

    res = lambda t, x: x - xn - h * g(t+h/2, (xn + x)/2).squeeze()
    res_jac = lambda t, x: np.eye(len(x)) - h/2 * Jg(t+h/2, (xn+x)/2).squeeze()
    return newtons_method(tn, x, res, res_jac)

    