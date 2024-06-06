import numpy as np

def lorenz96(x, F):
    """
    Lorenz 96 1D model with constant forcing.

    Args:
        x (np.ndarray): State vector.
        F (float): Forcing term.

    Returns:
        np.ndarray: Time derivative of the state vector x
    """
    N = len(x)  # Number of state vectors (x_N = x_0)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt


def lorenz96_rk4_step(x, F, dt):
    """
    Performs a single time step of the Lorenz 96 1D model using the 
    4th-order Runge-Kutta (RK4) integration method.
    the slopes (k values) are calculated using the lorenz96 function.

    Args:
        x (np.ndarray): State vector at the current time step.
        F (float): Forcing term.
        dt (float): Time step size.

    Returns:
        np.ndarray: State vector at the next time step.
    """
    k1 = lorenz96(x, F)
    k2 = lorenz96(x + dt / 2 * k1, F)
    k3 = lorenz96(x + dt / 2 * k2, F)
    k4 = lorenz96(x + dt * k3, F)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)