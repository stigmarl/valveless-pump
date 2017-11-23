import numpy as np



def dtt(f, f_1, f_2, dt):
    """
    Second order central difference differentiation in time.

    Parameters
    ----------
    f: ndarray
        Variable at current timestep t.
    f_1: ndarray
        Variable at timestep t-dt.
    f_2: ndarray
        Variable at timestep t-2*dt.
    dt: scalar
        Timestep.

    Returns
    -------
    ndarray
        The differentiated variable.

    """
    return (f[1:-1,1:-1] - 2*f_1[1:-1,1:-1] + f_2[1:-1,1:-1])/(dt**2)


def dr_central(f, dr):
    """
    First order central difference differentiation.

    Parameters
    ----------
    f: ndarray
        The variable to be differentiated.
    dr: scalar
        Step size in r.

    Returns
    -------
    ndarray

    """

    return (f[2:, 1:-1] - f[:-2,1:-1])/(2*dr)


def dzz(f, dz):

    return (f[1:-1, 2:] - 2*f[1:-1,1:-1] + f[1:-1, :-2])/(dz**2)


def dt(f, f_1, dt):

    return (f[1:-1, 1:-1] - f_1[1:-1, 1:-1])/dt


def drr(f, dr):

    return (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2, 1:-1])/(dr**2)


def dz_central(f, dz):

    return (f[1:-1, 2:] - f[1:-1, :-2])/(2*dz)


def dr_forward(f, dr):

    return (f[2:, 1:-1] - f[1:-1, 1:-1])/(dr)


def dz_forward(f, dz):

    return (f[1:-1, 2:] - f[1:-1, 1:-1])/(dz)


def dr_backward(f, dr):

    return (f[1:-1, 1:-1] - f[0:-2, 1:-1])/(dr)


def dz_backward(f, dz):

    return (f[1:-1, 1:-1] - f[1:-1, 0:-2])/(dz)


def drz(f, dr, dz):

    return (f[2:, 2:] - f[:-2, 2:] - f[2:, :-2] + f[:-2, :-2])/(4*dr*dz)