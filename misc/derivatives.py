import numpy as np
from numba import *

@jit(float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64))
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

@jit(float64[:,:](float64[:,:],float64))
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


@jit(float64[:,:](float64[:,:],float64))
def dzz(f, dz):

    return (f[1:-1, 2:] - 2*f[1:-1,1:-1] + f[1:-1, :-2])/(dz**2)

@jit(float64[:,:](float64[:,:],float64[:,:],float64))
def dt(f, f_1, dt):

    return (f[1:-1, 1:-1] - f_1[1:-1, 1:-1])/dt

@jit(float64[:,:](float64[:,:],float64))
def drr(f, dr):

    return (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2, 1:-1])/(dr**2)

@jit(float64[:,:](float64[:,:],float64))
def dz_central(f, dz):

    return (f[1:-1, 2:] - f[1:-1, :-2])/(2*dz)

@jit(float64[:,:](float64[:,:],float64))
def dr_forward(f, dr):

    return (f[2:, 1:-1] - f[1:-1, 1:-1])/(dr)

@jit(float64[:,:](float64[:,:],float64))
def dz_forward(f, dz):

    return (f[1:-1, 2:] - f[1:-1, 1:-1])/(dz)

@jit(float64[:,:](float64[:,:],float64))
def dr_backward(f, dr):

    return (f[1:-1, 1:-1] - f[0:-2, 1:-1])/(dr)

@jit(float64[:,:](float64[:,:],float64))
def dz_backward(f, dz):

    return (f[1:-1, 1:-1] - f[1:-1, 0:-2])/(dz)

@jit(float64[:,:](float64[:,:],float64,float64))
def drz(f, dr, dz):

    return (f[2:, 2:] - f[:-2, 2:] - f[2:, :-2] + f[:-2, :-2])/(4*dr*dz)