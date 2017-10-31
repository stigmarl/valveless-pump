import numpy as np 

def driving_conditions_mr(dt, f, psi_ca, z_array, L, D):
    """
    Returns an array of the capillary surface vibration for the tissue matrix.

    Only calculates in r direction.

    Parameters
    ----------
    dt: scalar 
        Size of one timestep, [s].
    f: scalar: 
        The frequency of the driving force.
    psi_ca: scalar
        Diameter oscillation amplitude for the bubble, [m].
    z: scalar
        Position along z axis
    L: scalar
        Length of the along bubble the z axis, [m].
    D: scalar
        Length of transition region on the bubble, [m]

    Returns
    -------
    surface_vibrations: ndarray
        Array of capillary for tissue matrix surface vibrations.

        Each row signify values at a certain timestep.
    """

    T = 1/f                         # time of one period
    Np = int(round(T/float(dt)))    # number of timesteps in one period

    omega = 2*np.pi*f

    surface_vibrations_mr = np.zeros(Np, z_array.shape[0])

    vectorized_amplitude = np.vectorize(_psi_c_amplitude)

    for i in range(Np):
        surface_vibrations_mr[i,:] = vectorized_amplitude(psi_ca, z_array, L, D)*np.sin(omega*i*dt)

    return surface_vibrations_mr


    
def driving_conditions_fr(dt, f, psi_ca, z_array, L, D):
    """
    Returns an array of the capillary surface vibration for the interstitial fluid.

    Only calculates in r direction.

    Parameters
    ----------
    dt: scalar 
        Size of one timestep, [s].
    f: scalar: 
        The frequency of the driving force.
    psi_ca: scalar
        Diameter oscillation amplitude for the bubble, [m].
    z: scalar
        Position along z axis
    L: scalar
        Length of the along bubble the z axis, [m].
    D: scalar
        Length of transition region on the bubble, [m]

    Returns
    -------
    surface_vibrations: ndarray
        Array of capillary for tissue matrix surface vibrations.

        Each row signify values at a certain timestep.
    """

    T = 1/f                         # time of one period
    Np = int(round(T/float(dt)))    # number of timesteps in one period

    omega = 2*np.pi*f

    surface_vibrations_fr = np.zeros(Np, z_array.shape[0])

    vectorized_amplitude = np.vectorize(_psi_c_amplitude)

    for i in range(Np):
        surface_vibrations_fr[i,:] = vectorized_amplitude(psi_ca, z_array, L, D)*omega*np.cos(omega*i*dt)

    return surface_vibrations_fr
    


def _psi_c_amplitude(psi_ca, z, L, D):
    """
    Helper function that returns the amplitude of the capillary surface vibrations at a specific position.

    Parameters
    ----------
    psi_ca: scalar
        Diameter oscillation amplitude for the bubble, [m].
    z: scalar
        Position along z axis
    L: scalar   
        Length of the along bubble the z axis, [m].
    D: scalar
        Length of transition region on the bubble, [m]
    
    Returns
    -------


    """

    if z <= L/2 - D: 
        return psi_ca
    elif z > L/2 - D and z < L/2:
        return psi_ca/2*(1-np.cos(np.pi*(z-L/2)/D))
    else:
        return 0
        
    

    