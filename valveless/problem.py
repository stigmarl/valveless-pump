import numpy as np 


class Problem(object):

    """
    This class encapsulates the relevant physical parameters of the problem into an object.


    """

    def __init__(self, L, D, f, psi_ca, a_0, alpha_m, rho_t, mu_m, eta_f, gamma):
        """
        Initializes and sets the various physical parameters.

        Parameters
        ----------
        L: scalar
            Length of the along bubble the z axis [mm]
        D: scalar
            Length of transition region [mm]
        f: scalar
            Frequency of the driving wave [kHz]
        psi_ca: scalar
            Diameter oscillation amplitude for the bubble [mm]
        a_0: scalar
            Radius of gas bubble [mm]
        alpha_m: scalar
            Relative vole for tissue matrix []
        rho_t: scalar
            Tissue mass density, including tissue matrix and fluid [kg/cm³]
        mu_m: scalar
            Shear stiffness the tissue matrix [kg/(m*s)] = [Pa*s]
        eta_f: scalar
            Dynamic viscosity of the fluid [Pa*s
        gamma: scalar
            Viscous friction between fluid and tissue matrix.
        """

        self.L = L
        self.D = D
        self.f = f 
        self.psi_ca = psi_ca 
        self.a_0 = a_0
        self.alpha_m = alpha_m
        self.alpha_f = 1- self.alpha_m
        self.rho_t = rho_t 
        self.mu_m = mu_m
        self.eta_f = eta_f
        self.gamma = gamma


        self.rho_m = self.alpha_m*self.rho_t
        self.rho_f = self.alpha_m*self.rho_t


    def _psi_c_amplitude(self, z):
        """
        Helper function that returns the amplitude of the capillary surface vibrations at a specific position.

        Parameters
        ----------
        z: scalar
            Position along z axis
        Returns
        -------
        scalar:
            The amplitude at a position z > 0.

        """
        if z <= self.L/2 - self.D: 
            return self.psi_ca
        elif z > self.L/2 - self.D and z < self.L/2:
            return self.psi_ca/2*(1-np.cos(np.pi*(z-self.L/2)/self.D))
        else:
            return 0

    def vec_psi_c_amplitude(self, z_array):
        """
        Vectorized version of _psi_c_amplitude.
    
        Parameters
        ----------
        z_array: ndarray
            1D array of positions along the z axis.
        Returns
        -------
        ndarray
            1D array of amplitudes.
        """
        
        vec_psi_z = np.vectorize(self._psi_c_amplitude)

        return vec_psi_z(z_array)   


    