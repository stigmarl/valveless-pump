import numpy as np 


class Problem(object):
    def __init__(self, L, D, f, psi_ca, a_0, alpha_m, rho_t, mu_m, eta_f, gamma):
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

        # setup mass densities
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
        The amplitude at a position z > 0.

        """
        if z <= self.L/2 - self.D: 
            return self.psi_ca
        elif z > self.L/2 - self.D and z < self.L/2:
            return self.psi_ca/2*(1-np.cos(np.pi*(z-self.L/2)/self.D))
        else:
            return 0

    def _vec_psi_c_amplitude(self, z_array): 
        """
        Vectorized version of _psi_c_amplitude.
    
        Parameters
        ----------
        z_array: ndarray
            1D array of positions along the z axis.
        Returns
        -------
        1D array of amplitudes.
        """
        
        vec_psi_z = np.vectorize(self._psi_c_amplitude)

        return vec_psi_z(z_array)   


    