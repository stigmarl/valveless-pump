import numpy as np

import constants
from problem import driving_conditions_mr


class Solver(object):
    def __init__(self, prob, Lr, Lz, Nr, Nz, dt, Nc):
        self.prob = prob
        self.Lr = Lr
        self.Lz = Lz
        self.Nr = Nr
        self.Nz = Nz
        self.dt = dt 
        self.Nc = Nc

    def setup(self):
        
        domain_points = (Nr+2, Nz+2)

        self.r = np.linspace(0, self.Lr, self.Nr+2)                # array of mesh points in r direction
        self.z = np.linspace(0, self.Lz, self.Nz+2)                # array of mesh points in z direction
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

        self.T = 1/prob.f                                     # time of one period 
        self.t = np.arange(0,self.Nc*self.T, self.dt)                  # array of mesh points in time

        self.u_fr = np.zeros(domain_points)                # fluid velocity in r direction at next timestep
        self.u_fr_1 = np.zeros(domain_points)              # at previous timestep t-dt

        self.u_fz = np.zeros(domain_points)                # fluid velocity in z direction
        self.u_fz_1 = np.zeros(domain_points)              # at previous timestep t-dt
        
        self.psi_mr = np.zeros(domain_points)              # tissue matrix displacement in r direction at next timestep
        self.psi_mr_1 = np.zeros(domain_points)            # at previous timestep t-dt
        self.psi_mr_2 = np.zeros(domain_points)            # at previous timestep t-2*dt

        self.psi_mz = np.zeros(domain_points)              # tissue matrix displacement in z direction at next timestep
        self.psi_mz_1 = np.zeros(domain_points)            # at previous timestep t-dt
        self.psi_mz_2 = np.zeros(domain_points)            # at previous timestep t-2*dt

        self.Np = int(round(T/float(dt)))                  # number of timesteps in one period

        self.radiation_force_fr = np.zeros((self.Nr,self.Nz,self.Np))     # array that stores the values of u_fr from the last period
        self.radiation_force_fz = self.radiation_force_fr.copy()

        self.instant_rad_fr = np.zeros((self.Nr, self.Nz))
        self.instant_rad_fr = np.zeros((self.Nr, self.Nz))

    def iterate():
        
        omega = 2*np.pi*self.prob.f

        for ii in range(len(t)):
            # set boundary conditions for psi_mr and u_fr at r = 0
            self.psi_mr_1[0,:] = self.prob.vec_psi_c_amplitude(self.z)*np.sin(omega*self.t[ii])
            self.u_fr_1[0,:] = omega*self.prob.vec_psi_c_amplitude(self.z)*np.cos(omega*self.t[ii])

            # advance psi_m and u_f
            self.psi_mr = self.advance_psi_mr()
            self.psi_mz = self.advance_psi_mz()
            self.u_fr = self.advance_u_fr()
            self.u_fz = self.advance_u_fz()

            # set boundaries to 0 
            self.psi_mr[-1, :], self.psi_mr[:-1] = 0,0
            self.psi_mz[-1,:], self.psi_mz[:,-1], self.psi_mz[:,0] = 0,0,0
            
            self.u_fr[-1, :], self.u_fr[:,-1] = 0,0
            self.u_fz[-1, :], self.u_fz[:,-1], self.u_fz[:,0] = 0,0,0

            


            # update variables
            self.psi_mr_2, self.psi_mr_1 = self.psi_mr_1, self.psi_mr
            self.psi_mz_2, self.psi_mz_1 = self.psi_mz_1, self.psi_mz
            
            self.u_fr_1 = self.u_fr
            self.u_fz_1 = self.u_fz



           
             


    def advance_u_fr(self):
        psi_n_delta_mr = self.psi_mr-self.psi_mr_1
        
        A = self.prob.rho_f/self.dt

        B = -u_fr_1[1:-1, 1:-1]/(2*self.dr) + \
            self.prob.eta_f(1/(2*self.r[1:-1]*dr)+ 1/(self.dr**2))

        C = -self.prob.gamma - self.prob.eta_f/(self.r[1:-1]**2) - \
            2*self.prob.eta_f/(self.dr**2) - 2*self.prob.eta_f/(self.dz**2) + self.prob.rho_f/self.dt

        D = self.u_fr_1[1:-1,1:-1]/(2*self.dr) + self.prob.eta_f(1/(2*self.r[1:-1]*self.dr)+ 1/(self.dr**2))

        E = self.u_fr_1[1:-1,1:-1]/(2*self.dz) + self.prob.eta_f/(self.dz**2)

        F = self.u_fr_1[1:-1,1:-1]/(2*self.dz) + self.prob.eta_f/(self.dz**2)

        G = self.prob.gamma/self.dt 

        u = self.u_fr_1[2:, 1:-1]*B + self.u_fr_1[1:-1, 1:-1]*C + self.u_fr_1[:-2, 1:-1]*D + \
            self.u_fr_1[1:-1, 2:]*E + self.u_fr_1[1:-1, :-2]*F + delta_n_psi_mr[1:-1,1:-1]*G \
            + self.instant_rad_fr

        u = u / A

        #TODO: set boundary conditions

        return u


    def advance_u_fz(self):
        
        delta_n_psi_mz = self.psi_mz-self.psi_mz_1

        A = self.prob.rho_f/self.dt

        B = -self.u_fr_1[1:-1, 1:-1]/(2*self.dr) + \
            self.prob.eta_f(1/(2*self.r[1:-1]*self.dr)+ 1/(self.dr**2))

        C = -self.prob.gamma - 2*self.prob.eta_f/(self.dr**2) - \
            2*self.prob.eta_f/(self.dz**2) + self.prob.rho_f/self.dt

        D = self.u_fr_1[1:-1,1:-1]/(2*self.dr) + \
            self.prob.eta_f(1/(2*self.r[1:-1]*self.dr)+ 1/(self.dr**2))

        E = -self.u_fr_1[1:-1,1:-1]/(2*self.dz) + self.prob.eta_f/(self.dz**2)

        F = self.u_fr_1[1:-1,1:-1]/(2*self.dz) + self.prob.eta_f/(self.dz**2)

        G = self.prob.gamma/self.dt 


        u = self.u_fz_1[2:, 1:-1]*B + self.u_fz_1[1:-1, 1:-1]*C + self.u_fz_1[:-2, 1:-1]*D + \
            self.u_fz_1[1:-1, 2:]*E + self.u_fz_1[1:-1, :-2]*F + delta_n_psi_mz[1:-1,1:-1]*G \
            + self.instant_rad_fz

        u = u / A

        #TODO: set boundary conditions

        return u
            

    def advance_psi_mr(self):
        
        A = self.prob.rho_m/(self.dt**2) + self.prob.gamma/self.dt

        B = -self.prob.mu_m/(2*self.r[1:-1,1:-1]*self.dr)

        C = 2*self.prob.rho_m/(self.dt**2) - self.prob.mu_m/(self.r[1:-1]**2) -\
             2*self.prob.mu_m/(self.dz**2) + self.prob.gamma/self.dt 

        D = - B

        E = -self.prob.mu_m/(self.dz**2)

        F = -E 

        G = -self.prob.rho_m/(self.dt**2)

        psi = self.psi_mr_1[2:, 1:-1]*B + self.psi_mr_1[1:-1, 1:-1]*C + self.psi_mr_1[:-2, 1:-1]*D +\
            self.psi_mr_1[1:-1, 2:]*E + self.psi_mr_1[1:-1, :-2]*F + self.psi_mr_2[1:-1, 1:-1]*G \
            + self.prob.mu_m/(2*self.dz)*( -1/self.r[1:-1]*(self.psi_mz_1[1:-1, 2:] - self.psi_mz_1[1:-1, :-2]) - 1/(2*self.dr)*
                (self.psi_mz_1[2:, 2:] - self.psi_mz_1[2:, :-2] - self.psi_mz_1[:-2, 2:] + self.psi_mz_1[:-2, :-2])) \
            + self.u_fr_1[1:-1, 1:-1]*self.prob.gamma

        psi = psi / A

        #TODO: set boundary conditions

        return psi    


    def advance_psi_mz(self):
        A = self.prob.rho_m/(self.dt**2) + self.prob.gamma/self.dt

        B = self.prob.mu_m/(self.dr**2) + self.prob.mu_m/(2*self.r[1:-1]*self.dr)

        C = 2*self.prob.rho_m/(self.dt**2) - 2*self.prob.mu_m/(self.dr**2) + self.prob.gamma/self.dt 

        D = self.prob.mu_m/(self.dr**2) - self.prob.mu_m/(2*self.r[1:-1]*self.dr)

        E = -self.prob.rho_m/(self.dt**2)

        psi = self.psi_mz_1[2:, 1:-1]*B + self.psi_mz_1[1:-1,1:-1]*C + self.psi_mz_1[:-2, 1:-1]*D + \
            self.psi_mz_2[1:-1,1:-1]*E \
          + self.prob.mu_m/(2*self.dz)*( -1/self.r[1:-1]*(self.psi_mr_1[1:-1, 2:] - self.psi_mr_1[1:-1, :-2]) - 1/(2*self.dr)* \
            (self.psi_mr_1[2:, 2:] - self.psi_mr_1[2:, :-2] - self.psi_mr_1[:-2, 2:] + self.psi_mr_1[:-2, :-2])) \
          + self.prob.gamma*self.u_fz_1[1:-1]
          

        psi = psi / A

        #TODO: set boundary conditions

        return psi 
            

    def calcualate_radiation_force(self, i):
        self.radiation_force_fr[:,:,i % self.Np] = self.u_fr_1[1:-1, 1:-1]*(self.u_fr_1[2:, 1:-1] - \
            self.u_fr_1[:-2, 1:-1])/(2*self.dr) + self.u_fz_1[1:-1, 1:-1]*(self.u_fr_1[1:-1, 2:] - \
            self.u_fr_1[1:-1,:-2])/(2*self.dz)
        
        self.radiation_force_fz[:,:,i % self.Np] = self.u_fr_1[1:-1, 1:-1]*(self.u_fz_1[2:, 1:-1] - \
            self.u_fz_1[:-2, 1:-1])/(2*self.dr) + self.u_fz_1[1:-1, 1:-1]*(self.u_fz_1[1:-1, 2:] - \
            self.u_fz_1[1:-1,:-2])/(2*self.dz)

        if i < Np:          # first period
            rad_force_fr = np.mean(np.dot(self.radiation_force_fr[:,:,i+1], self.dt), axis=2)
            rad_force_fz = np.mean(np.dot(self.radiation_force_fz[:,:,i+1], self.dt), axis=2)
                
        else:               # second period onwards
            rad_force_fr = np.mean(np.dot(self.radiation_force_fr[:,:,:], self.dt), axis=2)
            rad_force_fz = np.mean(np.dot(self.radiation_force_fz[:,:,:], self.dt), axis=2)

        self.instant_rad_fr = -self.prob.rho_t*rad_force_fr 
        self.instant_rad_fz = -self.prob.rho_t*rad_force_fz



