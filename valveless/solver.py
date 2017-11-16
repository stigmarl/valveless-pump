import numpy as np
import numpy.matlib
import problem

import sys
sys.path.append('../')

from misc.derivatives import *

import time



class Solver(object):
    """

    Class that incapsulates numerical parameters into a single object.

    """


    def __init__(self, prob, Lr, Lz, Nr, Nz, dt, Nc):
        """
        Initializes and sets up the numerical parameters of the Solver.

        Parameters
        ----------
        prob: Problem
            An instance of Problem.
        Lr: scalar
            Length of domain in r [cm].
        Lz: scalar
            Length of domain in z [cm].
        Nr: scalar
            Number of points in r.
        Nz: scalar
            Number of points in z.
        dt: scalar
            Timestep of each iteration [ms]
        Nc: scalar
            Number of cycles to iterate over.
        """
        self.prob = prob
        self.Lr = Lr
        self.Lz = Lz
        self.Nr = Nr
        self.Nz = Nz
        self.dt = dt 
        self.Nc = Nc

        self._setup()

    def _setup(self):
        
        domain_points = (self.Nr+2, self.Nz+2)

        self.r = np.linspace(0, self.Lr, self.Nr+2)+self.prob.a_0   # array of mesh points in r direction
        self.z = np.linspace(0, self.Lz, self.Nz+2)                 # array of mesh points in z direction
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

        self.T = 1/self.prob.f                                      # time of one period
        self.t = np.arange(0,self.Nc*self.T, self.dt)               # array of mesh points in time

        self.u_fr = np.zeros(domain_points)                         # fluid velocity in r direction at next timestep
        self.u_fr_1 = np.zeros(domain_points)                       # at previous timestep t-dt

        self.u_fz = np.zeros(domain_points)                         # fluid velocity in z direction
        self.u_fz_1 = np.zeros(domain_points)                       # at previous timestep t-dt
        
        self.psi_mr = np.zeros(domain_points)                       # tissue matrix displacement in r direction at next timestep
        self.psi_mr_1 = np.zeros(domain_points)                     # at previous timestep t-dt
        self.psi_mr_2 = np.zeros(domain_points)                     # at previous timestep t-2*dt

        self.psi_mz = np.zeros(domain_points)                       # tissue matrix displacement in z direction at next timestep
        self.psi_mz_1 = np.zeros(domain_points)                     # at previous timestep t-dt
        self.psi_mz_2 = np.zeros(domain_points)                     # at previous timestep t-2*dt

        self.Np = int(round(self.T/float(self.dt)))                 # number of timesteps in one period

        self.radiation_force_fr = np.zeros((self.Nr,self.Nz,self.Np)) # array that stores the values of u_fr from the last period
        self.radiation_force_fz = self.radiation_force_fr.copy()

        self.instant_rad_fr = np.zeros((self.Nr, self.Nz))          # array that holds radiation force in r at current timestep
        self.instant_rad_fz = np.zeros((self.Nr, self.Nz))          # array that holds radiation force in z at current timestep

        self.r_array = np.repeat(self.r, self.Nz+2).reshape(self.Nr+2, self.Nz+2) # array of r values in whole domain
        #self.z_array = np.repeat(self.z, self.Nr+2).reshape(self.Nr+2, self.Nz+2) # array of z values in whole domain
        self.z_array = np.matlib.repmat(self.z, self.Nr+2, 1)

    def iterate(self):
        
        omega = 2*np.pi*self.prob.f

        for ii in range(40):
            
            print("Iterating at loop ", ii)

            # calculate radiation force
            #self.calcualate_radiation_force(ii)

            # set boundary conditions for psi_mr and u_fr at r = 0
            self.psi_mr_1[0,:] = self.prob.vec_psi_c_amplitude(self.z)*np.sin(omega*self.t[ii])
            self.u_fr_1[0,:] = omega*self.prob.vec_psi_c_amplitude(self.z)*np.cos(omega*self.t[ii])



            # advance psi_m and u_f
            self.psi_mr[1:-1, 1:-1] = self.advance_psi_mr()
            self.psi_mz[1:-1, 1:-1] = self.advance_psi_mz()
            self.u_fr[1:-1, 1:-1] = self.advance_u_fr()
            self.u_fz[1:-1, 1:-1] = self.advance_u_fz()

    

            # set boundaries to 0 
            self.psi_mr[-1, :], self.psi_mr[:,-1] = 0, 0
            self.psi_mz[-1,:], self.psi_mz[:,-1], self.psi_mz[:,0] = 0, 0, 0
            
            self.u_fr[-1, :], self.u_fr[:,-1] = 0, 0
            self.u_fz[-1, :], self.u_fz[:,-1] = 0, 0

            

            # update variables
            self.psi_mr_2, self.psi_mr_1 = self.psi_mr_1, self.psi_mr
            self.psi_mz_2, self.psi_mz_1 = self.psi_mz_1, self.psi_mz
            
            self.u_fr_1 = self.u_fr
            self.u_fz_1 = self.u_fz

            #print("sum delta_n_psi_mr: ", np.sum(self.psi_mr_1-self.psi_mr_2))

            # to easier see values when iterating

            print("max psi_mr_1: ", np.amax(self.psi_mr_1))
            print("max psi_mz_1: ", np.amax(self.psi_mz_1))

            print("max u_fr_1: ", np.amax(self.u_fr_1))
            print("max u_fz_1: ", np.amax(self.u_fz_1))
            print()

            time.sleep(1)
           
             


    def advance_u_fr(self):


        delta_n_psi_mr = self.psi_mr_1-self.psi_mr_2

        A = self.prob.rho_f/self.dt

        B = -self.u_fr_1[1:-1, 1:-1]/self.dr + \
            self.prob.eta_f*(1/(2*self.r_array[1:-1,1:-1]*self.dr)+ 1/(self.dr**2))

        C = -self.prob.gamma - self.prob.eta_f/(self.r_array[1:-1,1:-1]**2) - \
            2*self.prob.eta_f/(self.dr**2) - 2*self.prob.eta_f/(self.dz**2) + self.prob.rho_f/self.dt + \
            self.u_fr_1[1:-1,1:-1]/(self.dr) + self.u_fz_1[1:-1,1:-1]/self.dz

        # this term makes it explode
        D = self.prob.eta_f*(-1/(2*self.r_array[1:-1,1:-1]*self.dr)+ 1/(self.dr**2))

        E = -self.u_fz_1[1:-1,1:-1]/self.dz + self.prob.eta_f/(self.dz**2)

        F = self.prob.eta_f/(self.dz**2)

        # this term also makes it explode
        G = self.prob.gamma/self.dt
        
        #print("u_fr A: ", A)
        #print("u_fr B: ", B)
        #print("u_fr C: ", C)
        #print("u_fr D: ", D)
        #print("u_fr E: ", E)
        #print("u_fr F: ", F)
        #print("u_fr G: ", G)
        
        #print("max u_fr: ", np.amax(self.u_fr_1))
        #print("sum delta_n_psi_mr: ", np.sum(delta_n_psi_mr))

        u = self.u_fr_1[2:, 1:-1]*B + self.u_fr_1[1:-1, 1:-1]*C + self.u_fr_1[:-2, 1:-1]*D + \
            self.u_fr_1[1:-1, 2:]*E + self.u_fr_1[1:-1, :-2]*F + delta_n_psi_mr[1:-1,1:-1]*G \
            #+ self.instant_rad_fr

        u = u / A

        #TODO: set boundary conditions

        return u






    def advance_u_fz(self):

        delta_n_psi_mz = self.psi_mz_1-self.psi_mz_2


        A = self.prob.rho_f/self.dt

        B = -self.u_fr_1[1:-1, 1:-1]/self.dr + \
            self.prob.eta_f*(1/(2*self.r_array[1:-1,1:-1]*self.dr)+ 1/(self.dr**2))

        C = -self.prob.gamma -self.prob.eta_f/(self.r_array[1:-1,1:-1]**2)- 2*self.prob.eta_f/(self.dr**2) - \
            2*self.prob.eta_f/(self.dz**2) + self.prob.rho_f/self.dt + \
            self.u_fr_1[1:-1,1:-1]/self.dr + self.u_fz_1[1:-1,1:-1]/self.dz

        D =  self.prob.eta_f*(-1/(2*self.r_array[1:-1,1:-1]*self.dr)+ 1/(self.dr**2))

        E = -self.u_fz_1[1:-1,1:-1]/self.dz + self.prob.eta_f/(self.dz**2)

        F =  self.prob.eta_f/(self.dz**2)

        G = self.prob.gamma/self.dt

        # delta_n_psi_mz
        u = self.u_fz_1[2:, 1:-1]*B + self.u_fz_1[1:-1, 1:-1]*C + self.u_fz_1[:-2, 1:-1]*D + \
            self.u_fz_1[1:-1, 2:]*E + self.u_fz_1[1:-1, :-2]*F + delta_n_psi_mz[1:-1,1:-1]*G \
            + self.instant_rad_fz

        u = u / A

        #TODO: set boundary conditions

        return u
            

    def advance_psi_mr(self): 

        
        """
        A = self.prob.rho_m/(self.dt**2) + self.prob.gamma/self.dt

        B = -self.prob.mu_m/(2*self.r_array[1:-1,1:-1]*self.dr)

        C = 2*self.prob.rho_m/(self.dt**2) - self.prob.mu_m/(np.power(self.r_array[1:-1,1:-1],2)) -\
             2*self.prob.mu_m/(self.dz**2) + self.prob.gamma/self.dt 

        D = - B

        E = -self.prob.mu_m/(self.dz**2)

        F = -E 

        G = -self.prob.rho_m/(self.dt**2)

        
        #print("psi_mr A: ", A)
        #print("psi_mr B: ", B.shape)
        #print("psi_mr C: ", C)
        #print("psi_mr D: ", D)
        #print("psi_mr E: ", E)
        #print("psi_mr F: ", F)
        #print("psi_mr G: ", G)
        #print("max psi_mr: ", np.amax(self.psi_mr_1))
        

        psi = self.psi_mr_1[2:, 1:-1]*B + self.psi_mr_1[1:-1, 1:-1]*C + self.psi_mr_1[:-2, 1:-1]*D +\
            self.psi_mr_1[1:-1, 2:]*E + self.psi_mr_1[1:-1, :-2]*F + self.psi_mr_2[1:-1, 1:-1]*G \
            + self.prob.mu_m/(2*self.dz)*( -1/self.r_array[1:-1,1:-1]*0*(self.psi_mz_1[1:-1, 2:] - self.psi_mz_1[1:-1, :-2]) \
            - 1/(2*self.dr)*(self.psi_mz_1[2:, 2:] - self.psi_mz_1[2:, :-2] - self.psi_mz_1[:-2, 2:] + self.psi_mz_1[:-2, :-2])) \
            + self.u_fr_1[1:-1, 1:-1]*self.prob.gamma

        psi = psi / A

        #TODO: set boundary conditions

        return psi

        """

        RHS = self.prob.mu_m * (-1 / self.r_array[1:-1,1:-1] * (dr_central(self.psi_mr_1, self.dr) + \
                dz_central(self.psi_mz_1, self.dz) + self.psi_mr_1[1:-1,1:-1]/self.r_array[1:-1,1:-1]) - \
                drz(self.psi_mz_1, self.dr, self.dz) + dzz(self.psi_mr_1, self.dz)) + \
                + self.prob.gamma*(self.u_fr_1[1:-1, 1:-1] - dt(self.psi_mr_1, self.psi_mr_2, self.dt))

        #psi = 2*self.psi_mr_1[1:-1, 1:-1] - self.psi_mr_2[1:-1,1:-1] + np.dot(RHS, (dt*dt)/self.prob.rho_m)

        psi = 2 * self.psi_mr_1[1:-1, 1:-1] - self.psi_mr_2[1:-1, 1:-1] + RHS*self.dt**2 / self.prob.rho_m

        return psi


    def advance_psi_mz(self):
        """
        A = self.prob.rho_m/(self.dt**2) + self.prob.gamma/self.dt

        B = self.prob.mu_m/(self.dr**2) + self.prob.mu_m/(2*self.r_array[1:-1,1:-1]*self.dr)

        C = 2*self.prob.rho_m/(self.dt**2) - 2*self.prob.mu_m/(self.dr**2) + self.prob.gamma/self.dt 

        D = self.prob.mu_m/(self.dr**2) - self.prob.mu_m/(2*self.r_array[1:-1,1:-1]*self.dr)

        E = -self.prob.rho_m/(self.dt**2)

        psi = self.psi_mz_1[2:, 1:-1]*B + self.psi_mz_1[1:-1,1:-1]*C + self.psi_mz_1[:-2, 1:-1]*D + \
            self.psi_mz_2[1:-1,1:-1]*E \
            + self.prob.mu_m/(2*self.dz)*( -1/self.r_array[1:-1,1:-1]*(self.psi_mr_1[1:-1, 2:] - self.psi_mr_1[1:-1, :-2]) - 1/(2*self.dr)* \
            (self.psi_mr_1[2:, 2:] - self.psi_mr_1[2:, :-2] - self.psi_mr_1[:-2, 2:] + self.psi_mr_1[:-2, :-2]))\
            + self.prob.gamma*self.u_fz_1[1:-1,1:-1]
          

        psi = psi / A

        #TODO: set boundary conditions

        return psi

        """
        RHS = self.prob.mu_m * ( drr(self.psi_mz_1, self.dr) + 1/self.r_array[1:-1, 1:-1]*(dr_central(self.psi_mz_1, self.dr) \
                - dz_central(self.psi_mz_1, self.dz)) - drz(self.psi_mr_1, self.dr, self.dz)) + \
                self.prob.gamma*(self.u_fz_1[1:-1,1:-1] - dt(self.psi_mz_1, self.psi_mz_2, self.dt))

        psi = 2*self.psi_mz_1[1:-1,1:-1] - self.psi_mz_2[1:-1,1:-1] + RHS*(self.dt**2)/self.prob.rho_m

        return psi


    def calculate_radiation_force(self, i):
        self.radiation_force_fr[:,:,i % self.Np] = self.u_fr_1[1:-1, 1:-1]*(self.u_fr_1[2:, 1:-1] - \
            self.u_fr_1[:-2, 1:-1])/(2*self.dr) + self.u_fz_1[1:-1, 1:-1]*(self.u_fr_1[1:-1, 2:] - \
            self.u_fr_1[1:-1,:-2])/(2*self.dz)
        
        self.radiation_force_fz[:,:,i % self.Np] = self.u_fr_1[1:-1, 1:-1]*(self.u_fz_1[2:, 1:-1] - \
            self.u_fz_1[:-2, 1:-1])/(2*self.dr) + self.u_fz_1[1:-1, 1:-1]*(self.u_fz_1[1:-1, 2:] - \
            self.u_fz_1[1:-1,:-2])/(2*self.dz)

        if i < self.Np:          # first period
            rad_force_fr = np.mean(np.dot(self.radiation_force_fr[:,:,i], self.dt), axis=2)
            rad_force_fz = np.mean(np.dot(self.radiation_force_fz[:,:,i], self.dt), axis=2)
                
        else:               # second period onwards
            rad_force_fr = np.mean(np.dot(self.radiation_force_fr[:,:,:], self.dt), axis=2)
            rad_force_fz = np.mean(np.dot(self.radiation_force_fz[:,:,:], self.dt), axis=2)

        self.instant_rad_fr = -self.prob.rho_t*rad_force_fr 
        self.instant_rad_fz = -self.prob.rho_t*rad_force_fz



