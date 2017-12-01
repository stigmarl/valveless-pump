import numpy as np
import numpy.matlib
import problem

import sys
sys.path.append('../')

from misc.derivatives import *

import time

import matplotlib
import matplotlib.pyplot as plt




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

        self.r = np.linspace(0, self.Lr, self.Nr+2, dtype=np.float64)+self.prob.a_0               # array of mesh points in r direction
        self.z = np.linspace(0, self.Lz, self.Nz+2, dtype=np.float64)                 # array of mesh points in z direction
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

        self.T = 1/self.prob.f                                      # time of one period
        self.t = np.arange(-6*self.T,self.Nc*self.T, self.dt, dtype=np.float64 )               # array of mesh points in time

        self.u_fr = np.zeros(domain_points, dtype=np.float64)                         # fluid velocity in r direction at next timestep
        self.u_fr_1 = np.zeros(domain_points, dtype=np.float64)                       # at previous timestep t-dt

        self.u_fz = np.zeros(domain_points, dtype=np.float64)                         # fluid velocity in z direction
        self.u_fz_1 = np.zeros(domain_points, dtype=np.float64)                       # at previous timestep t-dt
        
        self.psi_mr = np.zeros(domain_points, dtype=np.float64)                       # tissue matrix displacement in r direction at next timestep
        self.psi_mr_1 = np.zeros(domain_points, dtype=np.float64)                     # at previous timestep t-dt
        self.psi_mr_2 = np.zeros(domain_points, dtype=np.float64)                     # at previous timestep t-2*dt

        self.psi_mz = np.zeros(domain_points, dtype=np.float64)                       # tissue matrix displacement in z direction at next timestep
        self.psi_mz_1 = np.zeros(domain_points, dtype=np.float64)                     # at previous timestep t-dt
        self.psi_mz_2 = np.zeros(domain_points, dtype=np.float64)                     # at previous timestep t-2*dt

        self.Np = int(round(self.T/float(self.dt)))                 # number of timesteps in one period

        self.radiation_force_fr = np.zeros((self.Nr,self.Nz,self.Np), dtype=np.float64) # array that stores the values of u_fr from the last period
        self.radiation_force_fz = self.radiation_force_fr.copy()

        self.instant_rad_fr = np.zeros(domain_points, dtype=np.float64)          # array that holds radiation force in r at current timestep
        self.instant_rad_fz = np.zeros(domain_points, dtype=np.float64)          # array that holds radiation force in z at current timestep

        #self.r_array = np.matlib.repmat(self.r, self.Nz+2,1).transpose() # array of r values in whole domain
        #self.z_array = np.repeat(self.z, self.Nr+2).reshape(self.Nr+2, self.Nz+2) # array of z values in whole domain
        #self.z_array = np.matlib.repmat(self.z, self.Nr+2, 1)
        self.z_array, self.r_array = np.meshgrid(self.z, self.r)


        # matplotlib stuff
        pgf_with_latex = {  # setup matplotlib to use latex for output# {{{
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "text.latex.unicode": True,
            "font.family": "serif",
            "font.serif": [],  # blank entries should cause plots
            "font.sans-serif": [],  # to inherit fonts from the document
            "font.monospace": [],
            "axes.labelsize": 10,  # LaTeX default is 10pt font.
            "font.size": 10,
            "legend.fontsize": 8,  # Make the legend/label fonts
            "xtick.labelsize": 8,  # a little smaller
            "ytick.labelsize": 8,
            #"figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts
                r"\usepackage[T1]{fontenc}",  # plots will be generated
                r"\usepackage[detect-all,locale=DE]{siunitx}",
            ]  # using this preamble
        }
        matplotlib.rcParams.update(pgf_with_latex)


    def iterate(self):
        
        omega = 2*np.pi*self.prob.f

        iters = 120

        numplots = 5

        u_time = np.zeros((numplots,iters))

        for ii in range(iters):
            
            print("Iterating at loop ", ii)

            # calculate radiation force
            self.calculate_radiation_force(ii)

            # set boundary conditions for psi_mr and u_fr at r = 0
            self.psi_mr_1[0,:] = self.prob.vec_psi_c_amplitude(self.z)*np.sin(omega*self.t[ii])*(1+np.tanh(self.t[ii]/self.T))*0.5

            self.u_fr_1[0,:] = omega*self.prob.vec_psi_c_amplitude(self.z)* \
                0.5*( (1+np.tanh(self.t[ii]/self.T))*np.cos(omega*self.t[ii]) + \
                      1/self.T*(1-np.tanh(self.t[ii]/self.T)**2)*np.sin(omega*self.t[ii]))

            #self.u_fr_1[:,0] = 4/3*self.u_fr_1[:,1] - 1/3*self.u_fr_1[:,2]
            #self.psi_mr_1[:, 0] = 4/3 * self.psi_mr_1[:, 1] - 1 / 3 * self.psi_mr_1[:, 2]
            self.u_fr_1[:, 0] = 4 / 3 * self.u_fr_1[:, 1] - 1 / 3 * self.u_fr_1[:, 2]
            self.psi_mr_1[:, 0] = 4 / 3 * self.psi_mr_1[:, 1] - 1 / 3 * self.psi_mr_1[:, 2]

            # advance psi_m and u_f
            self.psi_mr[1:-1, 1:-1] = self.advance_psi_mr()
            self.psi_mz[1:-1, 1:-1] = self.advance_psi_mz()
            self.u_fr[1:-1, 1:-1] = self.advance_u_fr()
            self.u_fz[1:-1, 1:-1] = self.advance_u_fz()

            #self.psi_mr[:, 0] = 4 / 3 * self.psi_mr_1[:, 1] - 1 / 3 * self.psi_mr_1[:, 2]

    

            # set boundaries
            self.psi_mr[-1, :], self.psi_mr[:,-1] = 0, 0
            self.psi_mz[-1,:], self.psi_mz[:,-1], self.psi_mz[:,0] = 0, 0, 0


            self.u_fr[-1, :], self.u_fr[:,-1] = 0, 0

            self.u_fz[-1, :], self.u_fz[:,-1] = 0, 0


            # update variables
            self.psi_mr_2, self.psi_mr_1 = self.psi_mr_1.copy(), self.psi_mr.copy()
            self.psi_mz_2, self.psi_mz_1 = self.psi_mz_1.copy(), self.psi_mz.copy()
            
            self.u_fr_1 = self.u_fr.copy()
            self.u_fz_1 = self.u_fz.copy()



            # update variable used for plotting
            u_time[:, ii] = self.u_fr_1[1:numplots+1,1]


            # to easier see values when iterating

            print("max psi_mr_1: ", np.amax(self.psi_mr_1))
            print("max psi_mz_1: ", np.amax(self.psi_mz_1))

            print("max u_fr_1: ", np.amax(self.u_fr_1))
            print("min u_fz_1: ", np.amin   (self.u_fz_1))
            print()

        self.plot_interstitium(u_time, numplots, ii)

    def plot_interstitium(self, u_time, numplots, ntimesteps):

        fig = plt.figure(0)

        sub1 = fig.add_subplot(211)
        for i in range(numplots):
            sub1.plot((self.t[:ntimesteps + 1] - self.t[0]) * 1e6, u_time[i, :], label='row %d' % i)
            sub1.set_xlabel(r't [$\mu s$]', fontsize=16)
            sub1.set_ylabel(r'u [$m/s$]', fontsize=16)
            sub1.set_title('fluid velocity', fontsize=18)

        sub1.legend(fontsize=14)
        sub1.grid(True, which='both')

        sub2 = fig.add_subplot(212)
        for j in range(numplots):
            sub2.plot((self.t[:ntimesteps + 1] - self.t[0]) * 1e6, np.cumsum(u_time[j, :]) * self.dt * 1e6, label='row %d' % j)
            sub2.set_xlabel(r't [$\mu s$]', fontsize=16)
            sub2.set_ylabel(r'$\Psi$ [$\mu m$]', fontsize=16)
            sub2.set_title('displacement from fluid velocity', fontsize=18)

        sub2.legend(fontsize=14)
        sub2.grid(True, which='both')

        plt.show()


    def advance_u_fr(self):

        u_fr_dr = dr_central(self.u_fr_1, self.dr)/self.r_array[1:-1,1:-1]
        u_fr_drr = drr(self.u_fr_1, self.dr)
        u_fr_dzz = dzz(self.u_fr_1, self.dz)
        u_fr_r2 = self.u_fr_1[1:-1,1:-1]/(self.r_array[1:-1,1:-1]**2)

        RHS = -self.prob.gamma*(self.u_fr_1[1:-1,1:-1] - dt(self.psi_mr, self.psi_mr_1, self.dt)) + \
                self.prob.eta_f*(-u_fr_r2 + u_fr_dr + u_fr_drr  + u_fr_dzz)

        LHS = self.u_fr_1[1:-1,1:-1]*dr_backward(self.u_fr_1, self.dr) + \
              self.u_fz_1[1:-1,1:-1]*dz_backward(self.u_fr_1, self.dz)

        u = self.u_fr_1[1:-1,1:-1] - self.dt*LHS + self.dt/self.prob.rho_f*(RHS )

        return u



    def advance_u_fz(self):

        RHS = -self.prob.gamma*(self.u_fz_1[1:-1,1:-1] - dt(self.psi_mz, self.psi_mz_1, self.dt)) + \
            self.prob.eta_f * (dr_central(self.u_fz_1, self.dr)/self.r_array[1:-1,1:-1] + \
            drr(self.u_fz_1, self.dz) + dzz(self.u_fz_1, self.dz))

        LHS = self.u_fr_1[1:-1,1:-1]*dr_backward(self.u_fz_1, self.dr) + \
              self.u_fz_1[1:-1,1:-1]*dz_backward(self.u_fz_1, self.dz)

        u = self.u_fz_1[1:-1,1:-1] - self.dt*LHS + self.dt/self.prob.rho_f*(RHS )

        return u
            

    def advance_psi_mr(self): 


        RHS = self.prob.mu_m * (-1 / self.r_array[1:-1,1:-1] * (dr_central(self.psi_mr_1, self.dr) + \
                dz_central(self.psi_mz_1, self.dz) + self.psi_mr_1[1:-1,1:-1]/self.r_array[1:-1,1:-1]) - \
                drz(self.psi_mz_1, self.dr, self.dz) + dzz(self.psi_mr_1, self.dz)) + \
                + self.prob.gamma*(self.u_fr_1[1:-1, 1:-1] - dt(self.psi_mr_1, self.psi_mr_2, self.dt))

        #psi = 2*self.psi_mr_1[1:-1, 1:-1] - self.psi_mr_2[1:-1,1:-1] + np.dot(RHS, (dt*dt)/self.prob.rho_m)

        psi = 2 * self.psi_mr_1[1:-1, 1:-1] - self.psi_mr_2[1:-1, 1:-1] + RHS*self.dt**2 / self.prob.rho_m

        return psi


    def advance_psi_mz(self):

        RHS = self.prob.mu_m * ( drr(self.psi_mz_1, self.dr) + 1/self.r_array[1:-1, 1:-1]*(dr_central(self.psi_mz_1, self.dr) \
                - dz_central(self.psi_mz_1, self.dz)) - drz(self.psi_mr_1, self.dr, self.dz)) + \
                self.prob.gamma*(self.u_fz_1[1:-1,1:-1] - dt(self.psi_mz_1, self.psi_mz_2, self.dt))

        psi = 2*self.psi_mz_1[1:-1,1:-1] - self.psi_mz_2[1:-1,1:-1] + RHS*(self.dt**2)/self.prob.rho_m

        return psi


    def calculate_radiation_force(self, i):
        self.radiation_force_fr[:,:,i % self.Np] = self.u_fr_1[1:-1,1:-1]*dr_central(self.u_fr_1, self.dr) - \
            self.u_fz_1[1:-1,1:-1]*dz_central(self.u_fr_1, self.dz)
        
        self.radiation_force_fz[:,:,i % self.Np] = self.u_fr_1[1:-1,1:-1]*dr_central(self.u_fz_1, self.dr) - \
            self.u_fz_1[1:-1,1:-1]*dz_central(self.u_fz_1, self.dz)

        if i < self.Np:          # first period
            rad_force_fr = np.ndarray.mean(self.radiation_force_fr[:,:,:i+1]*self.dt, axis=2)
            rad_force_fz = np.ndarray.mean(self.radiation_force_fz[:,:,:i+1]*self.dt, axis=2)
                
        else:               # second period onwards
            rad_force_fr = np.mean(self.radiation_force_fr[:,:,:]*self.dt, axis=2)
            rad_force_fz = np.mean(self.radiation_force_fz[:,:,:]* self.dt, axis=2)

        self.instant_rad_fr = -self.prob.rho_t*rad_force_fr
        self.instant_rad_fz = -self.prob.rho_t*rad_force_fz



