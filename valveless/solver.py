import numpy as np
import numpy.matlib
import problem

import sys
sys.path.append('../')

from misc.derivatives import *

import time

import matplotlib
import matplotlib.pyplot as plt

from decimal import Decimal
import numba





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
        #self.t = np.arange(0,self.Nc*self.T, self.dt, dtype=np.float64 )               # array of mesh points in time

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

        #self.radiation_force_fr = np.zeros((self.Nr,self.Nz,self.Np), dtype=np.float64) # array that stores the values of u_fr from the last period
        #self.radiation_force_fz = self.radiation_force_fr.copy()

        #self.instant_rad_fr = np.zeros(domain_points, dtype=np.float64)          # array that holds radiation force in r at current timestep
        #self.instant_rad_fz = np.zeros(domain_points, dtype=np.float64)          # array that holds radiation force in z at current timestep

        #self.r_array = np.matlib.repmat(self.r, self.Nz+2,1).transpose() # array of r values in whole domain
        #self.z_array = np.repeat(self.z, self.Nr+2).reshape(self.Nr+2, self.Nz+2) # array of z values in whole domain
        #self.z_array = np.matlib.repmat(self.z, self.Nr+2, 1)
        self.z_array, self.r_array = np.meshgrid(self.z, self.r)

        self.vec_psi_c_amplitude = self.prob.vec_psi_c_amplitude(self.z)


        # matplotlib stuff
        pgf_with_latex = {  # setup matplotlib to use latex for output# {{{
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "text.latex.unicode": True,
            "font.family": "serif",
            "font.serif": [],  # blank entries should cause plots
            "savefig.dpi": 200,
            "font.sans-serif": [],  # to inherit fonts from the document
            "font.monospace": [],
            "axes.labelsize": 10,  # LaTeX default is 10pt font.
            "font.size": 10,
            "legend.fontsize": 8,  # Make the legend/label fonts
            "xtick.labelsize": 12,  # a little smaller
            "ytick.labelsize": 12,
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

        iters = self.Np*self.Nc                  # total number of iterations

        print("Total number of iterations: %.2e" % iters)

        numplots = 4                                          # number of plots for fluid velocity

        plt_idx = 0                                           # index used for plot variables

        n_period_shift = 3

        points_per_period = 10                               # number of points to plot per period

        total_points = points_per_period*self.Nc              # total number of points to plot

        # 1D case
        u_time = np.zeros((total_points,numplots))
        u_displacement = np.zeros((total_points,))
        u_boundary = np.zeros((total_points,numplots))

        psi_time = np.zeros((total_points,numplots))
        psi_boundary = np.zeros((total_points,numplots))

        idx = 0

        for ii in range(iters):

            if (ii+1)%(iters//100) == 0:
                print("Done with ", (ii+1)*100/iters, 'percent')


            # calculate radiation force
            #self.calculate_radiation_force(ii)

            # set boundary conditions for psi_mr and u_fr at r = 0
            self.psi_mr_1[0,:] = self.vec_psi_c_amplitude*np.sin(omega*ii*self.dt)*(1+np.tanh((ii*self.dt-n_period_shift*self.T)/self.T))*0.5

            #self.u_fr_1[0, :] = omega * self.prob.vec_psi_c_amplitude(self.z)*np.cos(omega*(self.t[ii]-self.t[0]))*(1+np.tanh(self.t[ii]/self.T))*0.5

            self.u_fr_1[0,:] = self.vec_psi_c_amplitude*0.5*(\
                 (1+np.tanh((ii*self.dt-n_period_shift*self.T)/self.T))*omega*np.cos(omega*self.dt*ii) + \
                   1/self.T*(1-np.tanh((self.dt*ii-n_period_shift*self.T)/self.T)**2)*np.sin(omega*self.dt*ii))



                #u_boundary[ii] = self.u_fr_1[0,1]
                #psi_boundary[ii] = self.psi_mr_1[0,1]

                
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

            if ii % (iters // total_points) == 0:
                # update variable used for plotting
                u_time[plt_idx] = self.u_fr_1[1:numplots+1,1]
                psi_time[plt_idx] = self.psi_mr_1[1:numplots+1,1]

                disp = self.u_fr[idx+1,1]*self.dt+self.prob.a_0
                u_displacement[plt_idx] = disp
                idx = (np.abs(disp - self.r[1:])).argmin()
                plt_idx+=1

            # Update u_displacement. Look at one point at first



            # to easier see values when iterating
            """
            print("max psi_mr_1: ", np.amax(self.psi_mr_1))
            print("max psi_mz_1: ", np.amax(self.psi_mz_1))

            print("max u_fr_1: ", np.amax(np.abs(self.u_fr_1)))
            print("min u_fz_1: ", np.amin   (self.u_fz_1))
            print()
            """

        self.plot_interstitium(u_time, u_displacement, psi_time, psi_boundary, numplots, ii)

    def plot_interstitium(self, u_time, u_displacement, psi_time, psi_boundary, numplots, ntimesteps):

        CFL = self.dt / (self.dr ** 2)

        save_fig_string = "_gamma%.3e_cfl%.3e_L%d_N%d.png" % (self.prob.gamma, CFL, self.Lr*1e6, self.Nr)

        fig = plt.figure(0, figsize=(7,8))

        sub1 = fig.add_subplot(211)
        for i in range(numplots):
            sub1.plot( u_time[:,i],\
                      label=r'$r_{%d} = %.2f\mathrm{\mu m}$' % (i+1, self.r[i+1]*1e6))
            sub1.set_xlabel(r't [$\mathrm{\mu s}$]', fontsize=16)
            sub1.set_ylabel(r'u [$\mathrm{m/s}$]', fontsize=16)
            sub1.set_title(r'fluid velocity $\mathrm{u}_{i,1,n}^{\mathrm{f}r}$, $\gamma$ = %.3e ' % Decimal(self.prob.gamma), fontsize=18)

        sub1.legend(fontsize=14)
        sub1.grid(True, which='both')

        sub2 = fig.add_subplot(212)
        for j in range(1):
            sub2.plot( (u_displacement-self.prob.a_0) * 1e6)
            sub2.set_xlabel(r't [$\mathrm{\mu s}$]', fontsize=16)
            sub2.set_ylabel(r'$\Psi$ [$\mathrm{\mu m}$]', fontsize=16)
            sub2.set_title('Position from fluid velocity, CFL = %.3e' % CFL, fontsize=18)

        sub2.legend(fontsize=14)
        sub2.grid(True, which='both')

        plt.tight_layout(h_pad=0.5)



        fig2 = plt.figure(1,figsize=(7,8))
        sub3 = fig2.add_subplot(211)
        for i in range(numplots):
            sub3.plot( psi_time[:,i],
                      label=r'$r_{%d} = %.2f\mathrm{\mu m}$' % (i + 1, self.r[i + 1] * 1e6))
            #sub1.set_ylim((-1e-4, 1e-4))
            sub3.set_xlabel(r't [$\mathrm{\mu s}$]', fontsize=16)
            sub3.set_ylabel(r'$\Psi$ [$\mathrm{m}$]', fontsize=16)
            sub3.set_title(
                r'Tissue displacement $\Psi_{i,1,n}^{\mathrm{m}r}$, $\gamma$ = %.3e' % Decimal(self.prob.gamma),
                fontsize=18)

        sub3.legend(fontsize=14)
        sub3.grid(True, which='both')


        sub4 = fig2.add_subplot(212)
        for i in range(1):
            sub4.plot(psi_boundary*1e6)
            sub4.set_xlabel(r't [$\mathrm{\mu s}$]', fontsize=16)
            sub4.set_ylabel(r'Amplitude [$\mathrm{\mu m}$]', fontsize=16)
            sub4.set_title(
                r'Tissue boundary condition$\Psi_{\mathrm{m}r}(a_{0},z_{1},t)$',
                fontsize=18)

        sub4.legend(fontsize=14)
        sub4.grid(True, which='both')

        plt.tight_layout(h_pad=0.5)


        plt.show()

        fig.savefig('u_fr' + save_fig_string, format='png', dpi=200)
        fig2.savefig('psi_mr' + save_fig_string, format='png', dpi=200)


    def calculate_net_displacement_from_fluid(self):
        """
        1. Calculate displacement from fluid velocity and timestep
        2. Find closest point in grid and update position
        3. Use position is new point for calculations in next iteration

        Returns
        -------

        """


    def advance_u_fr(self):

        u_fr_dr = dr_central(self.u_fr_1, self.dr)/self.r_array[1:-1,1:-1]
        u_fr_drr = drr(self.u_fr_1, self.dr)
        u_fr_dzz = dzz(self.u_fr_1, self.dz)
        u_fr_r2 = self.u_fr_1[1:-1,1:-1]/(self.r_array[1:-1,1:-1]**2)

        RHS = -self.prob.gamma*(self.u_fr_1[1:-1,1:-1] - dt(self.psi_mr, self.psi_mr_1, self.dt)) + \
                self.prob.eta_f*(-u_fr_r2 + u_fr_dr + u_fr_drr  + u_fr_dzz)
        """

        RHS =  -self.prob.gamma*(self.u_fr_1[1:-1,1:-1] - dt(self.psi_mr, self.psi_mr_1, self.dt)) + \
                self.prob.eta_f*(-1/self.r_array[1:-1,1:-1]*( dr_central(self.u_fr_1, self.dr)  + \
                dz_central(self.u_fz_1, self.dz) + self.u_fr_1[1:-1,1:-1]/self.r_array[1:-1,1:-1]) - \
                drz(self.u_fz_1, self.dr, self.dz) + dzz(self.u_fr_1, self.dz))
        """

        LHS = self.u_fr_1[1:-1,1:-1]*dr_backward(self.u_fr_1, self.dr) + \
              self.u_fz_1[1:-1,1:-1]*dz_backward(self.u_fr_1, self.dz)

        u = self.u_fr_1[1:-1,1:-1] - self.dt*LHS + self.dt/self.prob.rho_f*(RHS)

        return u


    def advance_u_fz(self):


        RHS = -self.prob.gamma*(self.u_fz_1[1:-1,1:-1] - dt(self.psi_mz, self.psi_mz_1, self.dt)) + \
            self.prob.eta_f * (dr_central(self.u_fz_1, self.dr)/self.r_array[1:-1,1:-1] + \
            drr(self.u_fz_1, self.dz) + dzz(self.u_fz_1, self.dz))

        """
        RHS = self.prob.eta_f*( drr(self.u_fz_1, self.dr) + 1/self.r_array[1:-1,1:-1]*( dr_central(self.u_fz_1, self.dr) \
                - dz_central(self.u_fr_1, self.dz)) - drz(self.u_fr_1,self.dr,self.dz) ) \
              -self.prob.gamma * (self.u_fz_1[1:-1, 1:-1] - dt(self.psi_mz, self.psi_mz_1, self.dt))
        """
        LHS = self.u_fr_1[1:-1,1:-1]*dr_backward(self.u_fz_1, self.dr) + \
              self.u_fz_1[1:-1,1:-1]*dz_backward(self.u_fz_1, self.dz)

        u = self.u_fz_1[1:-1,1:-1] - self.dt*LHS + self.dt/self.prob.rho_f*(RHS)

        return u

    def advance_psi_mr(self):


        psi_mr_dr_r = dr_central(self.psi_mr_1, self.dr)/self.r_array[1:-1,1:-1]
        psi_mz_dz_r = dz_central(self.psi_mz_1, self.dz)/self.r_array[1:-1,1:-1]
        psi_mr_r2 = self.psi_mr_1[1:-1,1:-1]/(self.r_array[1:-1,1:-1]**2)
        psi_mz_drz = drz(self.psi_mz_1, self.dr,self.dr)
        psi_mr_dzz = dzz(self.psi_mr_1, self.dz)
        RHS = self.prob.mu_m*(-( psi_mr_dr_r + psi_mz_dz_r + psi_mr_r2) - psi_mz_drz + psi_mr_dzz) + \
              self.prob.gamma * (self.u_fr_1[1:-1, 1:-1] - dt(self.psi_mr, self.psi_mr_1, self.dt))

        """
        RHS = self.prob.mu_m * (-1 / self.r_array[1:-1,1:-1] * (dr_central(self.psi_mr_1, self.dr) + \
                dz_central(self.psi_mz_1, self.dz) + self.psi_mr_1[1:-1,1:-1]/self.r_array[1:-1,1:-1]) - \
                drz(self.psi_mz_1, self.dr, self.dz) + dzz(self.psi_mr_1, self.dz)) + \
                + self.prob.gamma*(self.u_fr_1[1:-1, 1:-1] - dt(self.psi_mr_1, self.psi_mr_2, self.dt))

        """

        RHS_product = RHS*self.dt**2 / self.prob.rho_m

        psi = 2 * self.psi_mr_1[1:-1, 1:-1] - self.psi_mr_2[1:-1, 1:-1] + RHS_product

        return psi

    def advance_psi_mz(self):
        """
        psi_mz_drr = drr(self.psi_mz_1, self.dr)
        psi_mz_dr_r = dr_central(self.psi_mz_1, self.dr)/self.r_array[1:-1,1:-1]
        psi_mr_dz_r = dz_central(self.psi_mr_1, self.dz)/self.r_array[1:-1,1:-1]
        psi_mr_drz = drz(self.psi_mr_1, self.dr, self.dz)
        RHS = self.prob.mu_m*(psi_mz_drr + psi_mz_dr_r - psi_mr_dz_r - psi_mr_drz) + \
                self.prob.gamma*(self.u_fz[1:-1,1:-1] - dt(self.psi_mz, self.psi_mz_1, self.dt))
        """
        RHS = self.prob.mu_m * ( drr(self.psi_mz_1, self.dr) + 1/self.r_array[1:-1, 1:-1]*(dr_central(self.psi_mz_1, self.dr) \
                - dz_central(self.psi_mz_1, self.dz)) - drz(self.psi_mr_1, self.dr, self.dz)) + \
                self.prob.gamma*(self.u_fz_1[1:-1,1:-1] - dt(self.psi_mz_1, self.psi_mz_2, self.dt))


        psi = 2 * self.psi_mz_1[1:-1, 1:-1] - self.psi_mz_2[1:-1, 1:-1] + RHS * (self.dt ** 2) / self.prob.rho_m

        return psi


    def calculate_radiation_force(self, i):
        self.radiation_force_fr[:,:,i % self.Np] = self.u_fr_1[1:-1,1:-1]*dr_backward(self.u_fr_1, self.dr) - \
            self.u_fz_1[1:-1,1:-1]*dz_backward(self.u_fr_1, self.dz)
        
        self.radiation_force_fz[:,:,i % self.Np] = self.u_fr_1[1:-1,1:-1]*dr_backward(self.u_fz_1, self.dr) - \
            self.u_fz_1[1:-1,1:-1]*dz_backward(self.u_fz_1, self.dz)

        if i < self.Np:          # first period
            rad_force_fr = np.ndarray.mean(self.radiation_force_fr[:,:,:i+1]*self.dt, axis=2)
            rad_force_fz = np.ndarray.mean(self.radiation_force_fz[:,:,:i+1]*self.dt, axis=2)
                
        else:               # second period onwards
            rad_force_fr = np.mean(self.radiation_force_fr[:,:,:]*self.dt, axis=2)
            rad_force_fz = np.mean(self.radiation_force_fz[:,:,:]* self.dt, axis=2)

        self.instant_rad_fr = -self.prob.rho_t*rad_force_fr
        self.instant_rad_fz = -self.prob.rho_t*rad_force_fz



