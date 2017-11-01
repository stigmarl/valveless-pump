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
        #u_fr_period = np.zeros((Nr+2,Nz+2, T/dt))

        self.u_fz = np.zeros(domain_points)                # fluid velocity in z direction
        self.u_fz_1 = np.zeros(domain_points)              # at previous timestep t-dt
        
        self.psi_mr = np.zeros(domain_points)              # tissue matrix displacement in r direction at next timestep
        self.psi_mr_1 = np.zeros(domain_points)            # at previous timestep t-dt
        self.psi_mr_2 = np.zeros(domain_points)            # at previous timestep t-2*dt

        self.psi_mz = np.zeros(domain_points)              # tissue matrix displacement in z direction at next timestep
        self.psi_mz_1 = np.zeros(domain_points)            # at previous timestep t-dt
        self.psi_mz_2 = np.zeros(domain_points)            # at previous timestep t-2*dt

        self.Np = int(round(T/float(dt)))                # number of timesteps in one period


    def advance_u_fr(self):
        pass 

    def advance_u_fz(self):
        pass

    def advance_psi_mr(self):
        pass

    def advance_psi_mz(self):
        pass
            


def solver(L, f, c_m, psi_ca, a_0, Lr, Lz, Nr, Nz, rho_m, rho_f, mu_m, eta_f, dt, Nc):
    """
    Solve 2D valveless pump by finite difference 
   
    Parameters
    ----------
    L: scalar
        Length of the bubble along the z axis, [m].
    f: scalar
        Frequency of the driving wave, [Hz].
    c_m: scalar
        Shear wave propagation velocity in the tissue matrix, [m/s].
    psi_ca: scalar
        Diameter oscillation amplitude for the bubble, [m].
    a_0: scalar
        Radius of gas bubble, [m].
    Lr: scalar
        Length of area in r direction, [m].
    Lz: scalar
        Length of area in z direction, [m].
    Nr: scalar
        Number of mesh points in r direction.
    Nz: scalar 
        Number of mesh points in z direction.
    rho_m: scalar 
        Mass density of tissue matrix, [kg/m³].
    rho_f: scalar 
        Mass density of tissue matrix, [kg/m³].
    mu_m: scalar
        Dynamic viscosity of the tissue matrix, [kg/(m*s)] = [Pa*s].
    eta_f: scalar
        Dynamic viscosity of the fluid, [Pa*s].
    dt: scalar
        The time step. Should be an order less than 1/f.
    Nc: scalar
        The number of cycles in the simulation.
    """

    
    r = np.linspace(0, Lr, Nr+2)                # array of mesh points in r direction
    z = np.linspace(0, Lz, Nz+2)                # array of mesh points in z direction
    dr = r[1] - r[0]
    dz = z[0] - z[0]

    T = 1/f                                     # time of one period 
    t = np.arange(0, Nc*T, dt)                  # array of mesh points in time

    u_fr = np.zeros((Nr+2,Nz+2))                # fluid velocity in r direction at next timestep
    u_fr_1 = np.zeros((Nr+2,Nz+2))              # at previous timestep t-dt
    u_fr_period = np.zeros((Nr+2,Nz+2, T/dt))

    u_fz = np.zeros((Nr+2,Nz+2))                # fluid velocity in z direction
    u_fz_1 = np.zeros((Nr+2,Nz+2))              # at previous timestep t-dt
    
    psi_mr = np.zeros((Nr+2,Nz+2))              # tissue matrix displacement in r direction at next timestep
    psi_mr_1 = np.zeros((Nr+2,Nz+2))            # at previous timestep t-dt
    psi_mr_2 = np.zeros((Nr+2,Nz+2))            # at previous timestep t-2*dt

    psi_mz = np.zeros((Nr+2,Nz+2))              # tissue matrix displacement in z direction at next timestep
    psi_mz_1 = np.zeros((Nr+2,Nz+2))            # at previous timestep t-dt
    psi_mz_2 = np.zeros((Nr+2,Nz+2))            # at previous timestep t-2*dt

    surface_vibrations_mr = problem.driving_conditions_mr(dt, f, psi_ca, z, L, D)
    surface_vibrations_fr = problem.driving_conditions_fr(dt, f, psi_ca, z, L, D)

    Np = int(round(T/float(dt)))                # number of timesteps in one period


    # main iteration loop
    # TODO: make this so that it uses a relaxation method that checks for each period if the system is stable?
    N = t.size
    for i in range(N): 
        t_index = i % Np
        psi_mr_1, u_fr_1 = set_driving_conditions(psi_mr_1, u_fr_1, z, surface_vibrations_mr[t_index, :], surface_vibrations_fr[t_index, :])


def radiation_force(rho_f, u_fr_period, u_fz_period, f, dt, mode):
    
    if mode == 'r':
        u = u_fr_period.copy()
    else:
        u = u_fz_period.copy()

    T = 1/f                             # time per period 
    Np = int(round(T/dt))               # number of timesteps per period 
    
    Rf = np.zeros((u_fr_period.shape[0]-2, u_fr_period[1]-2))



def advance_u_fr(u_fr, u_fr_1, delta_n_psi_mr, rho_f, eta_f, dt, r, z, gamma):
    
    dr = r[1] - r[0]
    dz = z[0] - z[0]

    A = rho_f/dt

    B = -u_fr[1:-1, 1:-1]/(2*dr) + eta_f(1/(2*r[1:-1]*dr)+ 1/(dr*dr))

    C = -gamma - eta_f/(r[1:-1]**2) - 2*eta_f/(dr**2) - 2*eta_f/(dz**2) + rho_f/dt

    D = u_fr[1:-1,1:-1]/(2*dr) + eta_f(1/(2*r[1:-1]*dr)+ 1/(dr*dr))

    E = -u_fr[1:-1,1:-1]/(2*dz) + eta_f/(dz**2)

    F = u_fr[1:-1,1:-1]/(2*dz) + eta_f/(dz**2)

    G = gamma/dt 

    u = u_fr[2:, 1:-1]*B + u_fr[1:-1, 1:-1]*C + u_fr[:-2, 1:-1]*D + 
        u_fr[1:-1, 2:]*E + u_fr[1:-1, :-2]*F + delta_n_psi_mr[1:-1,1:-1]*G

    u = u / A

    return u

def advance_u_fz(u_fr, u_fr_1, delta_n_psi_mz, rho_f, eta_f, dt, r, z, gamma):
    
    dr = r[1] - r[0]
    dz = z[0] - z[0]

    A = rho_f/dt

    B = -u_fr[1:-1, 1:-1]/(2*dr) + eta_f(1/(2*r[1:-1]*dr)+ 1/(dr*dr))

    C = -gamma - 2*eta_f/(dr**2) - 2*eta_f/(dz**2) + rho_f/dt

    D = u_fr[1:-1,1:-1]/(2*dr) + eta_f(1/(2*r[1:-1]*dr)+ 1/(dr*dr))

    E = -u_fr[1:-1,1:-1]/(2*dz) + eta_f/(dz**2)

    F = u_fr[1:-1,1:-1]/(2*dz) + eta_f/(dz**2)

    G = gamma/dt 


    u = u_fz[2:, 1:-1]*B + u_fz[1:-1, 1:-1]*C + u_fz[:-2, 1:-1]*D + 
        u_fz[1:-1, 2:]*E + u_fz[1:-1, :-2]*F + delta_n_psi_mz[1:-1,1:-1]*G

    u = u / A

    return u

def advance_psi_mr(psi_mr, psi_mr_1, psi_mz, u_fr, rho_m, mu_m, dt, r, z, gamma):
    
    dr = r[1] - r[0]
    dz = z[0] - z[0]

    A = rho_m/(dt**2) + gamma/dt

    B = -mu_m/(2*r[1:-1,1:-1]*dr)

    C = 2*rho_m/(dt**2) - mu_m/(r[1:-1]**2) - 2*mu_m/(dz**2) + gamma/dt 

    D = - B

    E = -mu_m/(dz**2)

    F = -E 

    G = -rho_m/(dt**2)

    psi = psi_mr[2:, 1:-1]*B + psi_mr[1:-1, 1:-1]*C + psi[:-2, 1:-1]*D +
          psi_mr[1:-1, 2:]*E + psi_mr[1:-1, :-2]*F + psi_mr_1[1:-1, 1:-1]*G 
          + mu_m/(2*dz)*( -1/r[1:-1]*(psi_mz[1:-1, 2:] - psi_mz[1:-1, :-2]) - 1/(2*dr)*
            (psi_mz[2:, 2:] - psi_mz[2:, :-2] - psi_mz[:-2, 2:] + psi_mz[:-2, :-2]))
          + u_fr[1:-1, 1:-1]*gamma

    psi = psi / A

    return psi 

def advance_psi_mz(psi_mr, psi_mz, psi_mz_1, u_fz, rho_m, mu_m, dt, r, z, gamma):
    
    dr = r[1] - r[0]
    dz = z[0] - z[0]

    A = rho_m/(dt**2) + gamma/dt

    B = mu_m/(dr**2) + mu_m/(2*r[1:-1]*dr)

    C = 2*rho_m/(dt**2) - 2*mu_m/(dr**2) + gamma/dt 

    D = mu_m/(dr**2) - mu_m/(2*r[1:-1]*dr)

    E = -rho_m/(dt**2)

     
    psi = psi_mz[2:, 1:-1]*B + psi_mz[1:-1,1:-1]*C + psi_mz[:-2, 1:-1]*D
          + mu_m/(2*dz)*( -1/r[1:-1]*(psi_mr[1:-1, 2:] - psi_mr[1:-1, :-2]) - 1/(2*dr)*
            (psi_mr[2:, 2:] - psi_mr[2:, :-2] - psi_mr[:-2, 2:] + psi_mr[:-2, :-2]))
          + gamma*u_fz[1:-1]
          

    psi = psi / A

    return psi 


def set_driving_conditions(psi_mr, u_fr, z_array, driving_conditions_mr, driving_conditions_fr):
    
    for i in range(z_array.size):
        index = np.min(np.where(z_array > driving_conditions_mr[i]))
        psi_mr[index, i] = driving_conditions_mr[i]
        u_fr[index, i] = driving_conditions_fr[i]

    return psi_mr, u_fr

    

