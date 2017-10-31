import numpy as np


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

    
    r = np.linspace(0, Lr, Nr+1)            # array of mesh points in r direction
    z = np.linspace(0, Lz, Nz+1)            # array of mesh points in z direction
    dr = r[1] - r[0]
    dz = z[0] - z[0]

    T = 1/f                                 # time of one period 
    t = np.arange(0, Nc*T, dt)              # array of mesh points in time

    u_fr = np.zeros((Nr+1,Nz+1))            # fluid velocity in r direction at next timestep
    u_fr_1 = np.zeros((Nr+1,Nz+1))          # at previous timestep t-dt

    u_fz = np.zeros((Nr+1,Nz+1))            # fluid velocity in z direction
    u_fz_1 = np.zeros((Nr+1,Nz+1))          # at previous timestep t-dt
    
    psi_mr = np.zeros((Nr+1,Nz+1))          # tissue matrix displacement in r direction at next timestep
    psi_mr_1 = np.zeros((Nr+1,Nz+1))        # at previous timestep t-dt
    psi_mr_2 = np.zeros((Nr+1,Nz+1))        # at previous timestep t-2*dt

    psi_mz = np.zeros((Nr+1,Nz+1))          # tissue matrix displacement in z direction at next timestep
    psi_mz_1 = np.zeros((Nr+1,Nz+1))        # at previous timestep t-dt
    psi_mz_2 = np.zeros((Nr+1,Nz+1))        # at previous timestep t-2*dt


    # main iteration loop
    # TODO: make this so that it uses a relaxation method that checks for each period if the system is stable?
    N = t.size
    for i in range(N): 
        pass


def radiation_force(rho_f, u_fr, u_fz, mode):
    pass 

    


        