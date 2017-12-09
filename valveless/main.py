import constants
import numpy as np
import time

from solver import Solver
from problem import Problem


def main():
    print("Hello from main!")
    print("Setting up the problem...")

    L = constants.L
    D = constants.D
    f = constants.f
    psi_ca = constants.psi_ca
    a_0 = constants.a_0
    alpha_m = constants.alpha_m
    alpha_f = constants.alpha_f
    rho_t = constants.rho_t
    mu_m = constants.mu_m
    eta_f = constants.eta_f #small eta does to that it doesn't explode

    # gamma is "small"
    gamma = 1e-8*0

    problem = Problem(L, D, f, psi_ca ,a_0, alpha_m, rho_t, mu_m, eta_f, gamma)

    print("Creating and setting up the solver...")

    Lr = 100e-6 #m, i.e 100 microns
    Lz = 100e-6 #m
    # error / instability is inversely proportional with domain SIZE
    # this is due to the Dirichlet boundary conditions not being valid anymore for a "small" domain.

    Nr = 25
    Nz = 25
    dt = np.float64(1/(problem.f * 100))

    Nc = 6

    solver = Solver(problem, Lr, Lz, Nr, Nz, dt, Nc)

    print("CFL = ", dt / (solver.dr ** 2))
    time.sleep(2)

    print("Starting iterations...")

    solver.iterate()







if __name__ == "__main__":
    main()