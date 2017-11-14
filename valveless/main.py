import constants
import numpy as np 

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
    eta_f = constants.eta_f

    # gamma is "small"
    gamma = 1e-5

    problem = Problem(L, D, f, psi_ca ,a_0, alpha_m, rho_t, mu_m, eta_f, gamma)

    print("Creating and setting up the solver...")

    Lr = 10 #mm, i.e 100 microns
    Lz = 10 #mm

    Nr = 1000
    Nz = 1001

    dt = 1/(problem.f * 21)

    Nc = 100

    solver = Solver(problem, Lr, Lz, Nr, Nz, dt, Nc)

    print("Starting iterations...")

    solver.iterate()

if __name__ == "__main__":
    main()