#file that contains all physical constants 

# Length of the along bubble the z axis [m]
L = 50e-6

# Frequency of the driving wave [Hz]
f = 300e3

# Shear wave propagation velocity in the tissue matrix [m/s]
c_m = 10 

# Diameter oscillation amplitude for the bubble [m]
psi_ca = 1e-6

# Radius of gas bubble [m]
a_0 = 5e-6

# Relative volumes for tissue matrix and fluid respectively []
alpha_m = 0.3
alpha_f = 1 - alpha_m

# Tissue mass density, including tissue matrix and fluid [kg/m³]
rho_t = 1e3

# Mass density of tissue matrix and fluid [kg/m³]
rho_m = alpha_m*rho_t
rho_f = alpha_f*rho_t

# Dynamic viscosity of the tissue matrix [kg/(m*s)] = [Pa*s]
mu_m = 0

# Dynamic viscosity of the fluid [Pa*s]
eta_f = 0 
