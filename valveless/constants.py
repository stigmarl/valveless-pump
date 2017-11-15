#file that contains all physical constants 

# Length of the along bubble the z axis [m]
L = 50e-6

# Length of transition region [m]
D = L/3

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

# Shear stiffness the tissue matrix [kg/(m*s)] = [Pa*s]. 
mu_m = rho_m*c_m**2

# Dynamic viscosity of the fluid [Pa*s]. Same as that of blood
eta_f = 1.2e-3 

# 

