#file that contains all physical constants 

# Length of the along bubble the z axis [cm]
L = 50

# Length of transition region [mm]
D = L/3

# Frequency of the driving wave [kHz]
f = 300

# Shear wave propagation velocity in the tissue matrix [m/s]
c_m = 10 

# Diameter oscillation amplitude for the bubble [cm]
psi_ca = 1e-4

# Radius of gas bubble [cm]
a_0 = 5e-4

# Relative volumes for tissue matrix and fluid respectively []
alpha_m = 0.3
alpha_f = 1 - alpha_m

# Tissue mass density, including tissue matrix and fluid [kg/cm³]
rho_t = 1e3*1e-4

# Mass density of tissue matrix and fluid [kg/cm³]
rho_m = alpha_m*rho_t
rho_f = alpha_f*rho_t

# Shear stiffness the tissue matrix [kg/(m*s)] = [Pa*s]. 
mu_m = rho_m*c_m**2

# Dynamic viscosity of the fluid [Pa*s]. Same as that of blood
eta_f = 1.2e-3 

# 

