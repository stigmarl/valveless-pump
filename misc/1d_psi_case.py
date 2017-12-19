import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import math

pgf_with_latex = {  # setup matplotlib to use latex for output# {{{
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "text.latex.unicode": True,
            "font.family": "serif",
            "font.serif": [],  # blank entries should cause plots
            "savefig.dpi": 200,
            "font.sans-serif": [],  # to inherit fonts from the document
            "font.monospace": [],
            "axes.labelsize": 18,  # LaTeX default is 10pt font.
            "font.size": 10,
            "legend.fontsize": 8,  # Make the legend/label fonts
            "xtick.labelsize": 15,  # a little smaller
            "ytick.labelsize": 15,
            #"figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts
                r"\usepackage[T1]{fontenc}",  # plots will be generated
                r"\usepackage[detect-all,locale=DE]{siunitx}",
            ]  # using this preamble
        }
matplotlib.rcParams.update(pgf_with_latex)

f0 = 300e3
dr = 1e-6
dt = 1e-7*0.64

psi_c  =1e-6

rho_m = 3e2
mu_m = 3e4

a0 = 5e-6
r = a0 + np.arange(40)*dr

T = 1/f0
T_final = 6*T

eta = 1.2e-3

psi = np.zeros((len(r)))
psi_1 = np.zeros((len(r)))


# implicit
psi_i = np.zeros((len(r)))
psi_i_1 = np.zeros((len(r)))
A = np.zeros((len(r), len(r)))



counter = 0

t = 0

M = mu_m/rho_m*dt**2/(dr**2)

print("M: ", M)

i,j = np.indices(A.shape)

A[i==j] = 1+M
A[i==j-1] = M/2
A[i==j+1] = -M/2
A[0,1:] = 0


"""
explicit case 

while t < T_final:

    print("max psi: ", max(psi[1:]))

    if math.isnan(max(psi[1:])):
        break


    #print("Counter: ", counter)

    psi[0] = psi_c*np.sin(2*np.pi*f0*t)

    psi_dr_r = 1/r[1:-1]* (psi[2:]-psi[:-2])/(2*dr)
    psi_rr = 1/(r[1:-1]**2)*psi[1:-1]

    psi_new = psi[1:-1]  -  mu_m/rho_m*(dt/dr)**2 * (psi_dr_r + psi_rr)

    t += dt

    psi_1[1:-1] = psi[1:-1]
    psi[1:-1] = psi_new

    counter+=1
"""

# implicit c

while t < T_final:

    psi_i[0] = psi_c*np.sin(2*np.pi*f0*t)

    B = 2*psi_i - psi_i_1

    psi_new = np.linalg.solve(A,B)

    psi_i_1 = psi_i
    psi_i = psi_new

    t += dt




#plt.plot(r[1:-1]*1e,psi_i[1:-1])

fig = plt.figure(0, figsize=(11,5))

sub1 = fig.add_subplot(121)
sub1.plot(r[1:-1]*1e6, psi_i[1:-1]*1e6)
sub1.set_xlabel(r'r [$\mathrm{\mu m}$]')
sub1.set_ylabel(r'$\Psi$ [$\mathrm{\mu m}$]')
sub1.set_title(r'1D tissue displacement $\Psi_{j}$ at $t = 6T$, $\beta = %.3f$' % M, fontsize=19)


t = 0

psi_i = np.zeros((len(r)))
psi_i_1 = np.zeros((len(r)))

M *= 1.1

A[i==j] = 1+M
A[i==j-1] = M/2
A[i==j+1] = -M/2
A[0,1:] = 0

while t < T_final:

    psi_i[0] = psi_c*np.sin(2*np.pi*f0*t)

    B = 2*psi_i - psi_i_1

    psi_new = np.linalg.solve(A,B)

    psi_i_1 = psi_i
    psi_i = psi_new

    t += dt


sub2 = fig.add_subplot(122)
sub2.plot(r[1:-1]*1e6, psi_i[1:-1]*1e6)
sub2.set_xlabel(r'r [$\mathrm{\mu m}$]')
sub2.set_ylabel(r'$\Psi$ [$\mathrm{\mu m}$]')
sub2.set_title(r'1D tissue displacement $\Psi_{j}$ at $t = 6T$, $\beta = %.3f$' % M, fontsize=19)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)

plt.show()


fig.savefig(fname='implicit_tissue_displacement.pdf', format='pdf', dpi=200)



