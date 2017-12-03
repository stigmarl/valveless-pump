import numpy as np
import matplotlib
import matplotlib.pyplot as plt

params = {
    'text.usetex': True,
    'legend.fontsize': 'x-large',
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize':12,
    'ytick.labelsize':12,
}

matplotlib.rcParams.update(params)

f = 1
omega = 2*np.pi*f
T = 1/f
dt = T/100

n = 3

t = np.arange(0,7*T,dt)

fig = plt.figure(0)


sub1 = fig.add_subplot(211)
y = np.cos(omega*t)*0.5*(1 + np.tanh((t-n*T)/T))

sub1.plot(y)
sub1.set_xlabel('number of iterations')
sub1.set_ylabel('amplitude')
sub1.set_title('Modified tissue boundary condition')
sub1.grid()


sub2 = fig.add_subplot(212)
dy = omega*np.cos(omega*t)*0.5*(1 + np.tanh((t-n*T)/T)) + np.sin(omega*t)*0.5*1/T*(1 - np.tanh((t-n*T)/T)**2)

sub2.plot(dy)
sub2.set_xlabel('number of iterations')
sub2.set_ylabel('amplitude')
sub2.set_title('Modified fluid velocity boundary condition')
sub2.grid()



"""
fig1 = plt.figure(1)
sub3 = fig1.add_subplot(111)
h = 0.5*(1+np.tanh((t-n*T)/T))
sub3.plot(h)
sub3.set_xlabel('number of iterations')
sub3.set_ylabel('amplitude')
sub3.set_title('Dampening term')
sub3.grid()
fig1.savefig('dampening_term.eps', format='eps',dpi=1000)
"""
plt.tight_layout()
fig.savefig('modified_boundary_conditions.eps', format='eps', dpi=1000)
plt.show()