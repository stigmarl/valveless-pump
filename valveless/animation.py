import numpy as np

import solver
import problem

from matplotlib import mlab
from scipy import integrate

class Animation():

    def __init__(self, domain, points, solv, prob):
        self.l_r, self.l_z = domain

        self.Nr, self.Nz = points

        self.X, self.Y = np.mgrid[0:self.l_r:complex(self.Nr), 0:self.l_z:complex(self.Nz)]

        self.solv = solv
        self.prob = prob

        self.U = self.solv.u_fr
        self.V = self.solv.u_fz

        self.speed = np.sqrt(self.U**2 + self.V**2)


















