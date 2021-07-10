from numpy.linalg import inv
import numpy as np
import math
from numba import jit

@jit
def _get_ref_velocity(udes, invAF, E, omegamax):
    omega_s_f = invAF.dot(udes)
    omega_s = E.dot(omega_s_f)
    omega_ref = np.zeros(8, dtype="float32")
    for idx, item in enumerate(omega_s):
        omega_ref[idx] = math.sqrt(max(0, min(item, omegamax)))
    return omega_ref

@jit
def _get_u(omega, A):
    return A.dot(np.square(omega.astype(np.float32)))

class ControlAllocation:
    def __init__(self, ControlAllocationParams):
        b = ControlAllocationParams["b"]
        d = ControlAllocationParams["d"]
        l = ControlAllocationParams["l"]
        self.omegamax = np.float32(ControlAllocationParams["omegamax"])
        sqrt2 = math.sqrt(2)
        self.AF = np.array([[2*b, 2*b, 2*b, 2*b], [b*l, 0, -b*l, 0], [-b*l, -sqrt2*b*l, b*l, sqrt2*b*l], [-2*d, 2*d, -2*d, 2*d]], dtype="float32")

        self.A = np.array([[b, b, b, b, b, b, b, b], [b*l, sqrt2/2*b*l, 0, -sqrt2/2*b*l, -b*l, -sqrt2/2*b*l, 0, sqrt2/2*b*l], [0, -sqrt2/2*b*l, -b*l, -sqrt2/2*b*l, 0, sqrt2/2*b*l, b*l, sqrt2/2*b*l], [-d, d, -d, d, -d, d, -d,d]], dtype="float32")

        self.invAF = inv(self.AF)
        self.E = np.array([[1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]], dtype="float32").transpose()

    def get_ref_velocity(self, udes):
        return _get_ref_velocity(udes, self.invAF, self.E, self.omegamax)


    def get_u(self, omega):
        return _get_u(omega, self.A)
