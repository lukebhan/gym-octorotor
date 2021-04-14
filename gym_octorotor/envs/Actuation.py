from numpy.linalg import inv
import numpy as np
import math

class ControlAllocation:
    def __init__(self, ControlAllocationParams):
        b = ControlAllocationParams["b"]
        d = ControlAllocationParams["d"]
        l = ControlAllocationParams["l"]
        self.omegamax = ControlAllocationParams["omegamax"]
        sqrt2 = math.sqrt(2)
        self.AF = np.array([[2*b, 2*b, 2*b, 2*b], [b*l, 0, -b*l, 0], [-b*l, -sqrt2*b*l, b*l, sqrt2*b*l], [-2*d, 2*d, -2*d, 2*d]])

        self.A = np.array([[b, b, b, b, b, b, b, b], [b*l, sqrt2/2*b*l, 0, -sqrt2/2*b*l, -b*l, -sqrt2/2*b*l, 0, sqrt2/2*b*l], [0, -sqrt2/2*b*l, -b*l, -sqrt2/2*b*l, 0, sqrt2/2*b*l, b*l, sqrt2/2*b*l], [-d, d, -d, d, -d, d, -d,d]])

        self.invAF = inv(self.AF)
        self.E = np.array([[1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]]).transpose()

    def get_ref_velocity(self, udes):
        omega_s_f = self.invAF.dot(udes)
        omega_s = self.E.dot(omega_s_f)
        omega_ref = np.zeros(8, dtype="float32")
        for idx, item in enumerate(omega_s):
            omega_ref[idx] = np.sqrt(max(0, min(item, self.omegamax)))
        return omega_ref


    def get_u(self, omega):
        omega_s = np.square(omega)
        u = self.A.dot(omega_s)
        return u
