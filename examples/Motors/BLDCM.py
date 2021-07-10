# Brushless DC motor
# Inherits the Motor based class
# Defines the dynamics of a standard dc motor

import numpy as np
import scipy.integrate 
from .Motor import Motor
from numba import jit

@jit
def f(km, R, v, d, omega, ke, newR, Izzm, idx, vidx, omegaidx, idx2, vidx2, omegaidx2, newR2):
    res =  (km/R*v-d*omega*omega -km*ke /R * omega)/Izzm
    residx1 = (km/newR*vidx-d*omegaidx*omegaidx - km*ke/newR * omegaidx)/Izzm
    residx2 = (km/newR2*vidx2 - d*omegaidx2*omegaidx2 - km*ke/newR2 * omegaidx2)/Izzm
    return res, residx1, residx2


class BLDCM(Motor):
    # Takes in as parameters:
    # km - mehcanical motor constant
    # ke - electrical motor constant
    # R - resistance
    # d - the modelled drage momment
    # Izzm - moment of inertia about z for a motor
    def __init__(self, motorArgs):
        self.motorArgs = motorArgs
        self.km = np.float32(motorArgs["km"])
        self.ke = np.float32(motorArgs["ke"])
        self.R = np.float32(motorArgs["R"])
        self.newR = np.float32(motorArgs["R"])
        self.d = np.float32(motorArgs["d"])
        self.Izzm = np.float32(motorArgs["Izzm"])
        self.stepNum = 0
        self.omega = np.zeros(8, dtype="float32")
        self.ode = scipy.integrate.ode(self.omega_dot_i).set_integrator('vode', method='bdf')
        self.idx = 0
        self.idx2 = 0
        self.newR2 = self.newR

    def reset(self):
        motorArgs = self.motorArgs
        self.km = np.float32(motorArgs["km"])
        self.ke = np.float32(motorArgs["ke"])
        self.R = np.float32(motorArgs["R"])
        self.newR = np.float32(motorArgs["R"])
        self.d = np.float32(motorArgs["d"])
        self.Izzm = np.float32(motorArgs["Izzm"])
        self.stepNum = 0
        self.omega = np.zeros(8, dtype="float32")
        self.ode = scipy.integrate.ode(self.omega_dot_i).set_integrator('vode', method='bdf')
        self.idx = 0
        self.idx2 = 0
        self.newR2 = self.newR

    # Return a motors angular velocity moving one step in time with a given voltage
    def update(self, voltage, dt):
        self.stepNum += 1
        self.v = voltage
        self.ode.set_initial_value(self.omega, 0)
        self.omega = self.ode.integrate(self.ode.t + dt)
        return self.omega

    # Helper Method to calculate omega_dot for our ode integrator. Can be written as a lambda function inside update for other shorter motors.
    def omega_dot_i(self, time, state):
        res, residx, residx2 =  f(self.km, self.R, self.v, self.d, self.omega, self.ke, self.newR, self.Izzm, self.idx, self.v[self.idx], self.omega[self.idx], self.idx2, self.v[self.idx2], self.omega[self.idx2], self.newR2)
        res[self.idx] = residx
        res[self.idx2] = residx2
        return res

    def update_r(self, r, idx):
        self.idx = int(idx)
        self.newR = r

    def update_r2(self, r, idx):
        self.idx2 = int(idx)
        self.newR2 = r
