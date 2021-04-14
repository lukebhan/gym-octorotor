# Brushless DC motor
# Inherits the Motor based class
# Defines the dynamics of a standard dc motor

import numpy as np
import scipy.integrate 
from .Motor import Motor

class BLDCM(Motor):
    # Takes in as parameters:
    # km - mehcanical motor constant
    # ke - electrical motor constant
    # R - resistance
    # d - the modelled drage momment
    # Izzm - moment of inertia about z for a motor
    def __init__(self, motorArgs):
        self.km = motorArgs["km"]
        self.ke = motorArgs["ke"]
        self.R = motorArgs["R"]
        self.d = motorArgs["d"]
        self.Izzm = motorArgs["Izzm"]
        self.stepNum = 0
        self.omega = np.zeros(8, dtype="float32")
        self.ode = scipy.integrate.ode(self.omega_dot_i).set_integrator('vode',method='bdf')

    # Return a motors angular velocity moving one step in time with a given voltage
    def update(self, voltage, dt):
        self.stepNum += 1
        self.v = voltage
        self.ode.set_initial_value(self.omega, 0)
        self.omega = self.ode.integrate(self.ode.t + dt)
        return self.omega

    # Helper Method to calculate omega_dot for our ode integrator. Can be written as a lambda function inside update for other shorter motors.
    def omega_dot_i(self, time, state):
        t1 = self.km/self.R*self.v
        t2 = -self.d*self.omega**2
        t3 = -self.km*self.ke / self.R * self.omega
        domega = (t1 + t2 + t3)/self.Izzm
        return domega
