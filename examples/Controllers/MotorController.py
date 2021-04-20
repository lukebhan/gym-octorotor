# Motor Contoller Class
# Inherits from the controller abc.
# Implements a motor controller that generates voltage for a BLDCM to achieve target angular velocity

import numpy as np
from .Controller import Controller

class MotorController(Controller):
    # Takes in as parameters:
    # R - the resistance of Motor
    # d - the modelled drag moment
    # km - the mechanical motor constant
    # ke - electrical motor constant
    # maxv - max voltage
    # komega - Proportional motor gain
    def __init__(self, ControllerArgs):
        self.R = ControllerArgs["R"]
        self.d = ControllerArgs["d"]
        self.km = ControllerArgs["km"]
        self.ke = ControllerArgs["ke"]
        self.maxv = ControllerArgs["maxv"]
        i8 = np.identity(8)
        self.komega = ControllerArgs["komega"]*i8

    # Outputs the voltage vector for each motor
    # The currentState object is the current angular velocity of each motor (8 dim vector)
    # The targetValue is the target angular velocity of each motor (8 dim vector)
    def output(self, currentState, targetValue):
        vff = self.ke*targetValue + (self.R*self.d / self.km)*np.array(np.square(targetValue))
        v = self.komega.dot(targetValue - currentState) + vff
        v_const = np.zeros(8)
        for idx, item in enumerate(v):
            v_const[idx]  = max(0, min(self.maxv, item))
        return v_const
