# Attitude Controller Class
# Inherits from the controller abc.
# Implements a simple pid controller for the torque of an octorotor

import numpy as np 
from .Controller import Controller

class AttitudeController(Controller):
    # Takes in as parameters:
    # j matrix - the moment of inertia matrix for the octorotor
    # kd - PID derivative gain
    # kp - PID proportional gain
    def __init__(self, ControllerArgs):
        self.J = ControllerArgs["j"]
        i3 = np.identity(3)
        self.resetkd = ControllerArgs["kd"]*i3
        self.resetkp = ControllerArgs["kp"]*i3
        self.kd = ControllerArgs["kd"]*i3
        self.kp = ControllerArgs["kp"]*i3

    # Outputs the torque for the octorotor
    # The current State objects is the Ocotorotrs current state consisting of Pos, Vel, Angle, AngleVel in 3 different directions each. 
    # The targetValue is the desired angularVelocity
    def output(self, currentState, targetValue):
        return -self.J.dot(self.kp.dot(currentState[6:9]-targetValue)+self.kd.dot(currentState[9:12]))

    def update_params(self, action):
        if action[0] == 0:
            self.kp += 0.05
        elif action[0] == 1:
            self.kp -= 0.05
        if action[1] == 0:
            self.kd += 0.01
        elif action[1] == 1:
            self.kd -= 0.01

    def reset(self):
        self.kd = self.resetkd
        self.kp = self.resetkp



