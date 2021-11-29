# Altitude Controller Class
# Inherits from the controller abc. 
# Implements a simple pid controller for the thrust of an octorotor

from .Controller import Controller

class AltitudeController(Controller):
    # Takes in as parameters:
    # m0 - the weight of octorotor
    # g - gravity constant
    # kdz - PID derivative gain
    # kpz - PID proportional gain
    def __init__(self, ControllerArgs):
        self.m0 = ControllerArgs['m0']
        self.g = ControllerArgs['g']
        self.kdz = ControllerArgs["kdz"]
        self.kpz = ControllerArgs["kpz"]

    # Outputs the thrust in the z direction
    # The Current State Object is the Octorotors current state consisting of Pos, Vel, Angle, AngleVel in 3 different directions each. We consider the z direction in this controller. 
    # The targetValue is the target z value
    def output(self, currentState, targetValue):
        return self.m0*(self.g - self.kdz*currentState[5]-self.kpz*(currentState[2]-targetValue))
