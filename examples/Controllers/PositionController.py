# Position Controller Class
# Inherits from the controller abc
# Implements a simple pid controller to alter reference values for moving the rotor to a target xy position

import math
from .Controller import Controller

class PositionController:
    # Takes in as parameters:
    # kpx - PID proportional gain for x direction
    # kdx - PID derivative gain for x direction
    # kpy - PID proportional gain for y direction
    # kdy - PID derivative gain for y direction
    # max_angle - Max angle turn of octorotor
    # min_angle - Min angle turn of octorotor
    def __init__(self, ControllerArgs):
        self.kpx = ControllerArgs["kpx"]
        self.kdx = ControllerArgs["kdx"]
        self.kpy = ControllerArgs["kpy"]
        self.kdy = ControllerArgs["kdy"]
        self.min_angle = ControllerArgs["min_angle"]
        self.max_angle = ControllerArgs["max_angle"]
    
    # Outputs reference angles for target position
    # The Current State Object is the Octorotors current state consisting of Pos, Vel, Angle, AngleVel in 3 different directions each.
    # The targetValue consists of a target x and target y value
    def output(self, currentState, targetValues):
        xref = targetValues["xref"]
        yref = targetValues["yref"]
        psi = currentState[11]
        x = currentState[0]
        y = currentState[1]
        print(x)
        xdot = currentState[3]
        ydot = currentState[4]
        xerror = xref - x
        yerror = yref - y
        cosPsi = math.cos(psi)
        sinPsi = math.sin(psi)
        xErrorBodyFrame = xerror*cosPsi + yerror*sinPsi
        yErrorBodyFrame = yerror*cosPsi - xerror*sinPsi
        theta_des = (xErrorBodyFrame-xdot)*self.kpx - xdot*self.kdx
        phi_des = -1*((yErrorBodyFrame-ydot)*self.kpy - ydot*self.kdy)
        theta_des = min(max(self.min_angle, theta_des), self.max_angle)
        phi_des = min(max(self.min_angle, phi_des), self.max_angle)
        return theta_des, phi_des

    def update_params(self, action):
        self.kpx = action[0]
        self.kdx = action[1]
        self.kpy = action[2]
        self.kdy = action[3]
