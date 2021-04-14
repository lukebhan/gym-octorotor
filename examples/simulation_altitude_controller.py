# This example is completely phyiscal based and does not use a reinforcement learning algorithm. It implements a PID controller in the Z direction and shows the position of the Octorotor via Matplot Lib. 

from Controllers.AltitudeController import AltitudeController
from Controllers.AttitudeController import AttitudeController
from Controllers.PositionController import PositionController
from Controllers.MotorController import MotorController
from Motors.BLDCM import BLDCM
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import gym
import math
import numpy as np
import gym_octorotor

# Simulation Parameters
g = 9.81
m0 = 2
d = 1.36787E-7
Ixx = 0.0429
Iyy = 0.0429
Izz = 0.0748

OctorotorParams = {
        "g": g,
        "m0": m0,
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Izz": Izz,
        "b": 8.54858E-6,
        "d": d,
        "l": 1,
        "omegamax": 600000,
        "dt": 0.001
}

MotorParams = {
        "Izzm": 2E-5,
        "km": 0.0107,
        "ke": 0.0107,
        "R": 0.2371,
        "d": d,
        "komega": 2,
        "maxv": 11.1
}

PositionParams = {
        "kpx": 0.2,
        "kdx":  0,
        "kpy": 0.2,
        "kdy": 0,
        "min_angle": -12*math.pi/180,
        "max_angle": 12*math.pi/180
}
J = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

AttitudeParams = {
        "kd": 20,
        "kp": 100, 
        "j" : J
}

AltitudeParams = {
        "g": g,
        "m0": m0,
        "kdz": 24, 
        "kpz": 144
}

if __name__ == "__main__":
    # Setup PID Controllers
    altc = AltitudeController(AltitudeParams)
    attc = AttitudeController(AttitudeParams)
    posc = PositionController(PositionParams)

    # Create Motor and motor controllers
    motor = BLDCM(MotorParams)
    motorc = MotorController(MotorParams)

    OctorotorParams["motor"] = motor
    OctorotorParams["motorController"] = motorc

    # storage for altitude arrays
    zref = 5
    zrefarr = []
    zarr = []
    psiref = np.zeros(3)

    env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
    env.reset()
    # Run envionrment passing in a control vector on each step
    for step in range(10000):
        state = env.get_state()
        zrefarr.append(zref)
        zarr.append(state[2])
        tau_des = attc.output(state, psiref)
        T_des = altc.output(state, zref)
        udes = np.array([T_des, tau_des[0], tau_des[1], tau_des[2]])
        env.step(udes)
    env.close()
    # Plot
    plt.plot(zrefarr, label="Zref")
    plt.plot(zarr, label="Z")
    plt.show()
