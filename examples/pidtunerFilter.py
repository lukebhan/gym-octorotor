import gym
import time
from Controllers.AltitudeController import AltitudeController
from Controllers.AttitudeController import AttitudeController
from Controllers.PositionController import PositionController
from Controllers.MotorController import MotorController
from Motors.BLDCM import BLDCM
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import gym_octorotor
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

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
        "dt": 0.01
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
        "kpx": 0.5,
        "kdx":  .1,
        "kpy": 0.5,
        "kdy": .1,
        "min_angle": -12*math.pi/180,
        "max_angle": 12*math.pi/180
}
J = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

AttitudeParams = {
        "kd": 10,
        "kp": 50,
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
    OctorotorParams["positionController"] = posc
    OctorotorParams["attitudeController"] = attc
    OctorotorParams["altitudeController"] = altc
    OctorotorParams["total_step_count"] = 5000
    OctorotorParams["reward_discount"] = 1
    resistance = np.full(8, 0.2371)
    OctorotorParams["resistance"] = resistance
    f = open("pidtunex", "w")
    f3 = open("pidtuneref", "w")
    f2 = open("pidtuney", "w")
    f5 = open("pidtunerefy", "w")
    f4 = open("pidtunerefx", "w")
    f6 = open("pidtuneestimatex", "w")
    f7 = open("pidtuneestimatey", "w")
    f8 = open("psi0Ref", "w")
    f9 = open("psi1ref", "w")
    env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
    for i in range(1):
        print(i)
        xerror = 0
        yerror = 0
        obs = env.reset()
        
        #model = PPO.load("./dualmotorPart3/model_1000_steps")
        #act = model.predict([obs])[0]
        _, rew, done, err = env.step([0.2, 0.1, 0.2, 0.1])
        error = np.sum(abs(np.array(err["xerror"])-5)) + np.sum(abs(np.array(err["yerror"]) -5))
        for x in err["xerror"]:
            f.write(str(x)  + "\n")
        for y in err["yerror"]:
            f2.write(str(y)  + "\n")
            f3.write(str(5) + "\n")
        for x in err["xref"]:
            f4.write(str(x)  + "\n")
        for x in err["yref"]:
            f5.write(str(x)  + "\n")
        for x in err["xestimate"]:
            f6.write(str(x)  + "\n")
        for x in err["yestimate"]:
            f7.write(str(x)  + "\n")
        for x in err["psi0"]:
            f8.write(str(x) + "\n")
        for x in err["psi1"]:
            f8.write(str(x) + "\n")
