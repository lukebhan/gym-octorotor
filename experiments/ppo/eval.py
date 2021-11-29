import gym
import time
from Controllers.AltitudeController import AltitudeController
from Controllers.AttitudeController import AttitudeController
from Controllers.PositionController import PositionController
from Controllers.MotorController import MotorController
from Motors.BLDCM import BLDCM
from stable_baselines3 import SAC
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
    end = False
    xarr = []
    yarr = []
    xrefarr = []
    yrefarr = []
    rewardArr =[]
    env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
    policy_kwargs = dict(net_arch=dict(pi=[64, 64, 64, 64], qf=[64, 64, 64, 64]))
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.load('tmp/best_model16')
    errorArr = []
    for i in range(100):
        obs  = env.reset()
        totalReward = 0
        totalErr = 0
        end =False
        while not end:
            action = model.predict(obs)
            obs, reward, end, prints = env.step(action[0])
            totalReward += reward
            x = prints["x"]
            xref = prints["xref"]
            y = prints["y"]
            yref = prints["yref"]
            error = np.sqrt((x-xref)*(x-xref) + (y-yref)*(y-yref))
            totalErr +=  error
        rewardArr.append(totalReward)
        errorArr.append(totalErr)
    np.savetxt('reweval.txt',rewardArr)
    np.savetxt('error.txt', errorArr)