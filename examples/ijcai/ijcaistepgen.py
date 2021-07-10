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
from stable_baselines3 import PPO, DDPG, SAC, TD3
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
    OctorotorParams["positionController"] = posc
    OctorotorParams["attitudeController"] = attc
    OctorotorParams["altitudeController"] = altc
    OctorotorParams["total_step_count"] = 5000
    OctorotorParams["reward_discount"] = 1
    resistance = np.full(8, 0.2371)
    resistance[2] = 1.9984
    OctorotorParams["resistance"] = resistance
    env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = PPO.load("./ijcaippo1Part2/model_1000_steps")
        act = model.predict([obs])[0]
        print(act)
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiPPO1X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiPPO1Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = DDPG.load("./ijcaiddpg1Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiDDPG1X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiDDPG1Y1.7.txt", np.array(err["yerror"]))


    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = TD3.load("./ijcaitd31Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiTD31X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiTD31Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = SAC.load("./ijcaisac1Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiSAC1X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiSAC1Y1.7.txt", np.array(err["yerror"]))


    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = PPO.load("./ijcaippo2Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiPPO2X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiPPO2Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = DDPG.load("./ijcaiddpg2Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiDDPG2X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiDDPG2Y1.7.txt", np.array(err["yerror"]))


    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = TD3.load("./ijcaitd32Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiTD32X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiTD32Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = SAC.load("./ijcaisac2Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiSAC2X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiSAC2Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = PPO.load("./ijcaippo3Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiPPO3X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiPPO3Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = DDPG.load("./ijcaiddpg3Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiDDPG3X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiDDPG3Y1.7.txt", np.array(err["yerror"]))


    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = TD3.load("./ijcaitd33Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiTD33X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiTD33Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = SAC.load("./ijcaisac3Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiSAC3X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiSAC3Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = PPO.load("./ijcaippo4Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiPPO4X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiPPO4Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = DDPG.load("./ijcaiddpg4Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiDDPG4X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiDDPG4Y1.7.txt", np.array(err["yerror"]))


    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = TD3.load("./ijcaitd34Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiTD34X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiTD34Y1.7.txt", np.array(err["yerror"]))

    for i in range(1):
        xerror = 0
        yerror = 0
        obs = env.reset(1.7)
        
        model = SAC.load("./ijcaisac4Part2/model_1000_steps")
        act = model.predict([obs])[0]
        obs, rew, done, err = env.step(act)
        np.savetxt("ijcaiSAC4X1.7.txt", np.array(err["xerror"]))

        np.savetxt("ijcaiSAC4Y1.7.txt", np.array(err["yerror"]))
