import gym
import time
from Controllers.AltitudeController import AltitudeController
from Controllers.AttitudeController import AttitudeController
from Controllers.PositionController import PositionController
from Controllers.MotorController import MotorController
from Motors.BLDCM import BLDCM
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import gym_octorotor
import torch 
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
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
    log_dir = "rl4/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
    env = Monitor(env, log_dir)
    #policy_kwargs = dict( net_arch=dict(pi=[64], qf=[64]))
    checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./rl4/',
                                             name_prefix='model')
    model = PPO.load("./rl3/model_1000_steps", env=env,verbose=1)
    #model = PPO("MlpPolicy", env=env, verbose=1, n_steps=2)
    model.learn(1000, callback = checkpoint_callback)
