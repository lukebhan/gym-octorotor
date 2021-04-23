import gym
import time
from Controllers.AltitudeController import AltitudeController
from Controllers.AttitudeController import AttitudeController
from Controllers.PositionController import PositionController
from Controllers.MotorController import MotorController
from Motors.BLDCM import BLDCM
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gym_octorotor
import numpy as np
import pandas as pd
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
        "kdx":  0.1,
        "kpy": 0.5,
        "kdy": 0.1,
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
    #motor.update_r(1.9984, 2)
    OctorotorParams["motor"] = motor
    OctorotorParams["motorController"] = motorc
    OctorotorParams["positionController"] = posc
    OctorotorParams["attitudeController"] = attc
    OctorotorParams["altitudeController"] = altc
    OctorotorParams["total_step_count"] = 5000
    OctorotorParams["reward_discount"] = 1

    env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
    model = PPO('MlpPolicy', env, verbose=1)
    model.save("ppo_cartpole")
    steps = 0
    model = PPO.load("ppo_cartpole")
    a = np.zeros(12)
    error = np.zeros(int(5000/50))
    while steps < 5000:
        action = np.random.uniform(low=-1, high=1, size=4)
        a,b,c,d = env.step(action)
        if steps % 50 == 0:
            error[int(steps/50)] = d
        steps += 1
        env.render()
    np.savetxt("errorTestBase.txt", error)
    
