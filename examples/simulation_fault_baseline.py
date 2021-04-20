# This example is completely phyiscal based and does not use a reinforcement learning algorithm. It implements a PID controller for placing the octorotor at some x y position.

from Controllers.AltitudeController import AltitudeController
from Controllers.AttitudeController import AttitudeController
from Controllers.PositionController import PositionController
from Controllers.MotorController import MotorController
from Motors.BLDCM import BLDCM
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import gym
import math
import gym_octorotor
import sys

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

    OctorotorParams["motor"] = motor
    OctorotorParams["motorController"] = motorc

    # storage for altitude arrays
    zref = 0
    xrefarr = pd.read_csv("./Paths/EpathX.csv", header=None)[1].to_numpy()
    yrefarr = pd.read_csv("./Paths/EpathY.csv", header=None)[1].to_numpy()
    index = 1
    xref = xrefarr[0]
    yref = yrefarr[0]
    psiref = np.zeros(3)
    secondStep = 1
    total_steps = int(len(xrefarr)/2/OctorotorParams["dt"]*secondStep)
    percentFault = 0.94
    args = sys.argv
    i = args[1]
    crFile = "BaseLineCRMotor" + str(int(i)+1) + ".txt"
    arr = np.loadtxt(crFile)
    r = percentFault*arr[-1]
    print("Motor:", i, "Res:", r)
    env = gym.make('octorotor-v0', OctorotorParams=OctorotorParams)
    env.reset(OctorotorParams)
    xerror = np.empty(len(xrefarr))
    yerror = np.empty(len(xrefarr))
    euclid_error = np.empty(len(xrefarr))
    start = datetime.now()
    # Run envionrment passing in a control vector on each step
    count = 0
    print("Printing Simulation Parameters:")
    print("Time Step:", OctorotorParams["dt"])
    print("Second Defined as:", secondStep)
    print("Total Steps:", total_steps)
    count = 0
    resis = "BaseLineResisMotor" + str(int(i)+1) + ".txt"
    fault = "BaseLineMotor" + str(int(i)+1) + "Fault.txt"
    file = open(resis, "w")
    for step in range(total_steps):
        env.update_r(r, i)
        count += 1
        if(count % 50 == 0 and index != len(xrefarr)):
            file.write(str(r) + "\n")
            #print(index, "/", len(xrefarr))
            print("Time Elapsed:", datetime.now()-start)
            xerror[index-1] = abs(env.get_state()[0] - xref)
            yerror[index-1] = abs(env.get_state()[1] - yref)
            euclid_error[index-1] = (math.sqrt(xerror[index-1]*xerror[index-1] + yerror[index-1]*yerror[index-1]))
            xref = xrefarr[index]
            yref = yrefarr[index]
            index+=1
        #env.render(xref, yref)
        state = env.get_state()
        targetValues = {"xref": xref, "yref": yref}
        psiref[1], psiref[0] = posc.output(state, targetValues)
        tau_des = attc.output(state, psiref)
        T_des = altc.output(state, zref)
        udes = np.array([T_des, tau_des[0], tau_des[1], tau_des[2]], dtype="float32")
        env.step(udes)
    env.close()
    env.reset(OctorotorParams)
    np.savetxt(fault, euclid_error)
