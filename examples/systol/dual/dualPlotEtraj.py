import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rlx = np.loadtxt("EtrajX")
rly = np.loadtxt("EtrajY")

pidx = np.loadtxt("EtrajPidX")
pidy = np.loadtxt("EtrajPidY")

xrefarr = pd.read_csv("./Paths/EpathX3.csv", header=None).iloc[:, 1]
yrefarr = pd.read_csv("./Paths/EpathY3.csv", header=None).iloc[:, 1]
plt.plot(rlx, rly, label="RL Contoller")
plt.plot(pidx, pidy, label="PID Controller")
plt.plot(xrefarr, yrefarr, label="Reference")
plt.legend()
plt.show()


