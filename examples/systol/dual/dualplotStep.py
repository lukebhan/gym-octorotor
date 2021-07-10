import numpy as np
import matplotlib.pyplot as plt

rlx = np.loadtxt("dualX0.5.txt")
rly = np.loadtxt("dualY0.5.txt")
pidx = np.loadtxt("dualmotorstepresponsepid0.5X")
pidy = np.loadtxt("dualmotorstepresponsepid0.5Y")

plt.figure()
plt.xlabel("Time (0.001 Sec)")
plt.ylabel("Position (m)")
plt.title("Controller Comparison for Resistance 2x Fault - X direction")
print(rlx)
plt.plot(rlx, label="Reinforcement Learning Controller")
plt.plot(pidx, label="Nominal PID Controller")
plt.plot(np.full(len(rlx), 5), label="Reference")
plt.legend()
plt.show()

plt.figure()
plt.xlabel("Time (0.001 Sec)")
plt.ylabel("Position (m)")
plt.title("Controller Comparison for Resistance 2x Fault - Y direction")
plt.plot(rly, label="Reinforcement Learning Controller")
plt.plot(pidy, label="Nominal PID Controller")
plt.plot(np.full(len(rly), 5), label="Reference")
plt.legend()
plt.show()
