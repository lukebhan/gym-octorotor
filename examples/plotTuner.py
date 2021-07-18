import numpy as np
import matplotlib.pyplot as plt

realX = np.loadtxt('pidtunex')
realY = np.loadtxt('pidtuney')

refX = np.loadtxt('pidtunerefx')
refY = np.loadtxt('pidtunerefy')

estimateX = np.loadtxt('pidtuneestimatex')
estimateY = np.loadtxt('pidtuneestimatey')

plt.figure()
plt.scatter(realX, realY, label="Real Trajectory", s=3)
plt.scatter(refX, refY, label="Reference Trajectory", s=3)
#plt.scatter(estimateX, estimateY, label="Estimate Trajectory", s=3)
plt.xlabel("X direction (m)")
plt.ylabel("Y direction (m)")
plt.legend()
plt.show()

plt.figure()
plt.plot(abs(estimateX-realX) + abs(estimateY-realY), label="error")
plt.show()
