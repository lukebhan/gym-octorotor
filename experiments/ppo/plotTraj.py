import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.loadtxt("x")
y = np.loadtxt("y")
xref = np.loadtxt("xref")
yref = np.loadtxt("yref")

plt.plot(x, y, label="Real")
plt.plot(xref, yref, label="Ref")
plt.legend()
plt.savefig("fig.png")
plt.show()

