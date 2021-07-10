import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
mydict = {}
base = np.loadtxt("dualmotorrobustpid1.0.txt")
x = []
y = []
y2 = []
for i in range(len(arr)):
    y2.append(base[i][1])
    data = pd.read_csv("dualmotorrobustrl" + str(arr[i]) + ".txt", header=None)
    values = []
    for idx, item in enumerate(data.to_numpy()):
        values.append(float(item[0].split()[1]))
    data = []
    for item in values:
        if item < 100000:
            data.append(item)
    mydict[str(arr[i])] = data
    y.append(np.median(data))
    x.append(1+i)
fig1, ax1 = plt.subplots()
ax1.boxplot(mydict.values())
ax1.set_xticklabels(["2x", "2.25x", "2.5x", "2.75x", "3x", "3.25x", "3.5x", "3.75x", "4x"])
ax1.plot(x, y, label="RL Controller")
ax1.plot(x, y2, label="PID Controller")
plt.xlabel("Resistance Fault")
plt.ylabel("Sum of Absolute X and Y Error (m)")
plt.title("Robustness of Controller and Comparison to Nominal PID Controller For 2 Motor Faults")
plt.legend()
plt.show()
