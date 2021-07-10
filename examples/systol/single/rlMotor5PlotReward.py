import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data4 = pd.read_csv("./rlMotor5Part4/monitor.csv", skiprows=[0])
data3 = pd.read_csv("./rlMotor5Part3/monitor.csv", skiprows=[0])
data2 = pd.read_csv("./rlMotor5Part2/monitor.csv", skiprows=[0])
data = pd.read_csv("./rlMotor5/monitor.csv", skiprows=[0])
num = data["r"].to_numpy()
num = np.append(num, data2["r"].to_numpy())
num = np.append(num, data3["r"].to_numpy())
num = np.append(num, data4["r"].to_numpy())
avgarr = []
for idx, item in enumerate(num):
    if idx > 0:
        avgarr.append(np.mean(num[idx:idx+50]))

plt.plot(num)
plt.plot(avgarr)
plt.savefig("r.png")
plt.show()
