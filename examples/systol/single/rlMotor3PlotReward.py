import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data3 = pd.read_csv("./rlMotor3Part3/monitor.csv", skiprows=[0])
data2 = pd.read_csv("./rlMotor3Part2/monitor.csv", skiprows=[0])
data = pd.read_csv("./rlMotor3/monitor.csv", skiprows=[0])
num = data["r"].to_numpy()
num = np.append(num, data2["r"].to_numpy())
num = np.append(num, data3["r"].to_numpy())
avgarr = []
for idx, item in enumerate(num):
    if idx > 0:
        avgarr.append(np.mean(num[idx:idx+50]))

plt.plot(num, label="Reward for Epsiode")
plt.plot(avgarr, label = "Average Reward Over 50 Episodes")
plt.title("Reward of Single Motor Fault Controller")
plt.xlabel("Epsiode")
plt.ylabel("Reward")
plt.legend()
plt.show()
