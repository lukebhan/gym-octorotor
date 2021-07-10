import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data3= pd.read_csv("./dualmotorPart3/monitor.csv", skiprows=[0])
data2= pd.read_csv("./dualmotorPart2/monitor.csv", skiprows=[0])
data = pd.read_csv("./dualmotor/monitor.csv", skiprows=[0])
num = data["r"].to_numpy()
num = np.append(num, data2["r"].to_numpy())
num = np.append(num, data3["r"].to_numpy())
avgarr = []
for idx, item in enumerate(num):
    if idx > 0:
        avgarr.append(np.mean(num[idx:idx+50]))
plt.title("Reward For 2 Motor Fault Controller")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(num, label="Reward for Epsiode")
plt.plot(avgarr, label="Average Reward Over 50 Epsiodes")
plt.legend()
plt.show()
