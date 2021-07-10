import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sac2= pd.read_csv("./ijcaisacPart2/monitor.csv", skiprows=[0])
sac = pd.read_csv("./ijcaisac/monitor.csv", skiprows=[0])
sacnum = sac["r"].to_numpy()
sacnum= np.append(sacnum, sac2["r"].to_numpy())

sac31= pd.read_csv("./ijcaisac1Part3/monitor.csv", skiprows=[0])
sac21= pd.read_csv("./ijcaisac1Part2/monitor.csv", skiprows=[0])
sac1 = pd.read_csv("./ijcaisac1/monitor.csv", skiprows=[0])
sacnum1 = sac1["r"].to_numpy()
sacnum1= np.append(sacnum1, sac21["r"].to_numpy())
sacnum1= np.append(sacnum1, sac31["r"].to_numpy())

sac32= pd.read_csv("./ijcaisac2Part3/monitor.csv", skiprows=[0])
sac22= pd.read_csv("./ijcaisac2Part2/monitor.csv", skiprows=[0])
sac2 = pd.read_csv("./ijcaisac2/monitor.csv", skiprows=[0])
sacnum2 = sac2["r"].to_numpy()
sacnum2= np.append(sacnum2, sac22["r"].to_numpy())
sacnum2= np.append(sacnum2, sac32["r"].to_numpy())

sac33= pd.read_csv("./ijcaisac2Part3/monitor.csv", skiprows=[0])
sac23= pd.read_csv("./ijcaisac3Part2/monitor.csv", skiprows=[0])
sac3 = pd.read_csv("./ijcaisac3/monitor.csv", skiprows=[0])
sacnum3 = sac3["r"].to_numpy()
sacnum3= np.append(sacnum3, sac23["r"].to_numpy())
sacnum3= np.append(sacnum3, sac33["r"].to_numpy())

sac34= pd.read_csv("./ijcaisac2Part3/monitor.csv", skiprows=[0])
sac24= pd.read_csv("./ijcaisac4Part2/monitor.csv", skiprows=[0])
sac4 = pd.read_csv("./ijcaisac4/monitor.csv", skiprows=[0])
sacnum4 = sac4["r"].to_numpy()
sacnum4= np.append(sacnum4, sac24["r"].to_numpy())
sacnum4= np.append(sacnum4, sac34["r"].to_numpy())

td32 = pd.read_csv("./ijcaitd3Part2/monitor.csv", skiprows=[0])
td3 = pd.read_csv("./ijcaitd3/monitor.csv", skiprows=[0])
td3num = td3["r"].to_numpy()
td3num = np.append(td3num, td32["r"].to_numpy())

td321 = pd.read_csv("./ijcaitd31Part2/monitor.csv", skiprows=[0])
td31 = pd.read_csv("./ijcaitd31/monitor.csv", skiprows=[0])
td3num1 = td31["r"].to_numpy()
td3num1 = np.append(td3num1, td321["r"].to_numpy())

td322 = pd.read_csv("./ijcaitd32Part2/monitor.csv", skiprows=[0])
td32 = pd.read_csv("./ijcaitd32/monitor.csv", skiprows=[0])
td3num2 = td32["r"].to_numpy()
td3num2 = np.append(td3num2, td322["r"].to_numpy())

td323 = pd.read_csv("./ijcaitd33Part2/monitor.csv", skiprows=[0])
td33 = pd.read_csv("./ijcaitd33/monitor.csv", skiprows=[0])
td3num3 = td33["r"].to_numpy()
td3num3 = np.append(td3num3, td323["r"].to_numpy())

td324 = pd.read_csv("./ijcaitd34Part2/monitor.csv", skiprows=[0])
td34 = pd.read_csv("./ijcaitd34/monitor.csv", skiprows=[0])
td3num4 = td34["r"].to_numpy()
td3num4 = np.append(td3num4, td324["r"].to_numpy())

ddpg2 = pd.read_csv("./ijcaiddpgPart2/monitor.csv", skiprows=[0])
ddpg = pd.read_csv("./ijcaiddpg/monitor.csv", skiprows=[0])
ddpgnum = ddpg["r"].to_numpy()
ddpgnum = np.append(ddpgnum, ddpg2["r"].to_numpy())

ddpg21 = pd.read_csv("./ijcaiddpg1Part2/monitor.csv", skiprows=[0])
ddpg1 = pd.read_csv("./ijcaiddpg1/monitor.csv", skiprows=[0])
ddpgnum1 = ddpg1["r"].to_numpy()
ddpgnum1 = np.append(ddpgnum1, ddpg21["r"].to_numpy())

ddpg22 = pd.read_csv("./ijcaiddpg2Part2/monitor.csv", skiprows=[0])
ddpg2 = pd.read_csv("./ijcaiddpg2/monitor.csv", skiprows=[0])
ddpgnum2 = ddpg2["r"].to_numpy()
ddpgnum2 = np.append(ddpgnum2, ddpg22["r"].to_numpy())

ddpg23 = pd.read_csv("./ijcaiddpg3Part2/monitor.csv", skiprows=[0])
ddpg3 = pd.read_csv("./ijcaiddpg3/monitor.csv", skiprows=[0])
ddpgnum3 = ddpg3["r"].to_numpy()
ddpgnum3 = np.append(ddpgnum3, ddpg23["r"].to_numpy())

ddpg24 = pd.read_csv("./ijcaiddpg4Part2/monitor.csv", skiprows=[0])
ddpg4 = pd.read_csv("./ijcaiddpg4/monitor.csv", skiprows=[0])
ddpgnum4 = ddpg4["r"].to_numpy()
ddpgnum4 = np.append(ddpgnum4, ddpg24["r"].to_numpy())

ppo3 = pd.read_csv("./rlMotor3Part3/monitor.csv", skiprows=[0])
ppo2 = pd.read_csv("./rlMotor3Part2/monitor.csv", skiprows=[0])
ppo = pd.read_csv("./rlMotor3/monitor.csv", skiprows=[0])
pponum = ppo["r"].to_numpy()
pponum = np.append(pponum, ppo2["r"].to_numpy())
pponum = np.append(pponum, ppo3["r"].to_numpy())

ppo21 = pd.read_csv("./ijcaippo1Part2/monitor.csv", skiprows=[0])
ppo1 = pd.read_csv("./ijcaippo1/monitor.csv", skiprows=[0])
pponum1 = ppo1["r"].to_numpy()
pponum1 = np.append(pponum1, ppo21["r"].to_numpy())

ppo22 = pd.read_csv("./ijcaippo2Part2/monitor.csv", skiprows=[0])
ppo2 = pd.read_csv("./ijcaippo2/monitor.csv", skiprows=[0])
pponum2 = ppo2["r"].to_numpy()
pponum2 = np.append(pponum2, ppo22["r"].to_numpy())

ppo23 = pd.read_csv("./ijcaippo3Part2/monitor.csv", skiprows=[0])
ppo3 = pd.read_csv("./ijcaippo3/monitor.csv", skiprows=[0])
pponum3 = ppo3["r"].to_numpy()
pponum3 = np.append(pponum3, ppo23["r"].to_numpy())

ppo24 = pd.read_csv("./ijcaippo4Part2/monitor.csv", skiprows=[0])
ppo4 = pd.read_csv("./ijcaippo4/monitor.csv", skiprows=[0])
pponum4 = ppo4["r"].to_numpy()
pponum4 = np.append(pponum4, ppo24["r"].to_numpy())

sacavg = []
sacavg2 = []
sacavg3 = []
sacavg4 = []
sacavg5 = []
ddpgavg = []
ddpgavg2 = []
ddpgavg3 = []
ddpgavg4 = []
ddpgavg5 = []
td3avg = []
td3avg2 = []
td3avg3 = []
td3avg4 = []
td3avg5 = []
ppoavg  = [] 
ppoavg2 = []
ppoavg3 = []
ppoavg4 = []
ppoavg5 = []

for idx, item in enumerate(ddpgnum):
    if idx > 0 and idx < len(ddpgnum)-50:
        sacavg.append(np.mean(sacnum[idx:idx+50]))
        sacavg2.append(np.mean(sacnum1[idx:idx+50]))
        sacavg3.append(np.mean(sacnum2[idx:idx+50]))
        sacavg4.append(np.mean(sacnum3[idx:idx+50]))
        sacavg5.append(np.mean(sacnum4[idx:idx+50]))

        ddpgavg.append(np.mean(ddpgnum[idx:idx+50]))
        td3avg.append(np.mean(td3num[idx:idx+50]))
        td3avg2.append(np.mean(td3num1[idx:idx+50]))
        td3avg3.append(np.mean(td3num2[idx:idx+50]))
        td3avg4.append(np.mean(td3num3[idx:idx+50]))
        td3avg5.append(np.mean(td3num4[idx:idx+50]))

        ppoavg.append(np.mean(pponum[idx:idx+50]))
        ppoavg2.append(np.mean(pponum1[idx:idx+50]))
        ppoavg3.append(np.mean(pponum2[idx:idx+50]))
        ppoavg4.append(np.mean(pponum3[idx:idx+50]))
        ppoavg5.append(np.mean(pponum4[idx:idx+50]))

        ddpgavg.append(np.mean(ddpgnum[idx:idx+50]))
        ddpgavg2.append(np.mean(ddpgnum1[idx:idx+50]))
        ddpgavg3.append(np.mean(ddpgnum2[idx:idx+50]))
        ddpgavg4.append(np.mean(ddpgnum3[idx:idx+50]))
        ddpgavg5.append(np.mean(ddpgnum4[idx:idx+50]))

sacmean = []
sactop = []
sacbottom = []

td3mean = []
td3top = []
td3bottom = []

ppomean = []
ppotop = []
ppobottom = []

ddpgmean = []
ddpgtop = []
ddpgbottom = []
for idx, item in enumerate(sacavg):
    td3m = np.mean([td3avg[idx], td3avg2[idx], td3avg3[idx], td3avg4[idx], td3avg5[idx]])
    td3mean.append(td3m)
    td3top.append(td3m + np.std([td3avg[idx], td3avg2[idx], td3avg3[idx], td3avg4[idx], td3avg5[idx]]))
    td3bottom.append(td3m - np.std([td3avg[idx], td3avg2[idx], td3avg3[idx], td3avg4[idx], td3avg5[idx]]))

    sacm = np.mean([sacavg[idx], sacavg2[idx], sacavg3[idx], sacavg4[idx], sacavg5[idx]])
    sacmean.append(sacm)
    sactop.append(sacm + np.std([sacavg[idx], sacavg2[idx], sacavg3[idx], sacavg4[idx], sacavg5[idx]]))
    sacbottom.append(sacm - np.std([sacavg[idx], sacavg2[idx], sacavg3[idx], sacavg4[idx], sacavg5[idx]]))

    ppom = np.mean([ppoavg[idx], ppoavg2[idx], ppoavg3[idx], ppoavg4[idx], ppoavg5[idx]])
    ppomean.append(ppom)
    ppotop.append(ppom + np.std([ppoavg[idx], ppoavg2[idx], ppoavg3[idx], ppoavg4[idx], ppoavg5[idx]]))
    ppobottom.append(ppom - np.std([ppoavg[idx], ppoavg2[idx], ppoavg3[idx], ppoavg4[idx], ppoavg5[idx]]))

    ddpgm = np.mean([ddpgavg[idx], ddpgavg2[idx], ddpgavg3[idx], ddpgavg4[idx], ddpgavg5[idx]])
    ddpgmean.append(ddpgm)
    ddpgtop.append(ddpgm + np.std([ddpgavg[idx], ddpgavg2[idx], ddpgavg3[idx], ddpgavg4[idx], ddpgavg5[idx]]))
    ddpgbottom.append(ddpgm - np.std([ddpgavg[idx], ddpgavg2[idx], ddpgavg3[idx], ddpgavg4[idx], ddpgavg5[idx]]))


x = np.linspace(0, len(sacmean)-1, len(sacmean))
fig, ax = plt.subplots(1)
plt.xlabel("Episode")
plt.ylabel("Average Reward of 50 Epsiodes")
plt.title("Reward Comparison of PID+RL Controller For Fault Tolerant Control")
ax.plot(x, sacmean, color='purple', label="PID Controller + SAC")
ax.fill_between(x, sactop, sacbottom, facecolor='purple', alpha=0.3)
ax.plot(x, td3mean, color='red', label="PID Controller + TD3")
ax.fill_between(x, td3top, td3bottom, facecolor='red', alpha=0.1)
ax.plot(x, ppomean, color='green', label="PID Controller + PPO")
ax.fill_between(x, ppotop, ppobottom, facecolor='green', alpha=0.3)
ax.plot(x, ddpgmean, color='blue', label="PID Controller + DDPG")
ax.fill_between(x, ddpgtop, ddpgbottom, facecolor='blue', alpha=0.1)
plt.legend()
plt.show()

