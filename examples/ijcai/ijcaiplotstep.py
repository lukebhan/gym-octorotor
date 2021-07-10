import numpy as np 
import matplotlib.pyplot as plt

pidx = np.loadtxt("pidStepResponseX1.7.txt")
pidy = np.loadtxt("pidStepResponseY1.7.txt")

sacx = np.loadtxt("ijcaiSACX1.7.txt")
sacy = np.loadtxt("ijcaiSACY1.7.txt")
sacx1 = np.loadtxt("ijcaiSAC1X1.7.txt")
sacy1 = np.loadtxt("ijcaiSAC1Y1.7.txt")
sacx2 = np.loadtxt("ijcaiSAC2X1.7.txt")
sacy2 = np.loadtxt("ijcaiSAC2Y1.7.txt")
sacx3 = np.loadtxt("ijcaiSAC3X1.7.txt")
sacy3 = np.loadtxt("ijcaiSAC3Y1.7.txt")
sacx4 = np.loadtxt("ijcaiSAC4X1.7.txt")
sacy4 = np.loadtxt("ijcaiSAC4Y1.7.txt")

ppox = np.loadtxt("ijcaiPPOX1.7.txt")
ppoy = np.loadtxt("ijcaiPPOY1.7.txt")
ppox1 = np.loadtxt("ijcaiPPO1X1.7.txt")
ppoy1 = np.loadtxt("ijcaiPPO1Y1.7.txt")
ppox2 = np.loadtxt("ijcaiPPO2X1.7.txt")
ppoy2 = np.loadtxt("ijcaiPPO2Y1.7.txt")
ppox3 = np.loadtxt("ijcaiPPO3X1.7.txt")
ppoy3 = np.loadtxt("ijcaiPPO3Y1.7.txt")
ppox4 = np.loadtxt("ijcaiPPO4X1.7.txt")
ppoy4 = np.loadtxt("ijcaiPPO4Y1.7.txt")

ddpgx = np.loadtxt("ijcaiDDPGX1.7.txt")
ddpgy = np.loadtxt("ijcaiDDPGY1.7.txt")
ddpgx1 = np.loadtxt("ijcaiDDPG1X1.7.txt")
ddpgy1 = np.loadtxt("ijcaiDDPG1Y1.7.txt")
ddpgx2 = np.loadtxt("ijcaiDDPG2X1.7.txt")
ddpgy2 = np.loadtxt("ijcaiDDPG2Y1.7.txt")
ddpgx3 = np.loadtxt("ijcaiDDPG3X1.7.txt")
ddpgy3 = np.loadtxt("ijcaiDDPG3Y1.7.txt")
ddpgx4 = np.loadtxt("ijcaiDDPG4X1.7.txt")
ddpgy4 = np.loadtxt("ijcaiDDPG4Y1.7.txt")

td3x = np.loadtxt("ijcaiTD3X1.7.txt")
td3y = np.loadtxt("ijcaiTD3Y1.7.txt")
td3x1 = np.loadtxt("ijcaiTD31X1.7.txt")
td3y1 = np.loadtxt("ijcaiTD31Y1.7.txt")
td3x2 = np.loadtxt("ijcaiTD32X1.7.txt")
td3y2 = np.loadtxt("ijcaiTD32Y1.7.txt")
td3x3 = np.loadtxt("ijcaiTD33X1.7.txt")
td3y3 = np.loadtxt("ijcaiTD33Y1.7.txt")
td3x4 = np.loadtxt("ijcaiTD34X1.7.txt")
td3y4 = np.loadtxt("ijcaiTD34Y1.7.txt")

td3meanX = []
td3meanY = []
td3topX = []
td3topY = []
td3bottomX = []
td3bottomY = []
for idx, item in enumerate(td3x):
    td3xm = abs(np.mean([td3x[idx], td3x1[idx], td3x2[idx], td3x3[idx], td3x4[idx]])-5)
    td3ym = abs(np.mean([td3y[idx], td3y1[idx], td3y2[idx], td3y3[idx], td3y4[idx]]) - 5)
    td3meanX.append(td3xm)
    td3meanY.append(td3ym)
    td3topX.append(td3xm + np.std([abs(td3x[idx] -5), abs(td3x1[idx] - 5), abs(td3x2[idx] - 5), abs(td3x3[idx] - 5), abs(td3x4[idx]- 5)]))
    td3bottomX.append(td3xm - np.std([abs(td3x[idx] -5), abs(td3x1[idx] - 5), abs(td3x2[idx] - 5), abs(td3x3[idx] - 5), abs(td3x4[idx]- 5)]))
    td3bottomY.append(td3ym - np.std([abs(td3y[idx] -5), abs(td3y1[idx] - 5), abs(td3y2[idx] - 5), abs(td3y3[idx] - 5), abs(td3y4[idx]- 5)]))
    td3topY.append(td3ym + np.std([abs(td3y[idx] -5), abs(td3y1[idx] - 5), abs(td3y2[idx] - 5), abs(td3y3[idx] - 5), abs(td3y4[idx]- 5)]))
    for pos,obj in enumerate(td3bottomX):
        if obj < 0:
            td3bottomX[pos] = 0
    for pos,obj in enumerate(td3bottomY):
        if obj < 0:
            td3bottomY[pos] = 0

ddpgmeanX = []
ddpgmeanY = []
ddpgtopX = []
ddpgtopY = []
ddpgbottomX = []
ddpgbottomY = []
for idx, item in enumerate(ddpgx):
    ddpgxm = abs(np.mean([ddpgx[idx], ddpgx1[idx], ddpgx2[idx], ddpgx3[idx], ddpgx4[idx]])-5)
    ddpgym = abs(np.mean([ddpgy[idx], ddpgy1[idx], ddpgy2[idx], ddpgy3[idx], ddpgy4[idx]]) - 5)
    ddpgmeanX.append(ddpgxm)
    ddpgmeanY.append(ddpgym)
    ddpgtopX.append(ddpgxm + np.std([abs(ddpgx[idx] -5), abs(ddpgx1[idx] - 5), abs(ddpgx2[idx] - 5), abs(ddpgx3[idx] - 5), abs(ddpgx4[idx]- 5)]))
    ddpgbottomX.append(ddpgxm - np.std([abs(ddpgx[idx] -5), abs(ddpgx1[idx] - 5), abs(ddpgx2[idx] - 5), abs(ddpgx3[idx] - 5), abs(ddpgx4[idx]- 5)]))
    ddpgbottomY.append(ddpgym - np.std([abs(ddpgy[idx] -5), abs(ddpgy1[idx] - 5), abs(ddpgy2[idx] - 5), abs(ddpgy3[idx] - 5), abs(ddpgy4[idx]- 5)]))
    ddpgtopY.append(ddpgym + np.std([abs(ddpgy[idx] -5), abs(ddpgy1[idx] - 5), abs(ddpgy2[idx] - 5), abs(ddpgy3[idx] - 5), abs(ddpgy4[idx]- 5)]))
    for pos,obj in enumerate(ddpgbottomX):
        if obj < 0:
            ddpgbottomX[pos] = 0
    for pos,obj in enumerate(ddpgbottomY):
        if obj < 0:
            ddpgbottomY[pos] = 0

ppomeanX = []
ppomeanY = []
ppotopX = []
ppotopY = []
ppobottomX = []
ppobottomY = []
for idx, item in enumerate(ppox):
    ppoxm = abs(np.mean([ppox[idx], ppox1[idx], ppox2[idx], ppox3[idx], ppox4[idx]])-5)
    ppoym = abs(np.mean([ppoy[idx], ppoy1[idx], ppoy2[idx], ppoy3[idx], ppoy4[idx]]) - 5)
    ppomeanX.append(ppoxm)
    ppomeanY.append(ppoym)
    ppotopX.append(ppoxm + np.std([abs(ppox[idx] -5), abs(ppox1[idx] - 5), abs(ppox2[idx] - 5), abs(ppox3[idx] - 5), abs(ppox4[idx]- 5)]))
    ppobottomX.append(ppoxm - np.std([abs(ppox[idx] -5), abs(ppox1[idx] - 5), abs(ppox2[idx] - 5), abs(ppox3[idx] - 5), abs(ppox4[idx]- 5)]))
    ppobottomY.append(ppoym - np.std([abs(ppoy[idx] -5), abs(ppoy1[idx] - 5), abs(ppoy2[idx] - 5), abs(ppoy3[idx] - 5), abs(ppoy4[idx]- 5)]))
    ppotopY.append(ppoym + np.std([abs(ppoy[idx] -5), abs(ppoy1[idx] - 5), abs(ppoy2[idx] - 5), abs(ppoy3[idx] - 5), abs(ppoy4[idx]- 5)]))
    for pos,obj in enumerate(ppobottomX):
        if obj < 0:
            ppobottomX[pos] = 0
    for pos,obj in enumerate(ppobottomY):
        if obj < 0:
            ppobottomY[pos] = 0

sacmeanX = []
sacmeanY = []
sactopX = []
sactopY = []
sacbottomX = []
sacbottomY = []
for idx, item in enumerate(sacx):
    sacxm = abs(np.mean([sacx[idx], sacx1[idx], sacx2[idx], sacx3[idx], sacx4[idx]])-5)
    sacym = abs(np.mean([sacy[idx], sacy1[idx], sacy2[idx], sacy3[idx], sacy4[idx]]) - 5)
    sacmeanX.append(sacxm)
    sacmeanY.append(sacym)
    sactopX.append(sacxm + np.std([abs(sacx[idx] -5), abs(sacx1[idx] - 5), abs(sacx2[idx] - 5), abs(sacx3[idx] - 5), abs(sacx4[idx]- 5)]))
    sacbottomX.append(sacxm - np.std([abs(sacx[idx] -5), abs(sacx1[idx] - 5), abs(sacx2[idx] - 5), abs(sacx3[idx] - 5), abs(sacx4[idx]- 5)]))
    sacbottomY.append(sacym - np.std([abs(sacy[idx] -5), abs(sacy1[idx] - 5), abs(sacy2[idx] - 5), abs(sacy3[idx] - 5), abs(sacy4[idx]- 5)]))
    sactopY.append(sacym + np.std([abs(sacy[idx] -5), abs(sacy1[idx] - 5), abs(sacy2[idx] - 5), abs(sacy3[idx] - 5), abs(sacy4[idx]- 5)]))
    for pos,obj in enumerate(sacbottomX):
        if obj < 0:
            sacbottomX[pos] = 0
    for pos,obj in enumerate(sacbottomY):
        if obj < 0:
            sacbottomY[pos] = 0
x = np.linspace(0, len(td3meanX) - 1, len(td3meanX))
fig, ax  = plt.subplots(1)
plt.title("Comparison of Reinforcement Learning Methods for X Direction On Single Motor Fault")
plt.ylabel("X Absolute Error (m)")
plt.xlabel("Time (0.001s)")
ax.plot(x, td3meanX, color='red', label="PID Controller + TD3")
ax.fill_between(x, td3topX, td3bottomX, facecolor='red', alpha=0.1)
ax.plot(x, ddpgmeanX, color='blue', label="PID Controller + DDPG")
ax.fill_between(x, ddpgtopX, ddpgbottomX, facecolor='blue', alpha=0.1)
ax.plot(x, ppomeanX, color='green', label="PID Controller + PPO")
ax.fill_between(x, ppotopX, ppobottomX, facecolor='green', alpha=0.3)
ax.plot(x, sacmeanX, color='purple', label="PID Controller + SAC")
ax.fill_between(x, sactopX, sacbottomX, facecolor='purple', alpha=0.3)
plt.legend()
plt.show()

fig2, ax2 = plt.subplots(1)
plt.title("Comparison of Reinforcement Learning Methods for Y Direction On Single Motor Fault")
plt.ylabel("Y Absolute Error (m)")
plt.xlabel("Time (0.001s)")
ax2.plot(x, td3meanY, color='red', label="PID Controller + TD3")
ax2.fill_between(x, td3topY, td3bottomY, facecolor='red', alpha=0.1)
ax2.plot(x, ddpgmeanY, color='blue', label="PID Controller + DDPG")
ax2.fill_between(x, ddpgtopY, ddpgbottomY, facecolor='blue', alpha=0.1)
ax2.plot(x, ppomeanY, color='green', label="PID Controller + PPO")
ax2.fill_between(x, ppotopY, ppobottomY, facecolor='green', alpha=0.3)
ax2.plot(x, sacmeanY, color='purple', label="PID Controller + SAC")
ax2.fill_between(x, sactopY, sacbottomY, facecolor='purple', alpha=0.3)
plt.legend()
plt.show()
