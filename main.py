import subprocess
import nyquist
import numpy as np
import matplotlib.pyplot as plt

path="./Impedance.txt"
freq = 0.1

for i in range(11):
    subprocess.call(f'C:\Powersim\PSIM12.0.1_Softkey_X64\PsimCmd.exe -i "C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor.psimsch" -o "./libdiag007capacitor_{i}.txt" -v "fsig={freq}"'.split(' '))
    nyquist.make_graph(f"./libdiag007capacitor_{i}.txt", freq)
    freq = freq*10**(1/5)

nyquist.f.close()
Im = np.genfromtxt(path,usecols=(1))
Re = np.genfromtxt(path,usecols=(0))
fig,ax2 = plt.subplots()
#ax2.plot(Re,Im)
ax2.invert_yaxis()
plt.scatter(Re,Im)