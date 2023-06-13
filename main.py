import subprocess
import nyquist
import numpy as np
import matplotlib.pyplot as plt

path = "./Impedance.txt"
freq = 0.1

# for i in range(11):
#     subprocess.call(f'C:\Powersim\PSIM12.0.1_Softkey_X64\PsimCmd.exe -i "C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor.psimsch" -o "./libdiag007capacitor_{i}.txt" -v "fsig={freq}"'.split(' '))
#     nyquist.make_graph(f"./libdiag007capacitor_{i}.txt", freq)
#     freq = freq*10**(1/5)


Re = np.loadtxt(path,delimiter=' ', usecols=0)
Im = np.loadtxt(path,delimiter=' ', usecols=1)

plt.scatter(Re,Im)
plt.show()