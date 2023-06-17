import subprocess
import nyquist
import numpy as np
import matplotlib.pyplot as plt

path = "./Impedance.txt"
freq = 0.1
total_time = 1
time_step = 10**(-10)

with open(path, "w") as f:
    pass

#  The format to call the command-line version of PSIM is as follows:

#     PsimCmd.exe  -i "[input file]"  -o "[output file]"  -v "Var_name=Var_value" -g   

#  The command line parameters are:
#       -i:  input schematic file. 
#       -o:  output result file in either text format (.txt) or binary format (.smv).
#       -v:  variable definition. "Var_name" is the variable name as defined in PSIM, and
#            "Var_value" is the numerical value of the variable. 
#            This parameter can be used multiple times. 
#            For example, to set the values of two parameters R1 and R2 in PSIM, use:
#               -v "R1=10" -v "R2=5" 
#       -t:  total time of the simulation
#       -s:  time step of the simulation
#       -g:  Run Simview after the simulation is complete.

for i in range(29):  
     subprocess.call(f'C:\Powersim\PSIM12.0.1_Softkey_X64\PsimCmd.exe -i "C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor.psimsch" -o "./libdiag007capacitor_{i}.txt" -v "fsig={freq}" -t "{total_time}" -s "{time_step}"'.split(' '))
     nyquist.make_graph(f"./libdiag007capacitor_{i}.txt", freq)
     freq = freq*10**(1/7)
     total_time=0.8 + 3/freq
     print(100*i/20,"%","has done!!")
     if(freq>=10):
          time_step=10**(-6)


Re = np.loadtxt(path, usecols=0)
Im = np.loadtxt(path, usecols=1)

fig, ax = plt.subplots()
ax.invert_yaxis()
plt.scatter(Re,Im)
plt.show()