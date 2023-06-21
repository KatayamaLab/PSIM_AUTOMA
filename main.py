import subprocess
#import nyquist
import FFT
import numpy as np
import matplotlib.pyplot as plt


###
import test
###


idea_path="C:/Users/ymnk2/Desktop/python-practice/mkgraph/idea.txt"
path = "C:/Users/ymnk2/Documents/GitHub/PSIM_AUTOMA/Impedance.txt"
psimfile_in = "C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor.psimsch"
out_path = "C:/Users/ymnk2/Downloads/EVbattery"
freq = 0.1
total_time = 1
time_step = 10**(-5)

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

imp = list()
N = 21
# PSIM起動および出力データ収集
for i in range(N):  
     total_time = 0.8 + 1 / freq
     subprocess.call(f'C:/Powersim/PSIM12.0.1_Softkey_X64/PsimCmd.exe -i {psimfile_in} -o "{out_path}/libdiag007capacitor_{i}.txt" -v "fsig={freq}" -t "{total_time}" -s "{time_step}"'.split(' '))
    #  imp.append(test.make_graph(f"C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor_{i}.txt", freq, time_step))
    #  FFT.make_graph(f"C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor_{i}.txt", freq, time_step)
     freq = freq*10**(1/5)
     print(100*i/20,"%","has done!!")
     if(freq>=10):
          time_step=10**(-6)

freq = 0.1

# データの読み込み
for i in range(1,N):
    FFT.make_graph(f"{out_path}/libdiag007capacitor_{i}.txt",freq)
    print(100*i/20,"%" ,"has done!!","fsig=",freq)
    freq = freq*10**(1/5)



Re = np.loadtxt(path, usecols=0)
Im = np.loadtxt(path, usecols=1)

x = np.loadtxt(idea_path, usecols=1)
y = np.loadtxt(idea_path, usecols=2)
fig, ax = plt.subplots()
ax.invert_yaxis()
plt.scatter(Re,Im,label="PSIM")
plt.plot(x,y,color="r",label="theorical")
ax.legend()
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()