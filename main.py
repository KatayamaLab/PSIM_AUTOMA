import subprocess
#import nyquist
import FFT
import numpy as np
import matplotlib.pyplot as plt
import os

###
# import test
###

# 作業ディレクトリを取得し、出力データを格納するディレクトリを作成


A = "outdir"
cd = os.getcwd()

if os.path.isdir(str(A)) == False:
    os.mkdir(A)
else:
    pass
os.chdir(A)
out = os.getcwd()
# 理論値
# idea_path="C:/Users/ymnk2/Desktop/python-practice/mkgraph/idea.txt"
# 入力ファイル(psimsch)
psimfile_in = "C:/Users/ymnk2/Documents/GitHub/PSIM_AUTOMA/libdiag007capacitor.psimsch"
finit = 0.1
freq = finit
total_time = 1
time_step = 10**(-5)

with open("./Impedance.txt", "w") as f:
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


N = 21

# PSIM起動および出力データ収集
for i in range(N):
     total_time = 0.8 + 3 / freq
     subprocess.call(f'C:/Powersim/PSIM12.0.1_Softkey_X64/PsimCmd.exe -i {psimfile_in} -o "{out}/libdiag007capacitor_{i}.txt" -v "fsig={freq}" -t "{total_time}" -s "{time_step}"'.split(' '))
    #  imp.append(test.make_graph(f"C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor_{i}.txt", freq, time_step))
    #  FFT.make_graph(f"C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor_{i}.txt", freq, time_step)
     freq = freq*10**(1/5)
     print(100*i/20,"%","has done!!")
     if(freq >= 10):
          time_step=10**(-6)

freq = finit
print(os.getcwd())
os.chdir(f"{cd}/{A}")
# データの読み込み
for i in range(1,N):
    # FFT.make_graph(f"{out}/libdiag007capacitor_{i}.txt")
    FFT.make_graph(f"c:/Users/ymnk2/Documents/GitHub/PSIM_AUTOMA/outdir/libdiag007capacitor_{i}.txt")
    print(100*i/20,"%" ,"has done!!","fsig=",freq)
    freq = freq*10**(1/5)

Re = np.loadtxt("./Impedance.txt", usecols=0)
Im = np.loadtxt("./Impedance.txt", usecols=1)

# x = np.loadtxt(idea_path, usecols=1)
# y = np.loadtxt(idea_path, usecols=2)

fig, ax = plt.subplots()
ax.invert_yaxis()
plt.plot(Re,Im,label="PSIM")
# plt.plot(x,y,color="r",label="theorical")
ax.legend()
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()