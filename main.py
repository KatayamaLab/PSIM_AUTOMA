import subprocess

freq = 0.1
for i in range(11):
    subprocess.call(f'C:\Powersim\PSIM12.0.1_Softkey_X64\PsimCmd.exe -i "C:/Users/ymnk2/Downloads/EVbattery/libdiag007capacitor.psimsch" -o "./libdiag007capacitor-{i}.txt" -v "fsig={freq}"'.split(' '))
    freq = freq*10**(1/5)

