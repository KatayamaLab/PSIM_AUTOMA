import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def make_graph(path_to_data: str,frequency: float,  time_Step: float):

    time  = np.loadtxt(path_to_data,usecols=0,skiprows=1)   # time
    Iac1 = np.loadtxt(path_to_data,usecols=2,skiprows=1)   # Iac1
    Vac1 = np.loadtxt(path_to_data,usecols=3,skiprows=1)   # Vac1

    N = len(time)                    # サンプル数
    dt = time[2]-time[1]     # サンプリング周期 [s] ※固定ステップか確認!!
    fn = 1/dt/2              # ナイキスト周波数

    F_I = np.fft.fft(Iac1)
    F_V = np.fft.fft(Vac1)
    freq = np.fft.fftfreq(N, d=dt) # 周波数

    F_I = F_I / (len(F_I) / 2)
    F_V = F_V / (len(F_V) / 2)

    impedance = F_V / F_I

    # I = F_I[np.abs(F_I[1:int(len(F_I)/2)]).argmax()]
    # V = F_V[np.abs(F_V[1:int(len(F_I)/2)]).argmax()]


    return impedance[impedance[1:].argmax()]



