# numpyのfftパッケージを用いて簡単にFFT解析
# Warning: time-step is satisfied with fixed-step because we need to caluculate DFT
# ピーク検出
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import math
import pandas as pd


path = "./Impedance.txt"

with open(path, "w") as f:
    pass

def ChangeImpedance(V_re, V_im, I_re, I_im):
    V=np.sqrt(V_re**2+V_im**2)
    I=np.sqrt(I_re**2+I_im**2)
    
    Z = V/I
    theta = -1*np.abs(np.arctan(V_im/V_re)-np.arctan(I_im/I_re))

    if np.pi/2 < np.abs(theta):
        theta = np.abs(theta) - np.pi

    print("(Z,theta)=",Z,theta)
    Z_re=(Z*np.cos(theta))
    Z_im=(Z*np.sin(theta))

    with open(path, "a") as f:
        f.write(str(Z_re)+"   ")
        f.write(str(Z_im)+"\n")    
    print(Z_re,Z_im)


def make_graph(path):
    # データの読み込み
    # cols0 = pd.read_table(path, usecols = [0], delimiter = "  ", engine = "python", dtype = "float", skiprows = 1)
    # cols1 = pd.read_table(path, usecols = [2], delimiter = "  ", engine = "python", dtype = "float", skiprows = 1)
    # cols2 = pd.read_table(path, usecols = [3], delimiter = "  ", engine = "python", dtype = "float", skiprows = 1)
    df = pd.read_table(
        path,
        header=0,
        delim_whitespace=True)
    # print(df)

    time = np.array(df['Time'], dtype = float)
    value_I = np.array(df['Iac1'], dtype = float)
    value_V = np.array(df['Vac1'], dtype = float)

    dt = time[2]-time[1]
    N = len(time)                    # サンプル数
    
    FI = np.fft.fft(value_I, N) # Iac1
    FV = np.fft.fft(value_V, N) # Vac1
    freq = np.fft.fftfreq(N, d=dt) # 周波数

    FI[0] = 0
    FV[0] = 0
    Amp_I = 2 * np.abs(FI/(N/2)) # Iac1
    Amp_V = 2 * np.abs(FV/(N/2)) # Vac1
    Amp_I_max = max(Amp_I[1:N//2])
    Amp_V_max = max(Amp_V[1:N//2]) 

    # signal.find_peaks(振度, 高さ)の戻り値は極大値のindexと高さ
    # 第二引数以上の高さのindexを取り出す
    # Amp_c_maxで高さの目安を計算し、heightに代入
    # Amp_I_max = max(Amp_I[1:N//2])
    # Amp_V_max = max(Amp_V[1:N//2]) 
    # index_I = np.where(Amp_I == Amp_I_max)
    # index_V = np.where(Amp_V == Amp_V_max)
    # plt.show()
    # index_I, _I = signal.find_peaks(Amp_I,height = Amp_I_max*0.8)
    # index_V, _I = signal.find_peaks(Amp_I,height = Amp_I_max*0.8)

    V_cartesian = FV[3] / (N/2)
    I_cartesian = FI[3] / (N/2)

    V_re,V_im,I_re,I_im = 0, 0, 0, 0
    V_re = V_cartesian.real
    V_im = V_cartesian.imag
    I_re = I_cartesian.real
    I_im = I_cartesian.imag
    print("(V,I)=",Amp_V[3], Amp_I[3])

    ChangeImpedance(V_re, V_im, I_re, I_im)