# numpyのfftパッケージを用いて簡単にFFT解析
# Warning: time-step is satisfied with fixed-step because we need to caluculate DFT
# ピーク検出
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import pandas as pd

path = "./Impedance2.txt"

with open(path, "w") as f:
    pass


def ChangeImpedance(V_re, V_im, I_re, I_im):
    V=np.sqrt(V_re**2+V_im**2)
    I=np.sqrt(I_re**2+I_im**2)
    print("(V,I)=",V,I)
    Z = V/I
    theta = math.atan(V_im/V_re)-math.atan(I_im/I_re)
    if math.pi/2 < np.abs(theta):
        theta = np.abs(theta) - math.pi

    print("(Z,theta)=",Z,theta)
    Z_re=(Z*math.cos(theta))
    Z_im=(Z*math.sin(theta))

    with open(path, "a") as f:
        f.write(str(Z_re)+"   ")
        f.write(str(Z_im)+"\n")

    
    print(Z_re,Z_im)


def make_graph(path):
    # データの読み込み
    cols0 = pd.read_table(path,usecols=[0],delimiter="  ",engine="python")
    cols1 = pd.read_table(path,usecols=[1],delimiter="  ",engine="python")
    cols2 = pd.read_table(path,usecols=[2],delimiter="  ",engine="python")

    time = cols0.iloc[:,0]
    value_I = cols1.iloc[:,0]
    value_V =cols2.iloc[:,0]

    N = len(time)                    # サンプル数
    dt = time[2]-time[1]     # サンプリング周期 [s] ※固定ステップか確認!!


    FI = np.fft.fft( value_I, N) # Iac1
    FV = np.fft.fft( value_V, N) # Vac1
    freq = np.fft.fftfreq(N, d=dt) # 周波数

    Amp_I = 2*np.abs(FI/(N/2)) # Iac1
    Amp_V = 2*np.abs(FV/(N/2)) # Vac1
    Amp_I_max=np.max(Amp_I[1:N//2])
    Amp_V_max=np.max(Amp_V[1:N//2]) 

    # signal.find_peaks(振度, 高さ)の戻り値は極大値のindexと高さ
    # 第二引数以上の高さのindexを取り出す
    # Amp_c_maxで高さの目安を計算し、heightに代入
    index_I, h_I = signal.find_peaks(Amp_I, height = Amp_I_max - Amp_I_max/100)
    index_V, h_V = signal.find_peaks(Amp_V, height = Amp_V_max - Amp_V_max/100)

 
    plt.show()

    V_cartesian = FV[index_V[0]] / (N/2)
    I_cartesian = FI[index_I[0]] / (N/2)

    V_re = V_cartesian.real
    V_im = V_cartesian.imag
    I_re = I_cartesian.real
    I_im = I_cartesian.imag    

    ChangeImpedance(V_re, V_im, I_re, I_im)





