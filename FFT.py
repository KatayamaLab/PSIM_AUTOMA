# numpyのfftパッケージを用いて簡単にFFT解析
# Warning: time-step is satisfied with fixed-step because we need to caluculate DFT
# ピーク検出
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

fsig = 0.1

path = "C:/Users/ymnk2/Documents/GitHub/PSIM_AUTOMA/Impedance.txt"

with open(path, "w") as f:
    pass

# 波形(x, y)からn個のピークを幅wで検出する関数(xは0から始まる仕様）
def findpeaks(x, y, n, w):
    index_all = list(signal.argrelmax(y, order=w))                  # scipyのピーク検出
    index = []                                                      # ピーク指標の空リスト
    peaks = []                                                      # ピーク値の空リスト
 
    # n個分のピーク情報(指標、値）を格納
    for i in range(n):
        index.append(index_all[0][i])
        peaks.append(y[index_all[0][i]])
    index = np.array(index) * x[1]                                  # xの分解能x[1]をかけて指標を物理軸に変換
    return index, peaks

def ChangeImpedance(V_re, V_im, I_re, I_im):
    V=math.sqrt(V_re**2+V_im**2)
    I=math.sqrt(I_re**2+I_im**2)
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
        f.write(str(Z_im)+"   ")
        f.write(str(fsig)+"\n")    

    
    print(Z_re,Z_im)


def make_graph(path,fs):
    time  = np.loadtxt(path,usecols=0,skiprows=1)   # time
    value_I = np.loadtxt(path,usecols=2,skiprows=1)   # Iac1
    value_V = np.loadtxt(path,usecols=3,skiprows=1)   # Vac1

    N = len(time)                    # サンプル数
    dt = time[2]-time[1]     # サンプリング周期 [s] ※固定ステップか確認!!
    fn = 1/dt/2              # ナイキスト周波数

    # グラフのサイズ
    # fig = plt.figure(figsize=(10, 5))

    FI = np.fft.fft( value_I, N) # Iac1
    FV = np.fft.fft( value_V, N) # Vac1
    freq = np.fft.fftfreq(N, d=dt) # 周波数
    sum_V = 0
    for i in range(1,len(value_V)):
        sum_V += value_V[i]
    
    Center_Amp = sum_V/(len(value_V)-1) # Vac1振動中心を求める

    fc = fs + 100           # カット周波数
    epsilon = 0
    saze_0 = len(FI)
    
    # フィルター
    if(0.1<=fs<20):
        FI[(freq > fs+10)|(freq < 0.001)] = 0 # BPF
        FV[(freq > fs+10)|(freq < 0.001)] = 0 # BPF        
        epsilon = 2
    if(20<fs<40):
        FI[(freq < fs-10)] = 0 # HPF fs-10未満の周波数は通さない
        FV[(freq < fs-10)] = 0 # HPF        
        epsilon = 2
    elif(40<=fs<=1000):   
        FI[(freq > 1200)|(freq < 10)] = 0 # BPF
        FV[(freq > 1200)|(freq < 10)] = 0 # BPF        
        epsilon = 2


    Amp_I = 2*np.abs(FI/(N/2)) # Iac1
    Amp_V = 2*np.abs(FV/(N/2)) # Vac1
    # 直流成分除去
    F3 = np.copy(FV)
    #F3[(Amp_V > 3)]=0          # Amplitudeが3以上の成分をカット
    Amp3 = 2*np.abs(F3/(N/2))
    F3_ifft=np.fft.ifft(F3)
    
    F3_ifft_real=F3_ifft.real*epsilon  
    F1_ifft=np.fft.ifft(FI)
    F1_ifft_real=F1_ifft.real*epsilon  

    # fig,ax = plt.subplot()
    
    V = value_V-Center_Amp
    # plt.plot(time, V,label="V")   
    # plt.plot(time, F3_ifft_real,label="V(IFFT)")
    # plt.plot(time, value_I,label="I")
    # plt.plot(time, F1_ifft_real,label="I(IFFT)")
    # plt.legend(loc="upper right")

    # Zとthetaを求める
    index_I, Amp_I_peak= findpeaks(freq, Amp_I, 1, 5)
    index_V, Amp_V_peak= findpeaks(freq, Amp_V, 1, 5)  

    # fig = plt.figure(figsize=(8, 5))
    # ax = fig.add_subplot(211)
    # ax.set_xscale("log")    
    # ax.plot(freq[:N//2], Amp_I[:N//2])
    # ax.scatter(index_I[:N//2], Amp_I_peak[:N//2])

    # fig1 = plt.figure(figsize=(8, 5))
    # ax1 = fig1.add_subplot(211)
    # ax1.plot(freq[:N//2], Amp_V[:N//2])
    # ax1.scatter(index_I[:N//2], Amp_V_peak[:N//2])
    # ax1.set_xscale("log")
 
    # plt.show()

    V_cartesian=FV[2*np.abs(FV/(N/2))==Amp_V_peak]/(N/2)
    I_cartesian=FI[2*np.abs(FI/(N/2))==Amp_I_peak]/(N/2)

    V_re=V_cartesian.real
    V_im=V_cartesian.imag
    I_re=I_cartesian.real
    I_im=I_cartesian.imag    
    ChangeImpedance(V_re, V_im, I_re, I_im)
    
  

    # print("(Re,Im)=",Re,Im,"\n(Z,theta)=",Z,theta)

    # plt.show()




