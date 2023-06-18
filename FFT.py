# numpyのfftパッケージを用いて簡単にFFT解析
# Warning: time-step is satisfied with fixed-step because we need to caluculate DFT
import numpy as np
import matplotlib.pyplot as plt

fsig = 0.1

impedance_path="C:/Users/ymnk2/Desktop/python-practice/mkgraph/new.txt"
with open(impedance_path, "w") as f:
    pass


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

    if(0.1<=fs<20):
        FI[(freq > fs+10)] = 0 # LPF
        FV[(freq > fs+10)] = 0 # LPF        
        epsilon = 1
    if(20<fs<40):
        FI[(freq < fs-10)] = 0 # HPF
        FV[(freq < fs-10)] = 0 # HPF        
        epsilon = 2
    elif(40<=fs<=1000):   
        FI[(freq > 1200)|(freq < 10)] = 0 # LPF
        FV[(freq > 1200)|(freq < 10)] = 0 # LPF        
        epsilon = 2


    Amp_I = 2*np.abs(FI/(N/2)) # Iac1
    Amp_V = 2*np.abs(FV/(N/2)) # Vac1
    # 直流成分除去
    F3 = np.copy(FV)
    F3[(Amp_V > 3)]=0          # Amplitudeが3以上の成分をカット
    Amp3 = 2*np.abs(F3/(N/2))
    F3_ifft=np.fft.ifft(F3)
    
    F3_ifft_real=F3_ifft.real*epsilon  
    F1_ifft=np.fft.ifft(FI)
    F1_ifft_real=F1_ifft.real*epsilon  

    print(Center_Amp)
    # fig,ax = plt.subplot()
    
    V = value_V-Center_Amp
    # 電流電圧波形の時間変化
    # plt.plot(time, V,label="V")   
    # plt.plot(time, F3_ifft_real,label="V(IFFT)")
    # plt.plot(time, value_I,label="I")
    # plt.plot(time, F1_ifft_real,label="I(IFFT)")
    # plt.legend(loc="upper right")

    # Zとthetaを求める
    V = np.abs(np.max(F3_ifft_real))
    I = np.abs(np.max(F1_ifft_real))
    Z = V/I
    li1=[]
    for i in range(1,len(time)-2):
        if F3_ifft_real[i]<0:
            if F3_ifft_real[i+1]>=0:
                li1.append((time[i]+time[i+1])/2)
        if F3_ifft_real[i]>0:
            if F3_ifft_real[i+1]<=0:
                li1.append((time[i]+time[i+1])/2)     
        if F1_ifft_real[i]<0:
            if F1_ifft_real[i+1]>=0:
                li1.append((time[i]+time[i+1])/2)
        if F1_ifft_real[i]>0:
            if F1_ifft_real[i+1]<=0:
                li1.append((time[i]+time[i+1])/2)
    
    li = [r - l for l, r in zip(li1, li1[1:])]

    delta_time = np.min(np.abs(li))      #初期化
    theta = np.abs(2*np.pi*fsig*delta_time)
    Re = Z*np.cos(theta)
    Im = -1*Z*np.sin(theta)
    
    with open(impedance_path, "a") as f:
        f.write(str(Re)+"   ")
        f.write(str(Im)+"   ")
        f.write(str(fsig)+"\n")      
    
    print("epsilon=",epsilon)
    print("(Re,Im)=",Re,Im,"\n(Z,theta)=",Z,theta)
    # ax = fig.add_subplot(231)
    # posは行数・列数・位置を表す3桁の整数。
    # 例えば234なら、2行3列のうち4番目の図。
    # 各数は当然10未満でなければならない


    plt.show()

# =====データの読み込み、以下をコメントアウトすることでこのファイルだけでデバック可能============
# for i in range(1,21):
#     FFT(f"C:/Users/ymnk2/Documents/GitHub/PSIM_AUTOMA/libdiag007capacitor_{i}.txt",fsig)
#     print(100*i/20,"%" ,"has done!!","fsig=",fsig)
#     fsig = fsig*10**(1/5)

# Re = np.loadtxt(impedance_path, usecols=0)
# Im = np.loadtxt(impedance_path, usecols=1)
# fig, ax = plt.subplots()
# ax.invert_yaxis()
# plt.scatter(Re,Im)
# plt.show()
# ==========================================================================================