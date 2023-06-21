import numpy as np
import matplotlib.pyplot as plt

# path = "./Impedance.txt"
# with open(path, "w") as f:
#     pass

def make_graph(file_path,frequency):

    sum_v=0
    sum_i=0
    
    path="./Impedance.txt"
    # グラフの作成

    sum_v=0
    sum_i=0
    term = 100
    # グラフの作成
    yI = np.loadtxt(file_path,usecols=2,skiprows=1)
    yV = np.loadtxt(file_path,usecols=3,skiprows=1)
    x = np.loadtxt(file_path,usecols=0,skiprows=1)



    for i in range(1,len(x)):
        sum_v+=yV[i]
        sum_i+=yI[i]
    ave_v=sum_v/(len(x)-1)
    ave_i=sum_i/(len(x)-1)
    liv=yV[1:]-ave_v
    liI=yI[1:]-ave_i

    # Zとthetaを求める
    V=np.max(yV[1:]-ave_v)
    I=np.max(yI[1:]-ave_i)
    Z=V/I
    li1=[]
    for i in range(1,len(x)-2):
        if liv[i]<0:
            if liv[i+1]>=0:
                li1.append((x[i]+x[i+1])/2)
        if liv[i]>0:
            if liv[i+1]<=0:
                li1.append((x[i]+x[i+1])/2)     
        if liI[i]<0:
            if liI[i+1]>=0:
                li1.append((x[i]+x[i+1])/2)
        if liI[i]>0:
            if liI[i+1]<=0:
                li1.append((x[i]+x[i+1])/2)    
    li = [r - l for l, r in zip(li1[:len(li1)], li1[1:])]
    print(liv[:10])
    delta_time=np.amin(li)    
    theta=np.abs(2*np.pi*frequency*delta_time)
    x_n=Z*np.cos(theta)
    y_n=-1*Z*np.sin(theta)

    if x_n>0:
        with open(path, "a") as f:
            f.write(str(x_n)+"  ")
            f.write(str(y_n)+"\n")

    fig, ax = plt.subplots()
    size=len(x)
    plt.plot(x[1:],yI[1:],"b")
    plt.show()
    print(x_n, y_n,"f=",frequency,Z,theta)

# fsig=0.1

# for i in range(1,21):
#     make_graph(f"C:/Users/ymnk2/Documents/GitHub/PSIM_AUTOMA/libdiag007capacitor_{i}.txt",fsig)
#     print(100*i/20,"%" ,"has done!!","fsig=",fsig)
#     fsig = fsig*10**(1/5)

# Re = np.loadtxt(path, usecols=0)
# Im = np.loadtxt(path, usecols=1)

# fig, ax = plt.subplots()
# ax.invert_yaxis()
# plt.scatter(Re,Im)
# plt.show()