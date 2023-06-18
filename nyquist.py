import numpy as np
import matplotlib.pyplot as plt


def make_graph(file_path,frequency):

    sum_v=0
    sum_i=0
    
    path="./Impedance.txt"
    # グラフの作成

    sum_v=0
    sum_i=0
    term = 100
    # グラフの作成
    yI0 = np.genfromtxt(file_path,usecols=(2))
    yV0 = np.genfromtxt(file_path,usecols=(3))
    x = np.genfromtxt(file_path,usecols=(0))
    kernel = np.ones(term)/float(term)
    yI=np.convolve(yI0,kernel)
    yV=np.convolve(yV0,kernel)

    for i in range(1,len(x)):
        sum_v+=yV[i]
        sum_i+=yI[i]
    ave_v=sum_v/(len(x)-1)
    ave_i=sum_i/(len(x)-1)
    liv=yV[1:]-ave_v
    liI=yI[1:]-ave_i
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

    # li = [r - l for l, r in zip(li1, li1[1:])]

    # delta_time=np.min(np.abs(li))      #初期化
    # theta=np.abs(2*np.pi*frequency*delta_time)
    # x_n=Z*np.cos(theta)
    # y_n=-1*Z*np.sin(theta)

    # if x_n>0:
    #     with open(path, "a") as f:
    #         f.write(str(x_n)+"  ")
    #         f.write(str(y_n)+"\n")

    fig, ax = plt.subplots()
    size=len(x)
    plt.plot(x[1:],yI[1:size],"r")
    plt.plot(x[1:],yI0[1:],"b")
    # print(delta_time)
    plt.show()
    # print(x_n, y_n,"f=",frequency,Z,theta)

make_graph("C:/Users/ymnk2/Documents/GitHub/PSIM_AUTOMA/libdiag007capacitor_20.txt",1000)