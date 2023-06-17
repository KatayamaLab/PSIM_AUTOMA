import numpy as np


def make_graph(file_path,frequency):

    sum_v=0
    sum_i=0
    
    path="./Impedance.txt"
    # グラフの作成
    yI = np.genfromtxt(file_path,usecols=(2))
    yV = np.genfromtxt(file_path,usecols=(3))
    x = np.genfromtxt(file_path,usecols=(0))


    for i in range(1,len(x)):
        sum_v+=yV[i]
        sum_i+=yI[i]
    ave_v=sum_v/(len(x))
    ave_i=sum_i/(len(x))
    liv=yV[1:]-ave_v
    liI=yI[1:]-ave_i


    sum_v=0
    sum_i=0
    # グラフの作成
    yI = np.genfromtxt(file_path,usecols=(2))
    yV = np.genfromtxt(file_path,usecols=(3))
    x = np.genfromtxt(file_path,usecols=(0))

    for i in range(1,len(x)):
        sum_v+=yV[i]
        sum_i+=yI[i]
    ave_v=sum_v/(len(x))
    ave_i=sum_i/(len(x))
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

    li = [r - l for l, r in zip(li1, li1[1:])]

    ave_theta=np.min(np.abs(li))      #初期化
    theta=np.abs(2*np.pi*frequency*ave_theta)
    x_n=Z*np.cos(theta)
    y_n=-1*Z*np.sin(theta)

    if x_n>0:
        with open(path, "a") as f:
            f.write(str(x_n)+"  ")
            f.write(str(y_n)+"\n")

    print(x_n, y_n,"f=",frequency,Z,theta)


