from math import asin, gamma, pi
import numpy as np
from numpy.linalg import linalg
import pandas as pd
import sympy as mp
from sympy.core.function import diff



# 读取数据
uav1=np.mat(pd.read_csv("uav1.txt",sep="  ",header=None,names=["t","a","b","c"],dtype=np.float64))
uav2=np.mat(pd.read_csv("uav2.txt",sep="  ",header=None,names=["t","a","b","c"],dtype=np.float64))
targe_mat=np.mat(pd.read_csv("target.txt",sep="  ",header=None,dtype=np.float64))
measur_mat=np.mat(pd.read_csv("measurement.txt",sep="  ",header=None,dtype=np.float64))


T=0.1

# 初始化状态变量
status0=np.array([[0,0,0,47.8109,18.1173,47.8109,0,0,0]]).T
A=np.c_[np.zeros([9,3]),np.r_[np.eye(6),np.zeros([3,6])]]
B=np.r_[np.zeros([6,3]),np.eye(3)]
fai=np.eye(9)+T*A+T**2/2*np.matmul(A,A)
tao=np.matmul(T*np.eye(9)+T**2/2*np.r_[np.c_[np.zeros([6,3]),np.eye(6)],np.zeros([3,9])]+T**3/6*np.r_[np.c_[np.zeros([3,6]),np.eye(3)],np.zeros([6,9])],B)


xrs1=mp.Symbol('xrs1')
xrs2=mp.Symbol('xrs2')
yrs1=mp.Symbol('yrs1')
yrs2=mp.Symbol('yrs2')
zrs1=mp.Symbol('zrs1')
zrs2=mp.Symbol('zrs2')

gama1=mp.asin(yrs1/mp.sqrt(xrs1**2+yrs1**2+zrs1**2))
yita1=mp.atan2(-1*zrs1,xrs1)
gama2=mp.asin(yrs2/mp.sqrt(xrs2**2+yrs2**2+zrs2**2))
yita2=mp.atan2(-1*zrs2,xrs2)

#计算加速度方差
Q1=np.var(targe_mat[:,7])
Q2=np.var(targe_mat[:,7])
Q3=np.var(targe_mat[:,7])
Q=np.diag([Q1,Q2,Q3])

#计算测量方差：
R1=(0.3/pi*180)**2
R=np.diag([R1,R1,R1,R1])
xk=status0
pk=np.diag([100,100,100,10,10,10,1,1,1])
for i in range(np.shape(uav1)[0]):
    #一步预测：
    xkk=fai@xk
    #一步预测协方差
    pkk=fai@pk@fai.T+tao@Q@tao.T
    #量测方程雅可比矩阵求解
    fxrs1=xkk[0]-uav1[i,1]
    fxrs2=xkk[0]-uav2[i,1]
    fyrs1=xkk[1]-uav1[i,2]
    fyrs2=xkk[1]-uav2[i,2]
    fzrs1=xkk[2]-uav1[i,3]
    fzrs2=xkk[2]-uav2[i,3]
    h11=mp.diff(gama1,xrs1).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h12=mp.diff(gama1,yrs1).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h13=mp.diff(gama1,zrs1).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h21=mp.diff(yita1,xrs1).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h22=mp.diff(yita1,yrs1).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h23=mp.diff(yita1,zrs1).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h31=mp.diff(gama2,xrs2).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h32=mp.diff(gama2,yrs2).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h33=mp.diff(gama2,zrs2).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h41=mp.diff(yita2,xrs2).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h42=mp.diff(yita2,yrs2).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    h43=mp.diff(yita2,zrs2).evalf(subs={xrs1:fxrs1[0],xrs2:fxrs2[0],yrs1:fyrs1[0],yrs2:fyrs2[0],zrs1:fzrs1[0],zrs2:fzrs2[0]})
    H0=[[h11,h12,h13],[h21,h22,h23],[h31,h32,h33],[h41,h42,h43]]
    H=np.c_[H0,np.zeros([4,6])]
    a = (R+H@pkk@H.T).astype(np.float64)
    #计算滤波增益矩阵
    kk=pkk@H.T@np.linalg.pinv(a)
    



    





# f=x**2*y+y**2
# f_x=mp.diff(f,x)
# print(f_x.evalf(subs={x:3,y:4}))
# print(f_x)