from math import asin, gamma
import numpy as np
import pandas as pd
import sympy as mp



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
Q=[[Q1],[Q2],[Q3]]

xk=status0


for i in range(np.shape(uav1)[0]):
    pass


# f=x**2*y+y**2
# f_x=mp.diff(f,x)
# print(f_x.evalf(subs={x:3,y:4}))
# print(f_x)