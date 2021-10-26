import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd




# 读取数据
uav1=np.mat(pd.read_csv("uav1.txt",sep="  ",header=None,names=["t","a","b","c"],dtype=np.float64))
uav2=np.mat(pd.read_csv("uav2.txt",sep="  ",header=None,names=["t","a","b","c"],dtype=np.float64))
targe_mat=np.mat(pd.read_csv("target.txt",sep="  ",header=None,dtype=np.float64))
measur_mat=np.mat(pd.read_csv("measurement.txt",sep="  ",header=None,dtype=np.float64))

T=0.1

# 初始化状态变量
status0=np.array([[0,0,0,47.8109,18.1173,47.8109,0,0,0]]).T
A=np.c_[np.zeros([9,3]),np.r_[np.eye(6),np.zeros([3,6])]]
fai=np.eye(9)+T*A+T**2/2*np.matmul(A,A)
print(fai)


