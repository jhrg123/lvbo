import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd


# file=open("uav1.txt")
# lines=file.readlines()
# row=len(lines)
# print(row)


uav1=np.mat(pd.read_csv("uav1.txt",sep="  ",header=None,names=["t","a","b","c"],dtype=np.float64))
uav2=np.mat(pd.read_csv("uav2.txt",sep="  ",header=None,names=["t","a","b","c"],dtype=np.float64))
targe_mat=np.mat(pd.read_csv("target.txt",sep="  ",header=None,dtype=np.float64))
measur_mat=np.mat(pd.read_csv("measurement.txt",sep="  ",header=None,dtype=np.float64))


