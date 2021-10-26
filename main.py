import numpy as np
import pandas as pd

# file=open("uav1.txt")
# lines=file.readlines()
# row=len(lines)
# print(row)

df= pd.read_csv("uav1.txt",sep="  ",header=None,names=["t","a","b","c"],dtype=np.float64)
print(df.shape)

