from math import asin, gamma, pi
import numpy as np
import pandas as pd
import sympy as mp

x=mp.Symbol('x')
y=mp.Symbol('y')
z=mp.Symbol('z')
f=x**2*y+y**2

print(mp.diff(f,x).evalf(subs={x:3,y:4,z:1}))

