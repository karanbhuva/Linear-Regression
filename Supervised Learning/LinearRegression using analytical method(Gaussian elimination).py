# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:12:22 2020

@author: admin
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv("weight-height.csv")
height=list(data["Height"])
weight=list(data["Weight"])

x=np.array(height).reshape((-1,1))
y=np.array(weight)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


Exi=sum(x_train)
Eyi=sum(y_train)
Exi2=0
Exiyi=0
N=len(x_train)

for i in range(N):
    Exi2+=x_train[i]**2
    Exiyi+=x_train[i]*y_train[i]


eq=[[Exi,N,Eyi],[Exi2,Exi,Exiyi]]
for k in range(len(eq)-1):
    pvt=eq[k][k]
    for i in range(1+k,len(eq)):
        pvt2=eq[i][k]
        for j in range(len(eq[i])):
            eq[i][j]=eq[i][j]-((pvt2/pvt)*eq[k][j])
        
n=len(eq)
k=0
ans=[]
for i in range(n-1,-1,-1):
    num=eq[i][n]
    deno=eq[i][i]
    for j in range(n-1,i,-1):
        num=num-(eq[i][j]*ans[k])
        k=k+1;
    k=0
    ans.append(num/deno)

ans.reverse()
print("slope:",ans[0],"        Intercept:",ans[1])
equation=lambda x:(ans[0]*x)+ans[1]

plt.scatter(x_train,y_train)
plt.plot([min(x_train),max(x_train)],[equation(min(x_train)),equation(max(x_train))],color="red")
plt.show()

prediction=[]
for i in range(len(x_test)):
    prediction.append((ans[0]*x_test[i])+ans[1])