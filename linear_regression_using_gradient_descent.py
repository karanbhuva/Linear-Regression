# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:11:00 2020

@author: admin
"""

# -*- coding: utf-8 -*-
import matplotlib as plt

data=[[5.5,58],[5.42,56],[5.92,60],[5.42,45],[6.0,97],[5.75,60],[5.0,47],[5.33,70],[5.67,62],[5.08,42],[5.92,74],[5.42,61],[5.92,60],[5.67,72],[5.42,58],[5.92,103],[5.58,50],[5.5,62],[5.17,45],[5.67,58],[5.67,70],[5.5,60]]

Exi=0
Eyi=0
Exi2=0
Exiyi=0
N=len(data)
for i in range(N):
    Exi+=data[i][0]
    Eyi+=data[i][1]
    Exi2+=data[i][0]**2
    Exiyi+=data[i][0]*data[i][1]

learning_rate=0.003031

#derivative of loss function with respect to slope(a) and intercept(b)
f=lambda a,b:((a*Exi2)+(b*Exi)-Exiyi)
g=lambda a,b:((a*Exi)+(b*N)-Eyi)

#step-size for gradient_descent
learning_rate=0.003031

#initial value
an=0
bn=0

                                                                    
while True:
    an_1=an-(learning_rate*(f(an,bn)))
    bn_1=bn-(learning_rate*(g(an_1,bn)))
   
    
    if abs(an-an_1)<((0.4)*(10**-3.85)) and abs(bn-bn_1)<((0.5)*(10**-3.85)):
        break
    an=an_1
    bn=bn_1
    

#Equation of line
equation=lambda x:(an*x)+bn

#Data Visualization
x=[]
y=[]
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])
plt.pyplot.scatter(x,y)
plt.pyplot.plot([5,6],[equation(5),equation(6)])
plt.pyplot.show()

print("slope:",an,"        Intercept:",bn)