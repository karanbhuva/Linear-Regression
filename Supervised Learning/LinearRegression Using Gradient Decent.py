
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv("weight-height.csv")
height=list(data["Height"])
weight=list(data["Weight"])



x=np.array(height).reshape((-1,1))
y=np.array(weight)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

Exi=int(sum(x_train))
Eyi=int(sum(y_train))
Exi2=0
Exiyi=0
N=len(x_train)
for i in range(N):
    Exi2+=int(x_train[i]**2)
    Exiyi+=int(x_train[i]*y_train[i])



#derivative of loss function with respect to slope(a) and intercept(b)
f=lambda a,b:((a*Exi2)+(b*Exi)-Exiyi)
g=lambda a,b:((a*Exi)+(b*N)-Eyi)

#step-size for gradient_descent
learning_rate=0.00000009999

#initial value
an=0
bn=0

while True:
    an_1=an-(learning_rate*(f(an,bn)))
    bn_1=bn-(learning_rate*(g(an_1,bn)))
   
    
    if abs(an-an_1)<((0.4)*(10**-4.55)) and abs(bn-bn_1)<((0.5)*(10**-4.55)):
        break
    an=an_1
    bn=bn_1
    

#Equation of line
equation=lambda x:(an*x)+bn

#Data Visualization
x=[]
y=[]
plt.scatter(x_train,y_train)
plt.plot([min(x_train),max(x_train)],[equation(min(x_train)),equation(max(x_train))])
plt.show()

print("slope:",an,"        Intercept:",bn)

prediction=[]
for i in range(len(x_test)):
    prediction.append((an*x_test[i])+bn)
