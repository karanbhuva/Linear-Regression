# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:58:49 2020

@author: admin
"""

import copy
from matplotlib import pyplot as plt

import pandas as pd
import random

data1=pd.read_csv("s1-groundtruth-plot.csv")

x=data1.iloc[:,1]
y=data1.iloc[:,2]
data=[[x[i],y[i]] for i in range(len(x))]

def kmeans(k,data):
    cluster=[[] for i in range(k)]
    if(len(data)<k):
        return False
    cluster_new= [[] for i in range(k)]
    centroid=[random.choice(data) for i in range(k)]
    
    while True:
        cluster=[[] for i in range(k)]
        for i in range(len(data)):
            result=[]
            for j in range(len(centroid)):
                result.append(((data[i][0]-centroid[j][0])**2+(data[i][1]-centroid[j][1])**2)**0.5)
            minimum=result.index(min(result))
            
            cluster[minimum].append(data[i])

            sum1=0
            sum2=0
            for j in range(len(cluster[minimum])):
                sum1=sum1+cluster[minimum][j][0]
                sum2=sum2+cluster[minimum][j][1]
            
            result=sum1/len(cluster[minimum])
            result1=sum2/len(cluster[minimum])
            centroid[minimum]=[result,result1]
      
#       # STOPPING CRITERIA
        if cluster==cluster_new:
            break 
        cluster_new=copy.copy(cluster)
        
    #TO COUNT SSE
    sse=0
    for i in range(len(centroid)):
        for j in range(len(cluster[i])):
              sse=sse+(((cluster[i][j][0]-centroid[i][0])**2)+((cluster[i][j][1]-centroid[i][1])**2))
             
  
  
    # FOR PLOTTING
    for i in range(len(cluster)):
        x=[]
        y=[]
        for j in range(len(cluster[i])):
            for k in range(len(cluster[i][j])):
                if k==0:
                    x.append(cluster[i][j][k])
                else:
                    y.append(cluster[i][j][k])  
        #plot datapoints
        plt.scatter(x,y)
        #plot centroid of the datapoints
        plt.scatter(centroid[i][0],centroid[i][1],c="r",s=100)      
    plt.show()
    return [cluster,sse]

def final():    
    sse=0
    final_cluster=0
    
    #first argument is for number of clusters and second is for data
    final_cluster,sse=kmeans(15,data)

     
final()
