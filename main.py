import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
bc_data = pd.read_csv(r'D:\bc_data.csv',header=0)
# print(bc_data.head(5))
print(bc_data.describe())
data =bc_data.drop(['id'],axis=1)
print(data.head())
x_data = data.drop(['diagnosis'],axis=1)
y_data = np.ravel(data[['diagnosis']])
#测试数据与训练数据的拆分方法
from sklearn.model_selection import train_test_split
x_trainingSet , x_testSet , y_trainingSet , y_testSet = train_test_split(x_data,y_data,random_state=1)
#查看训练集形状
# print(x_testSet.shape)
from sklearn.neighbors import KNeighborsClassifier
#实例化KNN模型 ，并设计超级参数algorithm='kd_tree'
cancerModel = KNeighborsClassifier(algorithm='kd_tree')
#基于训练集新出的具体模型
myModel.fit(x_trainingSet,y_trainingSet)
y_predictSet = myModel.predict(x_testSet)
