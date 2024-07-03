import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics

dataset = pd.read_csv('Medicalinsurance\insurance.csv')
# print(dataset.shape)
# print(dataset.info())
#for categorical data we can assign values based on their importance
# sns.set()
# plt.figure(figsize=(6,6))
# sns.displot(dataset['age'])
# plt.show()

# sns.set()
# plt.figure(figsize=(6,6))
# # sns.displot(dataset['bmi']) normal range of bmi is 18.5 to 24.9
# # sns.countplot(x='sex',data=dataset)
# plt.show()

# sns.set()
# plt.figure(figsize=(6,6))
# sns.countplot(x='region',data=dataset)
# plt.show()

#encoding sex column
dataset.replace({'sex':{'male':0,'female':1}},inplace=True)

#encoding smoker column
dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)

#encoding region column
dataset.replace({'region':{'southwest':1,'southeast':0,'northwest':3,'northeast':2}},inplace=True)

x = dataset.drop(columns='charges',axis=1)
y = dataset['charges']

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=2)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)


training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train,training_data_prediction)
test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test,test_data_prediction)
input_data = (31,1,25.74,0,1,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
prediction = regressor.predict(input_data_reshape)
print("R2 for training data :" , r2_train)
print("R2 for test data :" , r2_test)
print("Cost is :","$",prediction[0])