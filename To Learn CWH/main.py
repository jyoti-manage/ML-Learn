import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


diabetes =  datasets.load_diabetes()  #built-in dataset
# print(diabetes.keys())  #{data,target,...}
# print(diabetes.target)  #lebels for every column

# print(diabetes.data)  # features list (data)

# diabetes_X = diabetes.data[:,np.newaxis,2]   # array of arrays at index no. /column no. 2 features
diabetes_X = diabetes.data
# print(diabetes_X)

diabetes_X_train = diabetes_X[:-30]  #last 30 rows
diabetes_X_test = diabetes_X[-20:] #starting 20

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-20:]


#make linear model by sklearn
model= linear_model.LinearRegression()

# make line using data  and save the line in the model means fitting data
model.fit(diabetes_X_train, diabetes_y_train)

# get value from the features from model
diabetes_y_predicted = model.predict(diabetes_X_test)

mse = mean_squared_error(diabetes_y_test,diabetes_y_predicted)  #(actual, predict)
print(mse)

print("weight: ", model.coef_)  #coefficents
print("Intercept: ",model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test,diabetes_y_predicted)

# plt.show()



