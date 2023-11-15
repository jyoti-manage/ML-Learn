import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

data_X = np.array([[1],[2],[3]])  #sklearn only work with numpy array

train_x = data_X
test_x = data_X

y_train = np.array([3,2,4])
y_test = np.array([3,2,4])

model= linear_model.LinearRegression()
model.fit(train_x,y_train)

predict = model.predict(test_x)

mse = mean_squared_error(y_test,predict)  #(actual, predict)
print(mse)

print("weight: ", model.coef_)  #coefficents
print("Intercept: ",model.intercept_)

plt.scatter(test_x, y_test)
plt.plot(test_x,predict)

plt.show()

