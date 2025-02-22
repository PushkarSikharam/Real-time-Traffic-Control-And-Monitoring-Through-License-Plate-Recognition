# MULTIPLE LINEAR REGRESSION
# y = m1x1 + m2x2 + m3x3 + .... + c

# Importing the libraries
import numpy as np
import pandas as pd
from math import floor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
# Use the correct path relative to the script
dataset = pd.read_csv("./cars.csv")
temp = pd.read_csv("./cars.csv", usecols=[0, 1, 4])

# Extracting feature matrix X and lane-specific target variables
X = temp.iloc[:, :].values
lane1 = dataset.iloc[:, 5].values
lane2 = dataset.iloc[:, 6].values
lane3 = dataset.iloc[:, 7].values
lane4 = dataset.iloc[:, 8].values

# Encoding categorical data
labelencoderObj = LabelEncoder()
X[:, 0] = labelencoderObj.fit_transform(X[:, 0])
X[:, 2] = labelencoderObj.fit_transform(X[:, 1])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train1, y_test1 = train_test_split(X, lane1, test_size=0.2, random_state=0)
_, _, y_train2, y_test2 = train_test_split(X, lane2, test_size=0.2, random_state=0)
_, _, y_train3, y_test3 = train_test_split(X, lane3, test_size=0.2, random_state=0)
_, _, y_train4, y_test4 = train_test_split(X, lane4, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training Set
regressor = LinearRegression()

# Predicting the test set results
regressor.fit(X_train, y_train1)
y_pred1 = regressor.predict(X_test)

regressor.fit(X_train, y_train2)
y_pred2 = regressor.predict(X_test)

regressor.fit(X_train, y_train3)
y_pred3 = regressor.predict(X_test)

regressor.fit(X_train, y_train4)
y_pred4 = regressor.predict(X_test)

# Calculate total cars for each prediction
totalCars = []
for i in range(len(y_pred1)):
    totalCars.append(floor(y_pred1[i] + y_pred2[i] + y_pred3[i] + y_pred4[i]))

# Output the total cars
print("Total cars for each prediction:")
print(totalCars)
