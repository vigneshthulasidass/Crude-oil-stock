#!/usr/bin/env python
# coding: utf-8


# A machine learning library used for linear regression
from sklearn.linear_model import LinearRegression
# numpy and pandas will be used for data manipulation
import numpy as np
import pandas as pd
# matplotlib will be used for visually representing our data
import matplotlib.pyplot as plt
# Quandl will be used for importing historical oil prices
import quandl


# Setting our API key
quandl.ApiConfig.api_key = "your api key goes here"

# Importing our data
data = quandl.get("FRED/DCOILWTICO", start_date="2000-01-01", end_date="2020-01-01")

data.head()

# Setting the text on the Y-axis
plt.ylabel("Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma")

# Setting the size of our graph
data.Value.plot(figsize=(15,8))

#moving average of 3 and 9 days

data['MA3'] = data['Value'].shift(1).rolling(window=3).mean()
data['MA9']= data['Value'].shift(1).rolling(window=9).mean()

# Dropping the NaN values
data = data.dropna()

# Initialising X and assigning the two feature variables
X = data[['MA3','MA9']]

# Getting the head of the data
X.head()

# Setting-up the dependent variable
y = data['Value']

# Getting the head of the data
y.head()


# Setting the training set to 80% of the data
training = 0.8
t = int(training*len(data))

# Training dataset
X_train = X[:t]
y_train = y[:t]

# Testing dataset
X_test = X[t:]
y_test = y[t:]

#Linear Regression model
model = LinearRegression().fit(X_train,y_train)

#prediction
predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
predicted_price.plot(figsize=(10,5))
y_test.plot()
plt.legend(['Predicted Price','Actual Price'])
plt.ylabel("Crude Oil Prices: West Texas Intermediate")
plt.show()

# Computing the accuracy of our model
R_squared_score = model.score(X[t:],y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print ("The model has a " + accuracy + "% accuracy.")

# Let’s change the start date to “1990–01–01” and see how it affects our model’s accuracy!

# Importing our data and repeat the previous data
newdata = quandl.get("FRED/DCOILWTICO", start_date="1990-01-01", end_date="2020-12-12")
newdata['MA3'] = newdata['Value'].shift(1).rolling(window=3).mean()
newdata['MA9']= newdata['Value'].shift(1).rolling(window=9).mean()

# Tweaking the mean value to improve the accuracy

newdata['MA1'] = newdata['Value'].shift(1).rolling(window=1).mean()
newdata['MA2']= newdata['Value'].shift(1).rolling(window=2).mean()

# Dropping the NaN values
newdata = newdata.dropna()

# Initialising X and assigning the two feature variables
X = newdata[['MA1','MA2']]

# Getting the head of the data
X.head()

# Setting-up the dependent variable
y = newdata['Value']


# Setting the training set to 80% of the data
training = 0.8
t = int(training*len(data))

# Training dataset
X_train = X[:t]
y_train = y[:t]

# Testing dataset
X_test = X[t:]
y_test = y[t:]


#Linear Regression model
linear = LinearRegression().fit(X_train,y_train)

#prediction
predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
predicted_price.plot(figsize=(15,15))
y_test.plot()
plt.legend(['Predicted Price','Actual Price'])
plt.ylabel("Crude Oil Prices: West Texas Intermediate")
plt.show()


# Computing the accuracy of our model
R_squared_score = linear.score(X[t:],y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print ("The model has a " + accuracy + "% accuracy.")


# Computing the accuracy of our model
R_squared_score = linear.score(X[t:],y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print ("The model has a " + accuracy + "% accuracy.")
