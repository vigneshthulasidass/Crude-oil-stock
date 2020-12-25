#!/usr/bin/env python
# coding: utf-8

# In[1]:


# A machine learning library used for linear regression
from sklearn.linear_model import LinearRegression
# numpy and pandas will be used for data manipulation
import numpy as np
import pandas as pd
# matplotlib will be used for visually representing our data
import matplotlib.pyplot as plt
# Quandl will be used for importing historical oil prices
import quandl


# In[2]:


# Setting our API key
quandl.ApiConfig.api_key = "RasVL2hRP88cgHWueMJE"

# Importing our data
data = quandl.get("FRED/DCOILWTICO", start_date="2000-01-01", end_date="2020-01-01")


# In[3]:


data.head()


# In[4]:


# Setting the text on the Y-axis
plt.ylabel("Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma")

# Setting the size of our graph
data.Value.plot(figsize=(15,8))


# In[5]:


#moving average of 3 and 9 days


# In[6]:


data['MA3'] = data['Value'].shift(1).rolling(window=3).mean()
data['MA9']= data['Value'].shift(1).rolling(window=9).mean()


# In[7]:


# Dropping the NaN values
data = data.dropna()

# Initialising X and assigning the two feature variables
X = data[['MA3','MA9']]

# Getting the head of the data
X.head()


# In[8]:


# Setting-up the dependent variable
y = data['Value']

# Getting the head of the data
y.head()


# In[9]:


# Setting the training set to 80% of the data
training = 0.8
t = int(training*len(data))

# Training dataset
X_train = X[:t]
y_train = y[:t]

# Testing dataset
X_test = X[t:]
y_test = y[t:]


# In[10]:


#Linear Regression model
model = LinearRegression().fit(X_train,y_train)


# In[11]:


#prediction
predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
predicted_price.plot(figsize=(10,5))
y_test.plot()
plt.legend(['Predicted Price','Actual Price'])
plt.ylabel("Crude Oil Prices: West Texas Intermediate")
plt.show()


# In[12]:


# Computing the accuracy of our model
R_squared_score = model.score(X[t:],y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print ("The model has a " + accuracy + "% accuracy.")


# In[13]:


# Let’s change the start date to “1990–01–01” and see how it affects our model’s accuracy!


# In[14]:


# Importing our data
newdata = quandl.get("FRED/DCOILWTICO", start_date="1990-01-01", end_date="2020-12-12")


# In[15]:


newdata['MA3'] = newdata['Value'].shift(1).rolling(window=3).mean()
newdata['MA9']= newdata['Value'].shift(1).rolling(window=9).mean()


# In[23]:


newdata['MA1'] = newdata['Value'].shift(1).rolling(window=1).mean()
newdata['MA2']= newdata['Value'].shift(1).rolling(window=2).mean()


# In[24]:


# Dropping the NaN values
newdata = newdata.dropna()

# Initialising X and assigning the two feature variables
X = newdata[['MA1','MA2']]

# Getting the head of the data
X.head()


# In[25]:


# Setting-up the dependent variable
y = newdata['Value']

# Getting the head of the data
y.head()


# In[26]:


# Setting the training set to 80% of the data
training = 0.8
t = int(training*len(data))

# Training dataset
X_train = X[:t]
y_train = y[:t]

# Testing dataset
X_test = X[t:]
y_test = y[t:]


# In[27]:


#Linear Regression model
linear = LinearRegression().fit(X_train,y_train)


# In[28]:


#prediction
predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
predicted_price.plot(figsize=(15,15))
y_test.plot()
plt.legend(['Predicted Price','Actual Price'])
plt.ylabel("Crude Oil Prices: West Texas Intermediate")
plt.show()


# In[22]:


# Computing the accuracy of our model
R_squared_score = linear.score(X[t:],y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print ("The model has a " + accuracy + "% accuracy.")


# In[29]:


# Computing the accuracy of our model
R_squared_score = linear.score(X[t:],y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print ("The model has a " + accuracy + "% accuracy.")


# In[ ]:




