
# coding: utf-8

# # Recurrent Neural Network to Predict Stock Prices
#
# ## Background
#
# This study is based on a paper from Stanford University.
#
# http://cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTimeSeriesPredictionwithRecurrentNeural.pdf
#
#
# ## Introduction
#
# Recurrent Neural Networks are excellent to use along with time series analysis to predict stock prices. What is time series analysis? Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values.
#
# An example is this. Would today affect the stock prices of tomorrow? Would last week affect the stock prices of tomorrow? How about last month? Last year? Seasons or fiscal quarters? Decades? Although stock advisors may have different opinions, recurrent neural networks uses every single case and finds the best method to predict.
#
# Problem: Client wants to know when to invest to get largest return in 2017.
#
# Data: 37 years of Apple stock prices. (1980-2017)
#
# Solution: Use recurrent neural networks to predict Apple stock prices in 2017 using data from 1980-2016.

# ## Data Visualization

# In[1]:

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading CSV file into training set
training_set = pd.read_csv('AAPL.csv')
training_set.head()



# ## Data Preprocessing
#
# Here, we are only interested in the opening price of the stock, so we just get that feature out.

# In[3]:

# Getting relevant feature
training_set = training_set.iloc[:,1:2]
training_set.head()


# In[4]:

# Converting to 2D array
training_set = training_set.values
training_set


# Scaling our features using normalization

# In[5]:

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
training_set


# Now we split our stock prices by shifting the cells one block. That way, the input would be one day and the output would be the very next day.

# In[6]:

# Getting the inputs and the ouputs
# 2012-2016
#X_train = training_set[7834:9091]
#y_train = training_set[7835:9092]
# 1980-2016
X_train = training_set[0:9091]
y_train = training_set[1:9092]

# Example
today = pd.DataFrame(X_train[0:5])
tomorrow = pd.DataFrame(y_train[0:5])
ex = pd.concat([today, tomorrow], axis=1)
ex.columns = (['today', 'tomorrow'])
ex


# We reshape our data into 3 dimensions, [batch_size, timesteps, input_dim], for Keras package data processing.
#
# Our batch size will be 1257 for the amount of data we have. Our timesteps will be 1 for each day. Our input will be 1 for one data point per observation.

# In[7]:

# Reshaping into required shape for Keras
#X_train = np.reshape(X_train, (1257, 1, 1))
X_train = np.reshape(X_train, (9091, 1, 1))
X_train


# # Building the Recurrent Neural Network
#
# We import the Keras Sqeuential model since our recurrent neural network will be in order. We import the Dense layers since we will use it for the nodes. We finally used the LSTM layer for time series analysis.

# In[8]:

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# We call our recurrent neural network a regressor since we are predicting using regression, not classification.

# In[9]:

# Initializing the Recurrent Neural Network
regressor = Sequential()


# Now we add the input layer with a sigmoid activation function. We use none then 1 for our input shape since we don't know what time step to use yet but we do know that each day contains one input.

# In[10]:

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))


# Our output layer will return the predicted stock price for the next day.

# In[11]:

# Adding the output layer
regressor.add(Dense(units = 1))


# In[12]:

# Compiling the Recurrent Neural Network
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[ ]:

# Fitting the Recurrent Neural Network to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)


# # Making Predictions and Visualizing the Results

# To test our recurrent neural network model, we take the real stock prices of 2017 and compare it with our model with predictions.

# In[14]:

# Getting the real stock price of 2017
test_set = pd.read_csv('AAPL_2017.csv')
test_set.head()


# In[15]:

# Getting relevant feature
real_stock_price = test_set.iloc[:,1:2]
real_stock_price.head()


# In[16]:

# Converting to 2D array
real_stock_price = real_stock_price.values


# In[17]:

# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (230, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# For 175 days, the predicted price is extremelly close.

# In[22]:

# Visualizing the results
fig = plt.figure()
plt.plot(real_stock_price, color = 'red', label = 'Real Apple Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Apple Stock Price')
plt.legend()
#plt.show()

fig.savefig('plot1.png')


# Now we see if we can predict the past 5 years based on our model. At first it looks like the 'real tesla stock prices' isn't showing.

# In[19]:

# Getting the real stock price of 2012 - 2016
real_stock_price_train = pd.read_csv('AAPL.csv')
real_stock_price_train = real_stock_price_train.iloc[:,1:2].values

# Getting the predicted stock price of 2012 - 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# Visualising the results
fig = plt.figure()
plt.plot(real_stock_price_train, color = 'red', label = 'Real Apple Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('37-year Apple Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Apple Stock Price')
plt.legend()
#plt.show()

fig.savefig('plot2.png')


# However, after splitting the data, we can see that our predictions were also extremelly accurate.

# In[20]:
fig = plt.figure()
plt.plot(real_stock_price_train, color = 'red', label = 'Real Apple Stock Price')
plt.title('37-year Apple Stock Prices')
plt.xlabel('Days')
plt.ylabel('Apple Stock Price')
plt.legend()
#plt.show()

fig.savefig('plot3.png')


# In[21]:
fig = plt.figure()
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('37-year Apple Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Apple Stock Price')
plt.legend()
#plt.show()

fig.savefig('plot4.png')


# In[ ]:



