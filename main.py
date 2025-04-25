```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Load the data
df = pd.read_csv('AAPL.csv')

# Convert the date to pandas datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plot the stock price
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Close'], label='Stock Price')
plt.title('Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Prepare data for prediction
df['Date'] = df['Date'].map(dt.datetime.toordinal)
X = df['Date'].values.reshape(-1,1)
y = df['Close'].values.reshape(-1,1)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# Predict the stock price for the next 6 months
dates = pd.date_range(start='2022-01-01', periods=180).map(dt.datetime.toordinal)
predictions = regressor.predict(dates.reshape(-1, 1))

# Plot the predictions
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Close'], label='Stock Price')
plt.plot(dates, predictions, label='Predicted Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```