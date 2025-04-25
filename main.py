```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from datetime import timedelta

# Load data
df = pd.read_csv('AAPL.csv')

# convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Plot stock prices
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Close'], label='Close Price history')
plt.savefig('stock_price.png', dpi=300)
plt.close()

# Prepare data for prediction
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

# Split data into 'X' features and 'Y' target datasets
X = df['Date'].values.reshape(-1,1)
y = df['Close'].values.reshape(-1,1)

# Split data into Train and Test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regressor
regressor = LinearRegression()  

# Train the model using the training sets
regressor.fit(X_train, y_train)

# Predict
X_future = np.array(range(max(X)[0], max(X)[0] + timedelta(days=180).days)).reshape(-1,1)
predicted_price = regressor.predict(X_future)

# Plot predicted prices
plt.figure(figsize=(10,6))
plt.plot([pd.Timestamp.fromordinal(i[0]) for i in X_future], predicted_price, label='Predicted Price')
plt.savefig('predicted_price.png', dpi=300)
plt.close()
```
