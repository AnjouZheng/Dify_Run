```python
import pandas as pd
import matplotlib.pyplot as plt

# Assuming that the data file is CSV and the column for dates is 'Date' and the column for prices is 'Price'
data = pd.read_csv('data.csv')

plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(data['Date']), data['Price'], label='APPL Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('APPL Stock Price Trend')
plt.legend()
plt.grid(True)
plt.show()
```