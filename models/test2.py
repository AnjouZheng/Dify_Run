import pandas as pd
import matplotlib.pyplot as plt

# Assuming the data file is named 'APPL.csv' and contains a 'Date' and 'Close' column for dates and closing prices
df = pd.read_csv('APPL.csv')

plt.figure(figsize=(14,7))
plt.plot(df['Date'], df['Close'], label='APPL')
plt.title('APPL Stock Price Trend')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()
