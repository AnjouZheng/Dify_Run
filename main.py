```python
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Fetch Apple stock data
ticker = 'AAPL'
start_date = '2023-01-01'
end_date = '2025-03-31'
df = yf.download(ticker, start=start_date, end=end_date, interval='1wk')

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], marker='o')
plt.title(f'{ticker} Weekly Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

# Format x-axis to show dates nicely
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

# Add grid for readability
plt.grid(True, linestyle='--', linewidth=0.5)

# Save the plot
plt.tight_layout()
plt.savefig('apple_stock_weekly_trend.png')
plt.close()
```