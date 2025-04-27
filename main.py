import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取本地AAPL.csv文件
file_path = 'AAPL.csv'
df = pd.read_csv(file_path)

# 确保日期列是datetime类型
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], marker='o')
plt.title('AAPL Weekly Closing Price')
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
