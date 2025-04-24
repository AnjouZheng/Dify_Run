import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('AAPL.csv')

# 创建图形
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='AAPL')
plt.title('AAPL Stock Price Trend')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)

# 保存图片（可以修改文件名和格式）
plt.savefig('AAPL_stock_trend.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
