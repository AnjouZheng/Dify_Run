import pandas as pd
import matplotlib.pyplot as plt
import os

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

# 保存图片（统一为 result.png）
plt.savefig('result.png', dpi=300, bbox_inches='tight')

# Debug 输出
print("Image saved:", os.path.exists("result.png"))

# 不再显示图像（避免阻塞）
# plt.show() ← 注释掉这一行
