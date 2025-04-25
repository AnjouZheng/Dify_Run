import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import datetime

# 读取数据
data = pd.read_csv('apple_stock_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# 股票价格走势图
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Close Price')
plt.title('Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('apple_stock_price_history.png')  # 保存历史价格图
plt.close()  # 关闭图形，不显示

# 预测接下来6个月
model = ARIMA(data['Close'], order=(5,1,0))
model_fit = model.fit(disp=0)
forecast, stderr, conf_int = model_fit.forecast(steps=180)

# 生成预测日期范围
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date, periods=180)
forecast_series = pd.Series(forecast, index=forecast_index)

# 预测股票价格走势图
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Historical Close Price')
plt.plot(forecast_series, label='Predicted Close Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('apple_stock_price_prediction.png')  # 保存预测图
plt.close()  # 关闭图形，不显示
