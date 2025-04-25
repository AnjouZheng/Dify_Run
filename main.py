import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # 注意新的导入路径
import datetime

# 读取数据
data = pd.read_csv('AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# 股票价格走势图
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Close Price')
plt.title('Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('apple_stock_price_history.png')
plt.close()

# 预测接下来6个月
model = ARIMA(data['Close'], order=(5,1,0))  # 现在使用新的 ARIMA
model_fit = model.fit()

# 生成预测
forecast = model_fit.get_forecast(steps=180)  # 新的预测方法
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# 生成预测日期范围
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date, periods=180)
forecast_series = pd.Series(forecast_mean, index=forecast_index)

# 预测股票价格走势图
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Historical Close Price')
plt.plot(forecast_series, label='Predicted Close Price')
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.3, label='Confidence Interval')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('apple_stock_price_prediction.png')
plt.close()
