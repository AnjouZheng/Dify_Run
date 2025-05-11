import pandas as pd
import numpy as np
import backtrader as bt
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

@dataclass
class BacktestStrategy(bt.Strategy):
    params = (
        ('rsi_periods', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
        ('stop_loss', 0.05),
        ('take_profit', 0.10)
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data, period=self.params.rsi_periods)
        self.order = None
        self.stop_price = None
        self.take_profit_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.rsi < self.params.rsi_lower:
                size = int(self.broker.getcash() * 0.9 / self.data.close[0])
                self.order = self.buy(size=size)
                self.stop_price = self.data.close[0] * (1 - self.params.stop_loss)
                self.take_profit_price = self.data.close[0] * (1 + self.params.take_profit)

        else:
            if self.rsi > self.params.rsi_upper:
                self.order = self.sell(size=self.position.size)

            if self.data.close[0] <= self.stop_price or self.data.close[0] >= self.take_profit_price:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            logging.info(f"Order {order.ref} executed at {order.executed.price}")

        self.order = None

class StockBacktester:
    def __init__(self, data_path: str, initial_cash: float = 100000):
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(initial_cash)
        self.cerebro.broker.setcommission(commission=0.001)
        self.load_data(data_path)

    def load_data(self, data_path: str):
        dataframe = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        data = bt.feeds.PandasData(dataname=dataframe)
        self.cerebro.adddata(data)

    def run_backtest(self) -> Dict[str, float]:
        self.cerebro.addstrategy(BacktestStrategy)
        results = self.cerebro.run()[0]
        
        return {
            'start_value': self.cerebro.broker.startingcash,
            'end_value': self.cerebro.broker.getvalue(),
            'total_return': (self.cerebro.broker.getvalue() / self.cerebro.broker.startingcash - 1) * 100,
            'max_drawdown': results.analyzers.drawdown.get_analysis()['max']['drawdown']
        }

    def plot_results(self):
        self.cerebro.plot()

def main():
    backtester = StockBacktester('data/processed/stock_data.csv')
    results = backtester.run_backtest()
    
    logging.info("Backtest Results:")
    for key, value in results.items():
        logging.info(f"{key}: {value}")

    backtester.plot_results()

if __name__ == "__main__":
    main()