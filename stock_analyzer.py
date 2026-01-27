from tvDatafeed import TvDatafeed, Interval
from sklearn.linear_model import LinearRegression
import numpy as np

class StockDataCollector:
    def __init__(self):
        self.tv = TvDatafeed()
        self.data = None

    def collect_data(self, symbol, exchange='NASDAQ', n_bars=60, interval=Interval.in_daily):
        self.data = self.tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            n_bars=n_bars,
            interval=interval
        )
        return self.data


class StockAnalyzer(StockDataCollector):
    def analyze_trend(self):
        self.data['Trend'] = self.data['close'].diff().apply(
            lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'Stable')
        )
        return self.data[['close', 'Trend']]

    def predict_future_close(self, days_ahead):
        df = self.data.reset_index()
        df['day'] = np.arange(len(df))

        X = df[['day']]
        y = df['close']

        model = LinearRegression()
        model.fit(X, y)

        future_day = [[len(df) + days_ahead - 1]]
        return model.predict(future_day)[0]
