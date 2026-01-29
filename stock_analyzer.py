from tvDatafeed import TvDatafeed, Interval
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

class StockDataCollector:
    def __init__(self):
        self.tv = TvDatafeed()
        self.data = None

    def collect_data(self, symbol, exchange='NASDAQ', n_bars=None, interval=Interval.in_daily):
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

    def predict_future_close(self, days_ahead: int, model_type='linear'):
        df = self.data.reset_index()
        df['day'] = np.arange(len(df))

        X = df[['day']].values
        y = df['close'].values

        model_type = model_type.lower()
        
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X, y)
        elif model_type == 'ridge':
            model = Ridge()
            model.fit(X, y)
        elif model_type == 'polynomial':
            poly = PolynomialFeatures(degree=2)
            X = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X, y)
        elif model_type == 'svr':
            model = SVR(kernel='rbf')
            model.fit(X, y)
        elif model_type == 'randomforest':
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
        elif model_type == 'gradientboosting':
            model = GradientBoostingRegressor(n_estimators=100)
            model.fit(X, y)
        elif model_type == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
            model.fit(X, y)
        elif model_type == 'lstm':
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, 1))
            model = Sequential([LSTM(50, input_shape=(1, 1)), Dense(1)])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_lstm, y_scaled, epochs=50, verbose=0)
            future_day_scaled = scaler_X.transform(np.array([[len(df) + days_ahead - 1]]))
            future_day_lstm = future_day_scaled.reshape((1, 1, 1))
            pred_scaled = model.predict(future_day_lstm, verbose=0)[0][0]
            predicted = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            pred_prices = scaler_y.inverse_transform(model.predict(X_lstm, verbose=0)).flatten()
            mae = mean_absolute_error(y, pred_prices)
            mse = mean_squared_error(y, pred_prices)
            return predicted, mae, mse, y, pred_prices
        else:
            model = LinearRegression()
            model.fit(X, y)

        future_day = np.array([[len(df) + days_ahead - 1]])
        if model_type == 'polynomial':
            future_day = PolynomialFeatures(degree=2).fit_transform(future_day)
        
        predicted = model.predict(future_day)[0]
        mae = mean_absolute_error(y, model.predict(X))
        mse = mean_squared_error(y, model.predict(X))
        
        return predicted, mae, mse, y, model.predict(X)
