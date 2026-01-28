from flask import Flask, render_template, request, jsonify
from stock_analyzer import StockAnalyzer
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Ensure static folder exists
os.makedirs('static', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['symbol'].upper()
    date_input = request.form['date']
    model_type = request.form['model_type'].lower()
    data_points = int(request.form.get('data_points', 60))

    analyzer = StockAnalyzer()
    data = analyzer.collect_data(symbol, n_bars=data_points)

    last_date = data.index[-1]
    future_date = datetime.datetime.strptime(date_input, "%Y-%m-%d")
    days_ahead = (future_date - last_date).days

    if future_date in data.index:
        predicted = float(data.loc[future_date, 'close'])
        mae = 0
        mse = 0
        actual_prices = data['close'].values
        pred_prices = data['close'].values
    else:
        result = analyzer.predict_future_close(days_ahead, model_type=model_type)
        predicted, mae, mse = result[0], result[1], result[2]
        actual_prices, pred_prices = result[3], result[4]

    # Generate and save plot
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label='Actual Prices', linewidth=2)
    plt.plot(pred_prices, label='Model Predictions', linewidth=2)
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(f'{symbol} - Actual vs Predicted ({model_type.upper()})')
    plt.tight_layout()
    
    filename = f'plot_{symbol}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    filepath = os.path.join('static', filename)
    plt.savefig(filepath)
    plt.close()

    return jsonify({
        "symbol": symbol,
        "predicted_close": round(predicted, 2),
        "last_date": last_date.strftime('%d-%m-%Y'),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "plot": f'/static/{filename}'
    })


if __name__ == '__main__':
    app.run(debug=True)
