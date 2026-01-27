from flask import Flask, render_template, request, jsonify
from stock_analyzer import StockAnalyzer
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['symbol'].upper()
    date_input = request.form['date']

    analyzer = StockAnalyzer()
    data = analyzer.collect_data(symbol)

    last_date = data.index[-1]
    future_date = datetime.datetime.strptime(date_input, "%Y-%m-%d")
    days_ahead = (future_date - last_date).days

    if future_date in data.index:
        predicted = float(data.loc[future_date, 'close'])
    else:
        predicted = float(analyzer.predict_future_close(days_ahead))

    return jsonify({
        "symbol": symbol,
        "predicted_close": round(predicted, 2),
        "last_date": last_date.strftime('%d-%m-%Y')
    })


if __name__ == '__main__':
    app.run(debug=True)
