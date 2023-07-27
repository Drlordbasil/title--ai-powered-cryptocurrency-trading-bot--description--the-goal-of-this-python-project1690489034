import json
import requests
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
from datetime import datetime

# Data Collection and Preprocessing
def fetch_cryptocurrency_data(exchange_name, cryptocurrency, timeframe):
    url = f"https://api.{exchange_name}.com/data/v2/histohour?symbol={cryptocurrency}&limit=1000&aggregate={timeframe}"
    response = requests.get(url)
    data = response.json()
    return data['Data']['Data']

def preprocess_data(data):
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'close', 'low', 'high', 'volumefrom', 'volumeto']]
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

exchange_name = 'example_exchange'  # Replace with actual exchange name
cryptocurrency = 'BTC'  # Replace with actual cryptocurrency symbol
timeframe = 1  # Replace with desired timeframe in hours

data = fetch_cryptocurrency_data(exchange_name, cryptocurrency, timeframe)
df = preprocess_data(data)

# Sentiment Analysis
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def calculate_average_sentiment_score(texts):
    total_score = 0
    for text in texts:
        total_score += analyze_sentiment(text)
    return total_score / len(texts)

sample_texts = ['This is great!', 'I am not sure.', 'Awesome project!']
average_sentiment_score = calculate_average_sentiment_score(sample_texts)
print("Average Sentiment Score:", average_sentiment_score)

# Technical Analysis
def calculate_technical_indicators(df):
    df['MA'] = df['close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['close'], 14)
    df['Bollinger Bands'] = calculate_bollinger_bands(df['close'], window=20)
    return df

def calculate_rsi(close_prices, window):
    delta = close_prices.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window).mean()
    avg_loss = abs(down.rolling(window).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(close_prices, window):
    rolling_mean = close_prices.rolling(window).mean()
    rolling_std = close_prices.rolling(window).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    return upper_band, lower_band

df = calculate_technical_indicators(df)

# Machine Learning
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']].values)

def prepare_train_data(data, num_time_steps):
    X, y = [], []
    for i in range(num_time_steps, len(data)):
        X.append(data[i - num_time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def create_lstm_model(num_time_steps):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(num_time_steps, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

num_time_steps = 60  # Replace with desired number of time steps

X_train, y_train = prepare_train_data(scaled_data, num_time_steps)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = create_lstm_model(num_time_steps)
model.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=True)

# Trading Strategy Execution
def execute_trading_strategy(data):
    signals = np.zeros(len(data))
    for i in range(num_time_steps, len(data)):
        prev_data = data[i - num_time_steps:i, :]
        prev_data_scaled = scaler.transform(prev_data)
        prev_data_reshaped = np.reshape(prev_data_scaled, (1, prev_data_scaled.shape[0], prev_data_scaled.shape[1]))
        predicted_price = model.predict(prev_data_reshaped)
        if predicted_price > data[i - 1]:
            signals[i] = 1  # Buy signal
        else:
            signals[i] = -1  # Sell signal
    return signals

df['Signals'] = execute_trading_strategy(scaled_data)

# Performance Evaluation
def calculate_performance_metrics(data, signals):
    investment_return = np.cumsum(np.diff(data['close']) * signals[:-1])
    sharpe_ratio = np.mean(investment_return) / np.std(investment_return)
    peak = np.maximum.accumulate(data['close'])
    drawdown = (peak - data['close']) / peak
    max_drawdown = np.max(drawdown)
    return investment_return, sharpe_ratio, drawdown, max_drawdown

investment_return, sharpe_ratio, drawdown, max_drawdown = calculate_performance_metrics(df, df['Signals'])

# Visualizations
def plot_performance(data, investment_return, drawdown):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(data.index, data['close'])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title('Cryptocurrency Price')
    ax2.plot(data.index[:-1], investment_return)
    ax2.fill_between(data.index[:-1], drawdown, 0, alpha=0.3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Return')
    ax2.set_title('Investment Return and Drawdown')
    plt.tight_layout()
    plt.show()

plot_performance(df, investment_return, drawdown)

# Reporting
def generate_report(data, investment_return, sharpe_ratio, max_drawdown):
    report = f"Report generated on: {datetime.now()}\n"
    report += f"Exchange: {exchange_name}\n"
    report += f"Cryptocurrency: {cryptocurrency}\n"
    report += f"Timeframe: {timeframe} hours\n\n"

    report += f"Performance Metrics:\n"
    report += f"   - Investment Return: {investment_return[-1]:.2f}\n"
    report += f"   - Sharpe Ratio: {sharpe_ratio:.2f}\n"
    report += f"   - Max Drawdown: {max_drawdown * 100:.2f}%\n\n"

    report += "Trading Signals:\n"
    report += "   - Buy Signal: 1\n"
    report += "   - Sell Signal: -1\n\n"

    report += "Sample Data:\n"
    report += f"{data.head()}\n\n"

    return report

report = generate_report(df, investment_return, sharpe_ratio, max_drawdown)
print(report)