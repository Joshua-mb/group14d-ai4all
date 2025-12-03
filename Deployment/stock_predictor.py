import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Helper Functions
def drop_na(merged_df):
    merged_df = merged_df.dropna()
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

def apply_lag_features(merged_df):
    # add previous one-day values
    merged_df["prev1_open"] = merged_df["Open"].shift(1)
    merged_df["prev1_close"] = merged_df["Close"].shift(1)

    # add previous two-day values
    merged_df["prev2_open"] = merged_df["Open"].shift(2)
    merged_df["prev2_close"] = merged_df["Close"].shift(2)

    # previous 1 day sentiment
    merged_df["prev1_sentiment_compound"] = merged_df["sentiment_compound"].shift(1)

    # previous 2 day sentiment
    merged_df["prev2_sentiment_compound"] = merged_df["sentiment_compound"].shift(2)

    # previous 1 day volume
    merged_df["prev1_volume"] = merged_df["Volume"].shift(1)

    # previous 2 day volume
    merged_df["prev2_volume"] = merged_df["Volume"].shift(2)

    return merged_df


features = [
    'Open', 'Close', 'Volume', 'sentiment_compound', 'prev1_open', 'prev1_close','prev1_volume',
    'prev2_volume', 'prev2_open', 'prev2_close', 'prev1_sentiment_compound',
    'prev2_sentiment_compound'
]

# Train a separate neural network model for each stock and store it in a dictionary
stocks = ['AAPL', 'GOOG', 'INTC', 'META', 'MSFT']
models = {}
scalers = {}

for stock in stocks:
    stock_finance_df = pd.read_csv(f'./{stock} Data/{stock}_finance_data.csv')
    stock_sentiment_df = pd.read_csv(f'./{stock} Data/{stock}_avg_sentiment_data.csv')
    merged_df = pd.merge(stock_finance_df, stock_sentiment_df, on='Date', how='inner')
    merged_df = apply_lag_features(merged_df)
    merged_df = drop_na(merged_df)

    X = merged_df[features]
    y = merged_df['target'].astype(float)
    valid_idx = np.isin(y, [-1, 1])
    X = X[valid_idx]
    y = y[valid_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scalers[stock] = scaler

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Neural Network Model (same as your notebook)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    models[stock] = model


# Prediction function
def prediction_model(stock, sentiment_value):
    model = models[stock]
    scaler = scalers[stock]

    ticker = yf.Ticker(stock)
    hist = ticker.history(period="5d")

    curr_opening_price = hist['Open'].iloc[-1]
    curr_closing_price = hist['Close'].iloc[-1]
    curr_volume = hist['Volume'].iloc[-1]
    sentiment_compound = sentiment_value

    prev1_open = hist['Open'].iloc[-2]
    prev1_close = hist['Close'].iloc[-2]
    prev1_volume = hist['Volume'].iloc[-2]
    prev1_sentiment = sentiment_value

    prev2_open = hist['Open'].iloc[-3]
    prev2_close = hist['Close'].iloc[-3]
    prev2_volume = hist['Volume'].iloc[-3]
    prev2_sentiment = sentiment_value

    input_data = pd.DataFrame([[
        curr_opening_price, curr_closing_price, curr_volume, sentiment_compound,
        prev1_open, prev1_close, prev1_volume,
        prev2_volume, prev2_open, prev2_close,
        prev1_sentiment, prev2_sentiment
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]


# Streamlit UI
with st.form("Direction_Predictor"):
    st.title("Next Day Stock Price Prediction")
    selected_stock = st.selectbox("Select a stock", stocks)

    ticker_data = yf.Ticker(selected_stock)
    try:
        current_price = ticker_data.info.get('currentPrice', None)
        if current_price is None:
            current_price = ticker_data.history(period="1d")['Close'].iloc[-1]
        st.subheader(f"Current Stock Price: ${current_price:.2f}")
    except:
        st.subheader("Current Stock Price: Data not available")

    st.subheader("Based on what you've seen on social media, how positive or negative is the sentiment for this stock for the past week?")
    sentiment_value = st.select_slider("Select on a scale from -5 to 5", [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    clicked = st.form_submit_button(label="Submit")
    if clicked:
        prediction = prediction_model(selected_stock, sentiment_value)
        if prediction == 1:
            st.success("🚀 Predicted direction for next day: UP")
        else:
            st.error("📉 Predicted direction for next day: DOWN")