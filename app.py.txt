import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import os

st.set_page_config(page_title="Vishuddh PMS - AI Stock Predictor", layout="wide")

st.title("📈 Vishuddh PMS - AI Stock Price Predictor")

ticker = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS)", "RELIANCE.NS")
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

if st.button("Predict"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("Invalid stock symbol or no data available.")
    else:
        st.subheader(f"{ticker} Stock Data")
        st.line_chart(df['Close'])

        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data) * 0.70)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - 100:]

        def create_dataset(dataset, time_step=100):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        X_test, Y_test = create_dataset(test_data, 100)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Load or train model
        from keras.models import Sequential
        from keras.layers import Dense, LSTM

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_test, Y_test, batch_size=1, epochs=1, verbose=0)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

        st.subheader("📊 Predicted vs Actual Closing Prices")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(Y_test_inv, label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig)

        last_pred = predictions[-1][0]
        st.success(f"📌 Last predicted price: ₹{last_pred:.2f}")
