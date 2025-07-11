import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Vishuddh PMS - Stock Predictor", layout="wide")
st.title("📈 Vishuddh PMS - Stock Price Predictor (ML Powered)")

# User input
ticker = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

if st.button("Predict"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found. Please check the stock symbol or date range.")
    else:
        st.subheader("Raw Closing Price Data")
        st.line_chart(df['Close'])

        # Prepare data
        df = df[['Close']].dropna().reset_index()
        df['Day'] = np.arange(len(df))

        X = df[['Day']]
        y = df['Close']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Plot results
        st.subheader("📊 Predicted vs Actual Closing Prices")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label="Actual Price", color='blue')
        ax.plot(y_test.index, y_pred, label="Predicted Price", color='orange')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Show final prediction
        future_day = [[len(df)]]
        future_price = model.predict(future_day)[0]
        st.success(f"📌 Predicted next price: ₹{float(future_price):.2f}")

