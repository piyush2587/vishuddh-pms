import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from io import BytesIO
import datetime

st.set_page_config(page_title="Vishuddh PMS – AI Stock Predictor", layout="wide")
st.title("📈 Vishuddh PMS – AI Stock Price Predictor")

# Sidebar stock selection
stocks = st.sidebar.multiselect(
    "Select one or more NSE stocks (e.g., RELIANCE.NS, TCS.NS)",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"],
    default=["RELIANCE.NS"]
)

start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Store predictions for Excel export
all_predictions = {}

for ticker in stocks:
    st.header(f"📊 {ticker} Prediction")

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or df.shape[0] < 40:
        st.warning(f"Not enough data for {ticker} to compute predictions.")
        continue

    df = df[['Close']].dropna().copy()
    df['Date'] = df.index
    df['7MA'] = df['Close'].rolling(window=7).mean()
    df['30MA'] = df['Close'].rolling(window=30).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare features
    df['Day'] = np.arange(len(df))
    X = df[['Day', '7MA', '30MA']]
    y = df['Close']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Build Plotly chart
    try:
        plot_dates = df.loc[y_test.index, 'Date'].values
        actual_prices = y_test.values.flatten()
        predicted_prices = y_pred.flatten()
        min_len = min(len(plot_dates), len(actual_prices), len(predicted_prices))

        if min_len == 0:
            st.warning("Not enough data to show chart.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_dates[:min_len],
                y=actual_prices[:min_len],
                name="Actual Price",
                mode='lines+markers',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=plot_dates[:min_len],
                y=predicted_prices[:min_len],
                name="Predicted Price",
                mode='lines+markers',
                line=dict(color='orange')
            ))
            fig.update_layout(
                title=f"{ticker} – Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart could not be displayed due to: {e}")

    # Predict next day's price (fixed Series ambiguity)
    try:
        last_row = df.iloc[-1]

        def to_float(x):
            return x.item() if hasattr(x, "item") else float(x)

        ma_7 = to_float(last_row['7MA'])
        ma_30 = to_float(last_row['30MA'])

        features = np.array([[len(df), ma_7, ma_30]])
        next_price = model.predict(features)[0]
        st.success(f"📌 Predicted next price for {ticker}: ₹{float(next_price):.2f}")
    except Exception as e:
        st.warning(f"Could not compute next price prediction for {ticker}: {e}")

    # Save for Excel export
    try:
        dates = df.loc[y_test.index, 'Date'].values
        actual = y_test.values.flatten()
        predicted = y_pred.flatten()
        min_len = min(len(dates), len(actual), len(predicted))

        result_df = pd.DataFrame({
            "Date": dates[:min_len],
            "Actual Price": actual[:min_len],
            "Predicted Price": predicted[:min_len]
        })
        all_predictions[ticker] = result_df
    except Exception as e:
        st.warning(f"Error creating Excel export for {ticker}: {e}")

# Export predictions to Excel
if all_predictions:
    try:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            for ticker, df_pred in all_predictions.items():
                df_pred.to_excel(writer, sheet_name=ticker[:31], index=False)
        st.download_button(
            label="📥 Download All Predictions as Excel",
            data=excel_buffer.getvalue(),
            file_name="vishuddh_pms_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.warning(f"Excel export failed: {e}")
