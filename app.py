import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from io import BytesIO
from fpdf import FPDF
import datetime
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Vishuddh PMS Pro", layout="wide")
st.title("📈 Vishuddh PMS - AI Stock Predictor (Pro Version)")

# Watchlist
watchlist_file = "watchlist.txt"
default_watchlist = "RELIANCE.NS"

st.sidebar.subheader("🔖 Watchlist")
symbols_input = st.sidebar.text_input("Enter NSE symbols (comma-separated)", default_watchlist)

if st.sidebar.button("💾 Save to Watchlist"):
    with open(watchlist_file, "w") as f:
        f.write(symbols_input)
    st.sidebar.success("Saved!")

if st.sidebar.button("📂 Load Watchlist"):
    if os.path.exists(watchlist_file):
        with open(watchlist_file, "r") as f:
            symbols_input = f.read()
        st.sidebar.success("Loaded!")

stocks = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def to_float(val):
    try:
        return float(val.item()) if hasattr(val, "item") else float(val)
    except:
        return 0.0

all_predictions = {}

for ticker in stocks:
    st.header(f"📊 {ticker} Analysis")

    with st.spinner("Downloading data..."):
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if df.empty or df.shape[0] < 60:
        st.warning(f"Not enough data for {ticker}")
        continue

    df['7MA'] = df['Close'].rolling(window=7).mean()
    df['30MA'] = df['Close'].rolling(window=30).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df['Day'] = np.arange(len(df))
    X = df[['Day', '7MA', '30MA', 'RSI', 'MACD', 'Signal']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        chart_dates = df.loc[y_test.index, 'Date'].values
        actual = y_test.values
        predicted = y_pred
        if len(actual) > 0 and len(predicted) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_dates, y=actual, name="Actual Price"))
            fig.add_trace(go.Scatter(x=chart_dates, y=predicted, name="Predicted Price"))
            fig.update_layout(title=f"{ticker} – Price Forecast", xaxis_title="Date", yaxis_title="₹")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("📉 Not enough prediction data to plot.")
    except Exception as e:
        st.warning(f"Chart rendering failed: {e}")

    # RSI Chart
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_fig.update_layout(title="RSI", yaxis_title="RSI")
    st.plotly_chart(rsi_fig, use_container_width=True)

    # MACD Chart
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='blue')))
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name="Signal", line=dict(color='orange')))
    macd_fig.update_layout(title="MACD", yaxis_title="MACD")
    st.plotly_chart(macd_fig, use_container_width=True)

    # Predict next price
    try:
        last = df.iloc[-1]
        features = np.array([[
            float(len(df)),
            to_float(last['7MA']),
            to_float(last['30MA']),
            to_float(last['RSI']),
            to_float(last['MACD']),
            to_float(last['Signal'])
        ]])
        next_price = float(model.predict(features)[0])
        st.success(f"📌 Predicted next price for {ticker}: ₹{next_price:.2f}")

        try:
            last_close_raw = df['Close'].iloc[-1]
            last_close = float(last_close_raw.item()) if hasattr(last_close_raw, "item") else float(last_close_raw)
            if next_price > last_close * 1.01:
                st.info("🔼 AI Insight: Likely Uptrend")
            elif next_price < last_close * 0.99:
                st.info("🔽 AI Insight: Possible Dip")
            else:
                st.info("⚖️ AI Insight: Sideways Market")
        except Exception as e:
            st.warning(f"AI Insight error: {e}")

    except Exception as e:
        st.warning(f"Prediction error: {e}")

    # Save Excel
    try:
        result_df = pd.DataFrame({
            "Date": df.loc[y_test.index, 'Date'].values,
            "Actual Price": y_test.values.flatten(),
            "Predicted Price": y_pred.flatten()
        })
        all_predictions[ticker] = result_df
    except Exception as e:
        st.warning(f"Excel save error: {e}")

    # PDF report
    if st.button(f"📄 Download PDF Report for {ticker}"):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Vishuddh PMS - AI Stock Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Stock: {ticker}", ln=True)
            pdf.cell(200, 10, txt=f"From: {start_date} To: {end_date}", ln=True)
            pdf.cell(200, 10, txt=f"Last Close: ₹{last_close:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Predicted: ₹{next_price:.2f}", ln=True)
            trend = "Uptrend" if next_price > last_close * 1.01 else "Downtrend" if next_price < last_close * 0.99 else "Sideways"
            pdf.cell(200, 10, txt=f"AI Trend Insight: {trend}", ln=True)
            pdf.ln(10)
            pdf.multi_cell(0, 10, "Note: This report is generated using AI. Use your own judgment when trading.")
            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button(
                label="⬇️ Save PDF",
                data=pdf_output.getvalue(),
                file_name=f"{ticker}_vishuddh_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF error: {e}")

# Excel download
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
