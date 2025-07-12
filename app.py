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

st.set_page_config(page_title="Vishuddh PMS Pro", layout="wide")
st.title("📈 Vishuddh PMS – AI Stock Predictor (Pro Version)")

# --- Watchlist Load ---
watchlist_file = "watchlist.txt"
default_watchlist = "RELIANCE.NS"

st.sidebar.subheader("🔖 Watchlist Manager")
symbols_input = st.sidebar.text_input("Enter NSE symbols (comma-separated)", default_watchlist)

if st.sidebar.button("💾 Save to Watchlist"):
    with open(watchlist_file, "w") as f:
        f.write(symbols_input)
    st.sidebar.success("Watchlist saved!")

if st.sidebar.button("📂 Load Watchlist"):
    if os.path.exists(watchlist_file):
        with open(watchlist_file, "r") as f:
            symbols_input = f.read()
        st.sidebar.success("Watchlist loaded!")
    else:
        st.sidebar.warning("No saved watchlist found.")

stocks = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
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

all_predictions = {}

for ticker in stocks:
    st.header(f"📊 {ticker} Analysis")

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or df.shape[0] < 60:
        st.warning(f"Not enough data for {ticker} to compute predictions.")
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

    # Plot predicted vs actual
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.loc[y_test.index, 'Date'], y=y_test, name="Actual Price"))
    fig.add_trace(go.Scatter(x=df.loc[y_test.index, 'Date'], y=y_pred, name="Predicted Price"))
    fig.update_layout(title=f"{ticker} – Prediction vs Actual", xaxis_title="Date", yaxis_title="₹ Price")
    st.plotly_chart(fig, use_container_width=True)

    # RSI & MACD subcharts
    st.subheader(f"📈 Technical Indicators: RSI & MACD – {ticker}")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_fig.update_layout(title="RSI Indicator", yaxis_title="RSI")
    st.plotly_chart(rsi_fig, use_container_width=True)

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='blue')))
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name="Signal", line=dict(color='orange')))
    macd_fig.update_layout(title="MACD & Signal Line", yaxis_title="MACD")
    st.plotly_chart(macd_fig, use_container_width=True)

    # Predict next day's price
    try:
        last = df.iloc[-1]
        features = np.array([[len(df), last['7MA'], last['30MA'], last['RSI'], last['MACD'], last['Signal']]])
        next_price = float(model.predict(features)[0])
        st.success(f"📌 Predicted next price for {ticker}: ₹{next_price:.2f}")

        # AI Insight
        last_close = df['Close'].iloc[-1]
        if next_price > last_close * 1.01:
            st.info("🔼 AI Insight: Likely Uptrend")
        elif next_price < last_close * 0.99:
            st.info("🔽 AI Insight: Possible Dip")
        else:
            st.info("⚖️ AI Insight: Sideways Market")
    except Exception as e:
        st.warning(f"Could not compute next price prediction for {ticker}: {e}")

    # Save predictions for Excel
    try:
        result_df = pd.DataFrame({
            "Date": df.loc[y_test.index, 'Date'].values,
            "Actual Price": y_test.values.flatten(),
            "Predicted Price": y_pred.flatten()
        })
        all_predictions[ticker] = result_df
    except Exception as e:
        st.warning(f"Error creating Excel data for {ticker}: {e}")

    # PDF Report Generator
    if st.button(f"📄 Download PDF Report for {ticker}"):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt=f"Vishuddh PMS – AI Prediction Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Stock: {ticker}", ln=True)
            pdf.cell(200, 10, txt=f"Date Range: {start_date} to {end_date}", ln=True)
            pdf.cell(200, 10, txt=f"Last Close: ₹{last_close:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Predicted Next Price: ₹{next_price:.2f}", ln=True)
            pdf.cell(200, 10, txt="Trend: " +
                     ("Uptrend" if next_price > last_close * 1.01 else
                      "Downtrend" if next_price < last_close * 0.99 else "Sideways"), ln=True)
            pdf.ln(10)
            pdf.cell(200, 10, txt="This report is auto-generated by Vishuddh PMS", ln=True)

            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button(
                label="⬇️ Save PDF",
                data=pdf_output.getvalue(),
                file_name=f"{ticker}_prediction_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

# --- Excel Export ---
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

