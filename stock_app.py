# ================================================================
# STOCK PRICE PREDICTION APP - CLEAN VERSION
# Author: Krishna Jha | IILM University
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# Title
st.title("📈 Stock Price Prediction App")
st.markdown("**Machine Learning Powered | Built by Krishna | IILM University**")
st.markdown("---")

# Stock Data
INDIAN_STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "Tata Motors": "TATAMOTORS.NS",
}

US_STOCKS = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
}

# Sidebar
st.sidebar.header("⚙️ Settings")

market = st.sidebar.radio("Select Market", ["Indian (NSE)", "US"])
stock_dict = INDIAN_STOCKS if market == "Indian (NSE)" else US_STOCKS
currency = "₹" if market == "Indian (NSE)" else "$"

selected_stock = st.sidebar.selectbox("Choose Stock", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]

start_date = st.sidebar.date_input("Start Date", datetime(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())


# Helper function to clean data
def clean_stock_data(df):
    """Clean and flatten stock data from yfinance."""
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Ensure all numeric columns are proper 1D series
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            # Convert to numpy array, flatten, then back to series
            df[col] = pd.to_numeric(df[col].values.flatten(), errors='coerce')
    
    # Drop any NaN rows
    df = df.dropna()
    
    return df


if st.sidebar.button("🚀 Analyze & Predict", type="primary"):
    
    # Load Data
    with st.spinner(f"Downloading {selected_stock} data..."):
        try:
            raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if raw_df.empty:
                st.error("No data found!")
                st.stop()
            
            df = clean_stock_data(raw_df)
            
            if len(df) < 60:
                st.error("Not enough data! Please select a longer date range.")
                st.stop()
            
            st.success(f"✅ Loaded {len(df)} days of data!")
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "🤖 Prediction"])
    
    # =============================================
    # TAB 1: OVERVIEW
    # =============================================
    with tab1:
        st.header(f"{selected_stock} Overview")
        
        # Metrics
        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{currency}{current_price:,.2f}", f"{change_pct:+.2f}%")
        col2.metric("Period High", f"{currency}{float(df['High'].max()):,.2f}")
        col3.metric("Period Low", f"{currency}{float(df['Low'].min()):,.2f}")
        col4.metric("Avg Volume", f"{float(df['Volume'].mean()):,.0f}")
        
        # Price Chart - Using native Streamlit
        st.subheader("📈 Price History")
        price_chart_data = pd.DataFrame({'Close Price': df['Close'].values}, index=df.index)
        st.line_chart(price_chart_data)
        
        # Volume Chart
        st.subheader("📊 Trading Volume")
        volume_chart_data = pd.DataFrame({'Volume': df['Volume'].values}, index=df.index)
        st.bar_chart(volume_chart_data)
        
        # Data Table
        st.subheader("📋 Recent Data")
        st.dataframe(df.tail(10).round(2))
    
    # =============================================
    # TAB 2: ANALYSIS
    # =============================================
    with tab2:
        st.header("📈 Technical Analysis")
        
        # Calculate indicators
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages Chart
        st.subheader("📊 Moving Averages")
        ma_chart_data = pd.DataFrame({
            'Close': df['Close'].values,
            '20-Day MA': df['MA_20'].values,
            '50-Day MA': df['MA_50'].values
        }, index=df.index).dropna().tail(200)
        st.line_chart(ma_chart_data)
        
        # RSI Chart
        st.subheader("📊 RSI Indicator")
        rsi_chart_data = pd.DataFrame({'RSI': df['RSI'].values}, index=df.index).dropna().tail(100)
        st.line_chart(rsi_chart_data)
        
        # RSI Analysis
        current_rsi = float(df['RSI'].iloc[-1])
        if current_rsi > 70:
            st.warning(f"⚠️ RSI = {current_rsi:.1f} - **OVERBOUGHT** (Stock may be overvalued)")
        elif current_rsi < 30:
            st.success(f"✅ RSI = {current_rsi:.1f} - **OVERSOLD** (Stock may be undervalued)")
        else:
            st.info(f"ℹ️ RSI = {current_rsi:.1f} - **NEUTRAL**")
    
    # =============================================
    # TAB 3: PREDICTION
    # =============================================
    with tab3:
        st.header("🤖 ML Prediction")
        
        with st.spinner("Training model..."):
            # Prepare data
            ml_data = df.copy()
            ml_data['MA_5'] = ml_data['Close'].rolling(5).mean()
            ml_data['MA_10'] = ml_data['Close'].rolling(10).mean()
            
            for lag in [1, 2, 3, 5]:
                ml_data[f'Lag_{lag}'] = ml_data['Close'].shift(lag)
            
            ml_data['Target'] = ml_data['Close'].shift(-1)
            ml_data = ml_data.dropna()
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10',
                       'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
            
            X = ml_data[features].values
            y = ml_data['Target'].values
            
            # Split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Train
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_s, y_train)
            
            # Predict
            y_pred = model.predict(X_test_s)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
        
        # Display Results
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE", f"{currency}{mae:.2f}")
        col3.metric("RMSE", f"{currency}{rmse:.2f}")
        
        st.progress(min(max(r2, 0), 1.0), text=f"Model Accuracy: {r2*100:.1f}%")
        
        # Tomorrow's Prediction
        st.subheader("🔮 Tomorrow's Prediction")
        
        last_row = ml_data[features].iloc[-1:].values
        last_scaled = scaler.transform(last_row)
        predicted_price = float(model.predict(last_scaled)[0])
        current = float(ml_data['Close'].iloc[-1])
        change = ((predicted_price - current) / current) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Today's Close", f"{currency}{current:,.2f}")
        col2.metric("Predicted Tomorrow", f"{currency}{predicted_price:,.2f}", f"{change:+.2f}%")
        col3.metric("Confidence", f"{r2*100:.1f}%")
        
        # Recommendation
        st.subheader("💡 Recommendation")
        if change > 1 and current_rsi < 70:
            st.success("### 📈 BULLISH - Consider Buying")
        elif change < -1 and current_rsi > 30:
            st.error("### 📉 BEARISH - Consider Selling")
        else:
            st.info("### ➡️ NEUTRAL - Hold Position")
        
        # Actual vs Predicted Chart
        st.subheader("📉 Actual vs Predicted")
        comparison_data = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        st.line_chart(comparison_data)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        st.bar_chart(importance_df.set_index('Feature'))
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: For educational purposes only. Not financial advice!")

else:
    st.info("👈 Select a stock and click **'Analyze & Predict'** to start!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by **Krishna Jha** | IILM University")
