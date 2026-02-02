# ================================================================
# STOCK PRICE PREDICTION APP - SIMPLE WORKING VERSION
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
st.markdown("**Machine Learning Powered | Built by Bhartendu Jha | IILM University**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")

indian_stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "Tata Motors": "TATAMOTORS.NS",
}

us_stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
}

market = st.sidebar.radio("Select Market", ["Indian (NSE)", "US"])
stock_dict = indian_stocks if market == "Indian (NSE)" else us_stocks
currency = "₹" if market == "Indian (NSE)" else "$"

selected_stock = st.sidebar.selectbox("Choose Stock", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]

start_date = st.sidebar.date_input("Start Date", datetime(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

if st.sidebar.button("🚀 Analyze & Predict", type="primary"):
    
    with st.spinner(f"Downloading {selected_stock} data..."):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Fix MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Convert to simple format
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            if len(df) < 50:
                st.error("Not enough data!")
                st.stop()
                
            st.success(f"✅ Loaded {len(df)} days of data!")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "🤖 Prediction"])
    
    # TAB 1: Overview
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
        
        # Simple line chart (Streamlit native - no errors!)
        st.subheader("📈 Price History")
        st.line_chart(df['Close'])
        
        # Data table
        st.subheader("📋 Recent Data")
        st.dataframe(df.tail(10).round(2))
    
    # TAB 2: Analysis
    with tab2:
        st.header("📈 Technical Analysis")
        
        # Calculate indicators
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))
        
        # Moving Averages Chart
        st.subheader("Moving Averages")
        ma_df = df[['Close', 'MA_20', 'MA_50']].tail(200)
        st.line_chart(ma_df)
        
        # RSI
        st.subheader("RSI Indicator")
        rsi_df = df[['RSI']].tail(100)
        st.line_chart(rsi_df)
        
        current_rsi = float(df['RSI'].iloc[-1])
        if current_rsi > 70:
            st.warning(f"⚠️ RSI = {current_rsi:.1f} - OVERBOUGHT")
        elif current_rsi < 30:
            st.success(f"✅ RSI = {current_rsi:.1f} - OVERSOLD")
        else:
            st.info(f"ℹ️ RSI = {current_rsi:.1f} - Neutral")
        
        st.markdown("""
        **RSI Guide:**
        - Above 70 = Overbought (may fall)
        - Below 30 = Oversold (may rise)
        - 30-70 = Neutral zone
        """)
    
    # TAB 3: Prediction
    with tab3:
        st.header("🤖 ML Prediction")
        
        with st.spinner("Training model..."):
            # Prepare data
            data = df.copy()
            data['Return'] = data['Close'].pct_change()
            data['MA_5'] = data['Close'].rolling(5).mean()
            data['MA_10'] = data['Close'].rolling(10).mean()
            
            for lag in [1, 2, 3, 5]:
                data[f'Lag_{lag}'] = data['Close'].shift(lag)
            
            data['Target'] = data['Close'].shift(-1)
            data = data.dropna()
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 
                       'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
            
            X = data[features].values
            y = data['Target'].values
            
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_s, y_train)
            
            y_pred = model.predict(X_test_s)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
        
        # Metrics
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE", f"{currency}{mae:.2f}")
        col3.metric("RMSE", f"{currency}{rmse:.2f}")
        
        st.progress(min(r2, 1.0), text=f"Model Accuracy: {r2*100:.1f}%")
        
        # Tomorrow's Prediction
        st.subheader("🔮 Tomorrow's Prediction")
        
        last_row = data[features].iloc[-1:].values
        last_scaled = scaler.transform(last_row)
        predicted = float(model.predict(last_scaled)[0])
        current = float(data['Close'].iloc[-1])
        change = ((predicted - current) / current) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Today's Close", f"{currency}{current:,.2f}")
        col2.metric("Predicted Tomorrow", f"{currency}{predicted:,.2f}", f"{change:+.2f}%")
        col3.metric("Confidence", f"{r2*100:.1f}%")
        
        # Recommendation
        st.subheader("💡 Recommendation")
        if change > 1 and current_rsi < 70:
            st.success("### 📈 BULLISH - Price may go UP")
        elif change < -1 and current_rsi > 30:
            st.error("### 📉 BEARISH - Price may go DOWN")
        else:
            st.info("### ➡️ NEUTRAL - Hold position")
        
        # Actual vs Predicted
        st.subheader("📉 Actual vs Predicted")
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        st.line_chart(comparison_df)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: For educational purposes only. Not financial advice!")

else:
    st.info("👈 Select a stock and click **'Analyze & Predict'** to start!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📌 How to Use
        1. Select **Indian** or **US** market
        2. Choose a **stock**
        3. Set **date range**
        4. Click **'Analyze & Predict'**
        """)
    with col2:
        st.markdown("""
        ### 🛠️ Features
        - 📊 Price visualization
        - 📈 Technical indicators
        - 🤖 ML price prediction
        - 🎯 Feature importance
        """)

st.markdown("---")
st.markdown("Built with ❤️ by **Krishna Jha** | IILM University")
