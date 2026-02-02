# ================================================================
# STOCK PRICE PREDICTION APP - FIXED VERSION
# Author: Krishna Jha | IILM University
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Stock Price Prediction App")
st.markdown("**Machine Learning Powered | Built by Bhartendu Jha | IILM University**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Stock options
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

if market == "Indian (NSE)":
    stock_dict = indian_stocks
    currency = "₹"
else:
    stock_dict = us_stocks
    currency = "$"

selected_stock = st.sidebar.selectbox("Choose Stock", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]

# Date inputs
start_date = st.sidebar.date_input("Start Date", datetime(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Main function
if st.sidebar.button("🚀 Analyze & Predict", type="primary"):
    
    # Download data
    with st.spinner(f"Downloading {selected_stock} data..."):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(df) < 50:
                st.error("Not enough data. Please select a longer date range.")
                st.stop()
                
            st.success(f"✅ Loaded {len(df)} days of data!")
            
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "🤖 Prediction"])
    
    # TAB 1: Overview
    with tab1:
        st.header(f"{selected_stock} Overview")
        
        # Metrics - FIXED: Convert to float
        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        high_price = float(df['High'].max())
        low_price = float(df['Low'].min())
        avg_volume = float(df['Volume'].mean())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{currency}{current_price:,.2f}", f"{change_pct:+.2f}%")
        col2.metric("Period High", f"{currency}{high_price:,.2f}")
        col3.metric("Period Low", f"{currency}{low_price:,.2f}")
        col4.metric("Avg Volume", f"{avg_volume:,.0f}")
        
        # Price chart
        st.subheader("Price History")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['Close'], linewidth=1.5, color='#1E88E5')
        ax.fill_between(df.index, df['Close'], alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Price ({currency})')
        ax.set_title(f'{selected_stock} Stock Price')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Data table
        st.subheader("Recent Data")
        st.dataframe(df.tail(10).round(2))
    
    # TAB 2: Analysis
    with tab2:
        st.header("Technical Analysis")
        
        # Calculate indicators
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving average chart
        st.subheader("Moving Averages")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[-100:], df['Close'].tail(100), label='Close', linewidth=1.5)
        ax.plot(df.index[-100:], df['MA_20'].tail(100), label='MA 20', linestyle='--')
        ax.plot(df.index[-100:], df['MA_50'].tail(100), label='MA 50', linestyle='--')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # RSI chart
        st.subheader("RSI Indicator")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index[-100:], df['RSI'].tail(100), color='purple', linewidth=1.5)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax.fill_between(df.index[-100:], 30, 70, alpha=0.1, color='gray')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        current_rsi = float(df['RSI'].iloc[-1])
        if current_rsi > 70:
            st.warning(f"⚠️ RSI = {current_rsi:.1f} - OVERBOUGHT")
        elif current_rsi < 30:
            st.success(f"✅ RSI = {current_rsi:.1f} - OVERSOLD")
        else:
            st.info(f"ℹ️ RSI = {current_rsi:.1f} - Neutral")
    
    # TAB 3: Prediction
    with tab3:
        st.header("ML Prediction")
        
        with st.spinner("Training model..."):
            # Prepare data
            data = df.copy()
            data['Return'] = data['Close'].pct_change()
            data['MA_5'] = data['Close'].rolling(5).mean()
            data['MA_10'] = data['Close'].rolling(10).mean()
            data['Volatility'] = data['Return'].rolling(20).std()
            
            for lag in [1, 2, 3, 5]:
                data[f'Lag_{lag}'] = data['Close'].shift(lag)
            
            data['Target'] = data['Close'].shift(-1)
            data = data.dropna()
            
            # Features
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA_5', 'MA_10', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
            features = [f for f in features if f in data.columns]
            
            X = data[features]
            y = data['Target']
            
            # Split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Metrics - FIXED: Convert to float
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
        
        # Show metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE", f"{currency}{mae:.2f}")
        col3.metric("RMSE", f"{currency}{rmse:.2f}")
        
        # Tomorrow prediction
        st.subheader("🔮 Tomorrow's Prediction")
        
        last_features = X.iloc[-1:].values
        last_scaled = scaler.transform(last_features)
        predicted_price = float(model.predict(last_scaled)[0])
        
        current = float(data['Close'].iloc[-1])
        pred_change = ((predicted_price - current) / current) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Today's Close", f"{currency}{current:,.2f}")
        col2.metric("Predicted Tomorrow", f"{currency}{predicted_price:,.2f}", f"{pred_change:+.2f}%")
        col3.metric("Confidence", f"{r2*100:.1f}%")
        
        # Recommendation
        st.subheader("💡 Recommendation")
        if pred_change > 1 and current_rsi < 70:
            st.success("### 📈 BULLISH - Price may go UP")
        elif pred_change < -1 and current_rsi > 30:
            st.error("### 📉 BEARISH - Price may go DOWN")
        else:
            st.info("### ➡️ NEUTRAL - Hold position")
        
        # Actual vs Predicted chart
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(y_test)), y_test.values, label='Actual', linewidth=2)
        ax.plot(range(len(y_pred)), y_pred, label='Predicted', linewidth=2, alpha=0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: For educational purposes only. Not financial advice!")

else:
    st.info("👈 Select a stock and click **'Analyze & Predict'** to start!")
    
    st.markdown("""
    ### How to Use:
    1. Select Indian or US market
    2. Choose a stock
    3. Set date range
    4. Click 'Analyze & Predict'
    
    ### Features:
    - 📊 Price visualization
    - 📈 Technical indicators (RSI, Moving Averages)
    - 🤖 ML-based price prediction
    - 🎯 Feature importance analysis
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by **Krishna Jha** | IILM University | Data Science PBL")
