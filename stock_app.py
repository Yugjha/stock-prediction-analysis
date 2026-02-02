# ================================================================
# STOCK PRICE PREDICTION APP
# Author: Krishna Jha | IILM University
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# STOCK DATA
# ================================================================
INDIAN_STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
}

US_STOCKS = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "Netflix": "NFLX",
    "NVIDIA": "NVDA",
}

# ================================================================
# HELPER FUNCTIONS
# ================================================================

@st.cache_data(ttl=3600)
def load_stock_data(ticker, start_date, end_date):
    """Download and clean stock data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None, "No data found for this stock."
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index and ensure Date is datetime
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Ensure all columns are numeric and 1D
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN in essential columns
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        return df, None
        
    except Exception as e:
        return None, str(e)


def calculate_technical_indicators(df):
    """Calculate technical indicators."""
    data = df.copy()
    
    # Moving Averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std * 2)
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Daily Returns
    data['Daily_Return'] = data['Close'].pct_change() * 100
    
    # Volatility (20-day)
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
    
    return data


def prepare_ml_data(df):
    """Prepare data for machine learning model."""
    data = df.copy()
    
    # Features
    data['Return'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(5).mean()
    data['MA_10'] = data['Close'].rolling(10).mean()
    data['MA_20'] = data['Close'].rolling(20).mean()
    
    # Lag features
    for lag in [1, 2, 3, 5, 7]:
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    
    # Price differences
    data['High_Low_Diff'] = data['High'] - data['Low']
    data['Close_Open_Diff'] = data['Close'] - data['Open']
    
    # Target: Next day's close
    data['Target'] = data['Close'].shift(-1)
    
    # Drop NaN values
    data = data.dropna()
    
    return data


def train_model(data):
    """Train Random Forest model and return predictions."""
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_5', 'MA_10', 'MA_20',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Lag_7',
        'High_Low_Diff', 'Close_Open_Diff'
    ]
    
    X = data[features].values
    y = data['Target'].values
    
    # Train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Tomorrow's prediction
    last_row = data[features].iloc[-1:].values
    last_scaled = scaler.transform(last_row)
    tomorrow_pred = model.predict(last_scaled)[0]
    
    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'y_test': y_test,
        'y_pred': y_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'tomorrow_pred': tomorrow_pred,
        'current_price': data['Close'].iloc[-1]
    }


def get_recommendation(predicted_change, rsi):
    """Generate trading recommendation."""
    if predicted_change > 2 and rsi < 70:
        return "STRONG BUY", "success", "📈"
    elif predicted_change > 0.5 and rsi < 70:
        return "BUY", "success", "📈"
    elif predicted_change < -2 and rsi > 30:
        return "STRONG SELL", "error", "📉"
    elif predicted_change < -0.5 and rsi > 30:
        return "SELL", "error", "📉"
    else:
        return "HOLD", "info", "➡️"


# ================================================================
# MAIN APP
# ================================================================

def main():
    # Header
    st.markdown('<p class="main-header">📈 Stock Price Prediction App</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Powered | Built by Bhartendu Jha | IILM University</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("---")
        
        # Market Selection
        market = st.radio(
            "🌍 Select Market",
            ["Indian (NSE)", "US"],
            horizontal=True
        )
        
        stock_dict = INDIAN_STOCKS if market == "Indian (NSE)" else US_STOCKS
        currency = "₹" if market == "Indian (NSE)" else "$"
        
        # Stock Selection
        selected_stock = st.selectbox(
            "📊 Choose Stock",
            list(stock_dict.keys())
        )
        ticker = stock_dict[selected_stock]
        
        st.markdown("---")
        
        # Date Range
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                datetime(2021, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "End",
                datetime.today()
            )
        
        st.markdown("---")
        
        # Analyze Button
        analyze_btn = st.button(
            "🚀 Analyze & Predict",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### 📌 Quick Info")
        st.info(f"**Stock:** {selected_stock}\n\n**Ticker:** {ticker}")
    
    # Main Content
    if analyze_btn:
        # Load Data
        with st.spinner(f"📥 Loading {selected_stock} data..."):
            df, error = load_stock_data(ticker, start_date, end_date)
        
        if error:
            st.error(f"❌ Error: {error}")
            st.stop()
        
        if df is None or len(df) < 60:
            st.error("❌ Not enough data! Please select a longer date range (minimum 60 days).")
            st.stop()
        
        st.success(f"✅ Successfully loaded **{len(df)}** days of data!")
        
        # Calculate indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Technical Analysis", "🤖 ML Prediction"])
        
        # ============================================================
        # TAB 1: OVERVIEW
        # ============================================================
        with tab1:
            st.header(f"📊 {selected_stock} Overview")
            
            # Key Metrics
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "💰 Current Price",
                    f"{currency}{current_price:,.2f}",
                    f"{change_pct:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "📈 Period High",
                    f"{currency}{float(df['High'].max()):,.2f}"
                )
            
            with col3:
                st.metric(
                    "📉 Period Low",
                    f"{currency}{float(df['Low'].min()):,.2f}"
                )
            
            with col4:
                avg_volume = float(df['Volume'].mean())
                st.metric(
                    "📊 Avg Volume",
                    f"{avg_volume:,.0f}"
                )
            
            with col5:
                volatility = float(df_with_indicators['Volatility'].iloc[-1])
                st.metric(
                    "📉 Volatility",
                    f"{volatility:.2f}%"
                )
            
            st.markdown("---")
            
            # Price Chart
            st.subheader("📈 Price History")
            chart_data = df[['Close']].copy()
            chart_data.columns = ['Close Price']
            st.line_chart(chart_data, use_container_width=True)
            
            # Volume Chart
            st.subheader("📊 Trading Volume")
            volume_data = df[['Volume']].copy()
            st.bar_chart(volume_data, use_container_width=True)
            
            # Recent Data Table
            st.subheader("📋 Recent Data (Last 10 Days)")
            recent_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).round(2)
            recent_data['Volume'] = recent_data['Volume'].astype(int)
            st.dataframe(recent_data, use_container_width=True)
        
        # ============================================================
        # TAB 2: TECHNICAL ANALYSIS
        # ============================================================
        with tab2:
            st.header("📈 Technical Analysis")
            
            # Moving Averages
            st.subheader("📊 Moving Averages")
            ma_data = df_with_indicators[['Close', 'MA_20', 'MA_50']].dropna().tail(200)
            ma_data.columns = ['Close', '20-Day MA', '50-Day MA']
            st.line_chart(ma_data, use_container_width=True)
            
            # MA Analysis
            current_ma20 = float(df_with_indicators['MA_20'].iloc[-1])
            current_ma50 = float(df_with_indicators['MA_50'].iloc[-1])
            
            col1, col2 = st.columns(2)
            with col1:
                if current_price > current_ma20:
                    st.success(f"✅ Price ({currency}{current_price:.2f}) is **ABOVE** 20-Day MA ({currency}{current_ma20:.2f})")
                else:
                    st.warning(f"⚠️ Price ({currency}{current_price:.2f}) is **BELOW** 20-Day MA ({currency}{current_ma20:.2f})")
            
            with col2:
                if current_ma20 > current_ma50:
                    st.success("✅ **Golden Cross**: 20-Day MA is above 50-Day MA (Bullish)")
                else:
                    st.warning("⚠️ **Death Cross**: 20-Day MA is below 50-Day MA (Bearish)")
            
            st.markdown("---")
            
            # RSI
            st.subheader("📊 RSI (Relative Strength Index)")
            rsi_data = df_with_indicators[['RSI']].dropna().tail(100)
            st.line_chart(rsi_data, use_container_width=True)
            
            current_rsi = float(df_with_indicators['RSI'].iloc[-1])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current RSI", f"{current_rsi:.2f}")
            with col2:
                if current_rsi > 70:
                    st.error("🔴 OVERBOUGHT")
                elif current_rsi < 30:
                    st.success("🟢 OVERSOLD")
                else:
                    st.info("🟡 NEUTRAL")
            with col3:
                st.markdown("""
                **RSI Guide:**
                - \> 70: Overbought
                - < 30: Oversold
                - 30-70: Neutral
                """)
            
            st.markdown("---")
            
            # Bollinger Bands
            st.subheader("📊 Bollinger Bands")
            bb_data = df_with_indicators[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']].dropna().tail(100)
            bb_data.columns = ['Close', 'Upper Band', 'Middle Band', 'Lower Band']
            st.line_chart(bb_data, use_container_width=True)
            
            st.markdown("---")
            
            # MACD
            st.subheader("📊 MACD")
            macd_data = df_with_indicators[['MACD', 'MACD_Signal']].dropna().tail(100)
            macd_data.columns = ['MACD', 'Signal Line']
            st.line_chart(macd_data, use_container_width=True)
        
        # ============================================================
        # TAB 3: ML PREDICTION
        # ============================================================
        with tab3:
            st.header("🤖 Machine Learning Prediction")
            
            with st.spinner("🔄 Training ML model..."):
                ml_data = prepare_ml_data(df)
                results = train_model(ml_data)
            
            st.success("✅ Model trained successfully!")
            
            # Model Performance
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 R² Score", f"{results['r2']:.4f}")
            with col2:
                st.metric("📏 MAE", f"{currency}{results['mae']:.2f}")
            with col3:
                st.metric("📐 RMSE", f"{currency}{results['rmse']:.2f}")
            with col4:
                accuracy_pct = max(0, min(results['r2'] * 100, 100))
                st.metric("🎯 Accuracy", f"{accuracy_pct:.1f}%")
            
            # Accuracy Progress Bar
            st.progress(min(results['r2'], 1.0), text=f"Model Confidence: {accuracy_pct:.1f}%")
            
            st.markdown("---")
            
            # Tomorrow's Prediction
            st.subheader("🔮 Tomorrow's Price Prediction")
            
            current = float(results['current_price'])
            predicted = float(results['tomorrow_pred'])
            pred_change = ((predicted - current) / current) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📍 Today's Close", f"{currency}{current:,.2f}")
            with col2:
                st.metric(
                    "🎯 Predicted Tomorrow",
                    f"{currency}{predicted:,.2f}",
                    f"{pred_change:+.2f}%"
                )
            with col3:
                st.metric("📊 Confidence", f"{accuracy_pct:.1f}%")
            
            st.markdown("---")
            
            # Recommendation
            st.subheader("💡 Trading Recommendation")
            
            recommendation, status, icon = get_recommendation(pred_change, current_rsi)
            
            if status == "success":
                st.success(f"### {icon} {recommendation}")
                st.markdown("The model predicts an **upward movement**. Consider buying if other factors align.")
            elif status == "error":
                st.error(f"### {icon} {recommendation}")
                st.markdown("The model predicts a **downward movement**. Consider selling or waiting.")
            else:
                st.info(f"### {icon} {recommendation}")
                st.markdown("The model predicts **sideways movement**. Consider holding your position.")
            
            st.markdown("---")
            
            # Actual vs Predicted Chart
            st.subheader("📉 Actual vs Predicted Prices")
            comparison_df = pd.DataFrame({
                'Actual Price': results['y_test'],
                'Predicted Price': results['y_pred']
            })
            st.line_chart(comparison_df, use_container_width=True)
            
            st.markdown("---")
            
            # Feature Importance
            st.subheader("🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': results['features'],
                'Importance': results['model'].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            st.bar_chart(importance_df.set_index('Feature'), use_container_width=True)
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Disclaimer**: This app is for **educational purposes only**. 
        Stock market predictions are inherently uncertain. Do not make investment decisions 
        based solely on this app. Always consult a qualified financial advisor.
        """)
    
    else:
        # Welcome Screen
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📌 How to Use
            1. **Select Market** - Choose Indian (NSE) or US market
            2. **Pick a Stock** - Select from popular stocks
            3. **Set Date Range** - Choose your analysis period
            4. **Click Analyze** - Get predictions and insights!
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Features
            - 📊 **Price Visualization** - Interactive charts
            - 📈 **Technical Indicators** - RSI, MACD, Bollinger Bands
            - 🤖 **ML Prediction** - Random Forest model
            - 🎯 **Recommendations** - Buy/Sell/Hold signals
            """)
        
        st.markdown("---")
        
        st.info("👈 **Configure settings in the sidebar and click 'Analyze & Predict' to get started!**")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Built with ❤️ by <strong>Krishna Jha</strong> | IILM University | © 2024</p>",
        unsafe_allow_html=True
    )


# Run the app
if __name__ == "__main__":
    main()
