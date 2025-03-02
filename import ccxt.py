from inspect import isframe
from urllib import request
import ccxt # type: ignore
import pandas as pd # type: ignore
import numpy as np
from datetime import datetime
import time
import streamlit as st # type: ignore # type: ignore
from typing import Dict, List, Self
import plotly.graph_objects as go # type: ignore
from decimal import Decimal

import symbol # type: ignore

class CryptoTeamTrading:
    def __init__(self):
        self.exchanges = {
            'binance': ccxt.binance(),
            'coinbase': ccxt.coinbasepro(),
            'kraken': ccxt.kraken()
        }
        self.commission_rate = 0.002  # 0.2% commission for the platform
        self.user_balance = {}
        self.active_orders = []
        
    def initialize_exchange(self, exchange_id: str, api_key: str, secret: str):
        """Initialize exchange with API credentials"""
        if exchange_id in self.exchanges:
            self.exchanges[exchange_id].apiKey = api_key
            self.exchanges[exchange_id].secret = secret
            
    def get_market_data(self, exchange_id: str, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Get historical market data"""
        exchange = self.exchanges[exchange_id]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def place_order(self, exchange_id: str, symbol: str, order_type: str, side: str, amount: float, price: float = None):
        """Place trading order"""
        exchange = self.exchanges[exchange_id]
        
        # Calculate platform commission
        commission = amount * self.commission_rate
        adjusted_amount = amount - commission
        
        try:
            if order_type == 'market':
                order = exchange.create_market_order(symbol, side, adjusted_amount)
            elif order_type == 'limit':
                order = exchange.create_limit_order(symbol, side, adjusted_amount, price)
            
            self.active_orders.append(order)
            return order
        except Exception as e:
            return f"Error placing order: {str(e)}"
    
    def get_wallet_balance(self, exchange_id: str) -> Dict:
        """Get wallet balance"""
        exchange = self.exchanges[exchange_id]
        balance = exchange.fetch_balance()
        return balance['total']
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

class CryptoTeamTradingUI:
    def __init__(self):
        self.trading_bot = CryptoTeamTrading()
        self.selected_exchange = "binance"  # valor por defecto
        self.selected_symbol = "BTC/USDT"   # valor por defecto
        
    def update_charts(self):
        placeholder = st.empty()
        while True:
            with placeholder.container():
                # Create two columns for better layout
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Get latest data
                    df = self.trading_bot.get_market_data(
                        self.selected_exchange, 
                        self.selected_symbol, 
                        timeframe='1h'  # timeframe por defecto
                    )
                    df = self.trading_bot.calculate_technical_indicators(df)
                    
                    # Enhanced price chart with volume
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="OHLC"
                    ))
                    
                    # Add volume bars
                    colors = ['red' if row['open'] > row['close'] else 'green' for index, row in df.iterrows()]
                    fig.add_trace(go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name="Volume",
                        marker_color=colors,
                        yaxis="y2"
                    ))
                    
                    # Add moving averages with better styling
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_20'], 
                                          name="SMA 20", line=dict(color='blue', width=1)))
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], 
                                          name="EMA 20", line=dict(color='orange', width=1)))
                    
                    # Improve layout
                    fig.update_layout(
                        title=f"{self.selected_symbol} Price Chart",
                        yaxis_title="Price",
                        yaxis2=dict(
                            title="Volume",
                            overlaying="y",
                            side="right"
                        ),
                        template="plotly_dark",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                with chart_col2:
                    # Enhanced RSI chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=df['timestamp'], 
                        y=df['RSI'], 
                        name="RSI",
                        line=dict(color='purple', width=2)
                    ))
                    
                    # Add RSI zones
                    fig_rsi.add_hrect(
                        y0=70, y1=100,
                        fillcolor="red", opacity=0.2,
                        line_width=0
                    )
                    fig_rsi.add_hrect(
                        y0=0, y1=30,
                        fillcolor="green", opacity=0.2,
                        line_width=0
                    )
                    
                    fig_rsi.update_layout(
                        title="RSI Indicator",
                        yaxis_title="RSI Value",
                        template="plotly_dark",
                        height=500
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Add market statistics
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("24h Change", f"{df['close'].pct_change(24).iloc[-1]:.2%}")
                with stats_col2:
                    st.metric("Volume", f"${df['volume'].iloc[-1]:,.2f}")
                with stats_col3:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                
            time.sleep(60)  # Update every minute

    def run(self):
        st.title("CryptoTeam Trading Platform")
        
        # Sidebar
        st.sidebar.header("Configuration")
        self.selected_exchange = st.sidebar.selectbox("Select Exchange", ["binance", "coinbase", "kraken"])
        self.selected_symbol = st.sidebar.text_input("Trading Pair", "BTC/USDT")
        
        # API Configuration
        with st.sidebar.expander("API Configuration"):
            api_key = st.text_input("API Key", type="password")
            api_secret = st.text_input("API Secret", type="password")
            if st.button("Connect"):
                self.trading_bot.initialize_exchange(self.selected_exchange, api_key, api_secret)
                st.success("Exchange connected successfully!")
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Data")
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
            
            if st.button("Fetch Data"):
                df = self.trading_bot.get_market_data(self.selected_exchange, self.selected_symbol, timeframe)
                df = self.trading_bot.calculate_technical_indicators(df)
                
                # Create price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="OHLC"
                ))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_20'], name="SMA 20"))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], name="EMA 20"))
                st.plotly_chart(fig)
                
                # Show RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], name="RSI"))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(fig_rsi)
        
        with col2:
            st.subheader("Trading")
            order_type = st.selectbox("Order Type", ["market", "limit"])
            side = st.selectbox("Side", ["buy", "sell"])
            amount = st.number_input("Amount", min_value=0.0)
            
            if order_type == "limit":
                price = st.number_input("Price", min_value=0.0)
            else:
                price = None
                
            if st.button("Place Order"):
                order = self.trading_bot.place_order(self.selected_exchange, self.selected_symbol, order_type, side, amount, price)
                st.write(order)
            
            # Show wallet balance
            if st.button("Update Balance"):
                balance = self.trading_bot.get_wallet_balance(self.selected_exchange)
                st.write("Wallet Balance:", balance)

        # Iniciar actualización en tiempo real
        import threading
        threading.Thread(target=self.update_charts, daemon=True).start()

    def get_crypto_news(self):
        """Fetch latest crypto news using CryptoCompare News API"""
        try:
            # Create news section
            st.header("📰 Crypto News")
            
            news_col1, news_col2 = st.columns([2,1])
            
            with news_col1:
                st.subheader("Latest Headlines")
                
                # Fetch news from CryptoCompare
                news_url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
                response = request.get(news_url)
                news_data = response.json()['Data'][:6]  # Get latest 6 news items
                
                for article in news_data:
                    with st.expander(article['title']):
                        st.write(article['body'])
                        st.write(f"Source: {article['source']} | {article['published_on']}")
                        if article['imageurl']:
                            st.image(article['imageurl'])
                        st.markdown(f"[Read more]({article['url']})")
            
            with news_col2:
                st.subheader("Market Sentiment")
                # Add fear & greed index visualization
                fear_greed_url = "https://api.alternative.me/fng/"
                fg_response = request.get(fear_greed_url)
                fg_data = fg_response.json()['data'][0]
                
                fg_value = int(fg_data['value'])
                fg_label = fg_data['value_classification']
                
                st.metric("Fear & Greed Index", fg_value)
                st.progress(fg_value/100)
                st.caption(f"Market Sentiment: {fg_label}")
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return None

    def add_trading_features(self):
        """Add advanced trading features to the interface"""
        try:
            st.sidebar.header("Trading Tools")
            
            # Trading pair selector
            selected_pair = st.sidebar.selectbox(
                "Select Trading Pair",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
            )
            
            # Order type selector
            order_type = st.sidebar.radio(
                "Order Type",
                ["Market", "Limit"]
            )
            
            # Amount input
            amount = st.sidebar.number_input(
                "Amount",
                min_value=0.0,
                step=0.01
            )
            
            # Price input for limit orders
            price = None
            if order_type == "Limit":
                price = st.sidebar.number_input(
                    "Limit Price",
                    min_value=0.0,
                    step=0.01
                )
            
            col1, col2 = st.sidebar.columns(2)
            
            # Buy/Sell buttons
            with col1:
                if st.button("Buy", type="primary"):
                    if order_type == "Market":
                        self.trading_bot.place_order(self.selected_exchange, selected_pair, "market", "buy", amount)
                    else:
                        self.trading_bot.place_order(self.selected_exchange, selected_pair, "limit", "buy", amount, price)
                    
            with col2:
                if st.button("Sell", type="secondary"):
                    if order_type == "Market":
                        self.trading_bot.place_order(self.selected_exchange, selected_pair, "market", "sell", amount)
                    else:
                        self.trading_bot.place_order(self.selected_exchange, selected_pair, "limit", "sell", amount, price)
            
            # Portfolio Overview
            st.sidebar.header("Portfolio Overview")
            if st.sidebar.button("Refresh Balance"):
                balance = self.trading_bot.get_wallet_balance(self.selected_exchange)
                if balance and 'total' in balance:
                    for asset, value in balance['total'].items():
                        if value > 0:
                            st.sidebar.metric(asset, f"{value:.8f}")
        except Exception as e:
            st.error(f"Error in trading features: {str(e)}")

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['20dSTD'] = df['close'].rolling(window=20).std()
            df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
            df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)
            
            return df
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df

    def run_app(self):
        """Main method to run the Streamlit application"""
        try:
            st.set_page_config(layout="wide", page_title="Crypto Trading Dashboard")
            
            st.title("🚀 Crypto Trading Dashboard")
            
            # Initialize main components
            self.add_trading_features()
            self.get_crypto_news()
            
            # Add any additional features or customization here
            st.sidebar.markdown("---")
            st.sidebar.caption("Developed with ❤️ by Your Team")
        except Exception as e:
            st.error(f"Error running app: {str(e)}")

    def add_trading_courses(self):
        """Add trading courses section and affiliate program"""
        try:
            st.header("📚 Trading Courses for Beginners")
            
            # Course offerings
            courses = {
                "Crypto Trading Basics": {
                    "price": 99,
                    "description": "Learn the fundamentals of cryptocurrency trading"
                },
                "Technical Analysis 101": {
                    "price": 149, 
                    "description": "Master chart patterns and technical indicators"
                },
                "Risk Management Essentials": {
                    "price": 79,
                    "description": "Learn proper risk management techniques"
                }
            }

            # Display courses
            cols = st.columns(len(courses))
            for col, (course_name, details) in zip(cols, courses.items()):
                with col:
                    st.subheader(course_name)
                    st.write(details["description"])
                    st.write(f"Price: ${details['price']}")
                    if st.button(f"Enroll in {course_name}"):
                        # Payment processing would go here
                        commission = details["price"] * 0.2  # 20% commission
                        st.success(f"Thank you for enrolling! Processing payment...")
                        st.info("Commission will be sent to: USDT TP5vLL3MnVcb5yZ9i7mBQe3q9BVCrfiFUd")

            # Affiliate Program Section
            st.header("🤝 Become an Affiliate Partner")
            st.write("""
            Join our affiliate program and earn commissions by referring new traders!
            - 20% commission on course sales
            - 10% commission on referred user trading fees
            - Instant payments in USDT
            """)
            
            if st.button("Apply as Affiliate"):
                st.success("Thank you for your interest! Please fill out the form below:")
                with st.form("affiliate_form"):
                    st.text_input("Name")
                    st.text_input("Email")
                    st.text_input("Social Media Presence")
                    st.text_area("Why do you want to become an affiliate?")
                    submitted = st.form_submit_button("Submit Application")
                    if submitted:
                        st.success("Application received! We'll contact you soon.")

        except Exception as e:
            st.error(f"Error in trading courses section: {str(e)}")

    # Add real-time crypto charts section
    def add_crypto_charts(self):
        """Add real-time cryptocurrency charts similar to Binance style"""
        try:
            st.header("📈 Real-time Cryptocurrency Charts")
            
            # Create tabs for different timeframes
            timeframes = st.selectbox("Select Timeframe", 
                ["1m", "5m", "15m", "1h", "4h", "1d"])
            
            # Allow multiple crypto pairs selection
            pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
            selected_pairs = st.multiselect("Select Trading Pairs", pairs, default=["BTC/USDT"])
            
            # Create columns for charts
            cols = st.columns(len(selected_pairs))
            
            for idx, pair in enumerate(selected_pairs):
                with cols[idx]:
                    st.subheader(pair)
                    
                    # Get OHLCV data from exchange
                    try:
                        ohlcv = self.trading_bot.exchanges[self.selected_exchange].fetch_ohlcv(
                            pair, 
                            timeframes,
                            limit=100
                        )
                        
                        # Convert to pandas DataFrame
                        df = pd.DataFrame(
                            ohlcv, 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Create candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=df['timestamp'],
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close']
                        )])
                        
                        # Customize chart layout
                        fig.update_layout(
                            xaxis_rangeslider_visible=False,
                            height=400,
                            template="plotly_dark",
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add trading volume
                        volume_fig = go.Figure(data=[go.Bar(
                            x=df['timestamp'],
                            y=df['volume'],
                            name='Volume'
                        )])
                        volume_fig.update_layout(
                            height=100,
                            template="plotly_dark",
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False
                        )
                        st.plotly_chart(volume_fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error fetching data for {pair}: {str(e)}")
            
            # Auto-refresh every minute
            time.sleep(60)
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error in crypto charts section: {str(e)}")
            st.markdown("---")
            st.markdown("### Support")
            st.write("For support, please contact: bitcbase@gmail.com")
            st.markdown("### Donations")
            st.write("If you would like to support this project, you can send donations to:")
            st.code("bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh", language="text")
            # Technical Indicators Section
            st.markdown("### Technical Indicators")
            
            # Calculate and display technical indicators
            if not df.empty:
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                # Display indicators
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(df['RSI'])
                with col2:
                    st.line_chart(df[['MACD', 'Signal_Line']])
            
            # Price Alerts
            st.markdown("### Price Alerts")
            alert_price = st.number_input("Set Price Alert", value=float(df['close'].iloc[-1]))
            alert_condition = st.selectbox("Alert Condition", ["Above", "Below"])
            if st.button("Set Alert"):
                if alert_condition == "Above" and float(df['close'].iloc[-1]) > alert_price:
                    st.warning(f"Price is above {alert_price}!")
                elif alert_condition == "Below" and float(df['close'].iloc[-1]) < alert_price:
                    st.warning(f"Price is below {alert_price}!")
            
            # Trading Pairs
            st.markdown("### Available Trading Pairs")
            try:
                exchange = ccxt.binance()
                markets = exchange.load_markets()
                trading_pairs = [market for market in markets.keys() if market.endswith('/USDT')]
                st.multiselect("Select Trading Pairs", trading_pairs)
            except:
                st.error("Could not load trading pairs")
            
            # Automated Trading Strategy
            st.markdown("### Automated Trading")
            strategy = st.selectbox("Select Strategy", ["MACD Crossover", "RSI Oversold/Overbought"])
            
            if strategy == "MACD Crossover":
                if not df.empty and len(df) > 26:
                    # Simple MACD strategy
                    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
                        st.success("Buy Signal!")
                    elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]:
                        st.success("Sell Signal!")
            
            # Risk Management
            st.markdown("### Risk Management")
            position_size = st.slider("Position Size (%)", 1, 100, 10)
            stop_loss = st.slider("Stop Loss (%)", 1, 50, 5)
            take_profit = st.slider("Take Profit (%)", 1, 100, 15)
            
            # Calculate position sizes and risk
            if st.button("Calculate Risk"):
                current_price = float(df['close'].iloc[-1])
                stop_loss_price = current_price * (1 - stop_loss/100)
                take_profit_price = current_price * (1 + take_profit/100)
                st.write(f"Stop Loss Price: {stop_loss_price:.2f}")
                st.write(f"Take Profit Price: {take_profit_price:.2f}")
                st.write(f"Risk/Reward Ratio: {(take_profit)/(stop_loss):.2f}")
            
            # Trade History
            st.markdown("### Trade History")
            # This would typically connect to a database
            # Here's a sample display
            trade_history = pd.DataFrame({
                'Date': [pd.Timestamp.now()],
                'Type': ['Buy'],
                'Price': [float(df['close'].iloc[-1])],
                'Amount': [1.0],
                'Status': ['Open']
            })
            st.dataframe(trade_history)
            # User Authentication
            st.markdown("### User Authentication")
            
            # Create login form
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

            # Simple user session management
            if 'authenticated' not in st.session_state:
                st.session_state.authenticated = False
                
            if submitted:
                # Here you would typically verify against a database
                # This is a simple example - replace with proper authentication
                if username and password:  # Add your authentication logic here
                    st.session_state.authenticated = True
                    st.success("Successfully logged in!")
                else:
                    st.error("Invalid credentials")

            # API Key Management with encryptioncryptography.hazmat.primitives.kdf.pbkdf2
            if st.session_state.authenticated:
                st.markdown("### API Key Management")
                
                # Use Fernet encryption for API keys
                from cryptography.fernet import Fernet # type: ignore
                import base64
                from cryptography.hazmat.primitives import hashes # type: ignore
                from  import PBKDF2HMAC # type: ignore
                
                def generate_key(password):
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=b'salt_',  # Use a secure random salt in production
                        iterations=100000,
                    )
                    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                    return key

                def encrypt_api_key(api_key, encryption_key):
                    f = Fernet(encryption_key)
                    return f.encrypt(api_key.encode()).decode()

                def decrypt_api_key(encrypted_api_key, encryption_key):
                    f = Fernet(encryption_key)
                    return f.decrypt(encrypted_api_key.encode()).decode()

                # API Key input
                api_key = st.text_input("Enter API Key", type="password")
                api_secret = st.text_input("Enter API Secret", type="password")

                if api_key and api_secret:
                    encryption_key = generate_key(password)  # Use user's password as encryption key
                    encrypted_api_key = encrypt_api_key(api_key, encryption_key)
                    encrypted_api_secret = encrypt_api_key(api_secret, encryption_key)
                    
                    # Store encrypted keys securely (e.g., in a database)
                    # This is just an example using session state
                    st.session_state.encrypted_api_key = encrypted_api_key
                    st.session_state.encrypted_api_secret = encrypted_api_secret

                # Trading Limits
                st.markdown("### Trading Limits")
                
                # Daily trading limits
                if 'daily_trades' not in st.session_state:
                    st.session_state.daily_trades = 0
                    
                max_daily_trades = st.slider("Max Daily Trades", 1, 50, 10)
                remaining_trades = max_daily_trades - st.session_state.daily_trades
                
                st.write(f"Remaining trades today: {remaining_trades}")
                
                # Trading volume limits
                max_trade_volume = st.number_input("Max Trade Volume (USDT)", min_value=0.0, value=1000.0)
                
                # Check limits before executing trades
                def check_trading_limits(trade_volume):
                    if st.session_state.daily_trades >= max_daily_trades:
                        st.error("Daily trading limit reached!")
                        return False
                    if trade_volume > max_trade_volume:
                        st.error("Trade volume exceeds limit!")
                        return False
                    return True
                
                # Reset daily trades at midnight
                from datetime import datetime
                if 'last_reset' not in st.session_state:
                    st.session_state.last_reset = datetime.now().date()
                
                current_date = datetime.now().date()
                if current_date > st.session_state.last_reset:
                    st.session_state.daily_trades = 0
                    st.session_state.last_reset = current_date

if __name__ == "__main__":
    app = CryptoTeamTradingUI()
    app.run_app()
