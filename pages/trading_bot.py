"""
Trading Bot Interface

This page provides a UI for configuring and running the trading bot with Binance API.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import json
import logging

# Import our trading bot modules
from trading_bot.binance_api import test_binance_connection, BinanceAPI
from trading_bot.strategy_bot import StrategyBot

# Set up page config
st.set_page_config(page_title="Trading Bot", page_icon="📈", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trading_bot_ui")

def create_equity_chart(equity_data, price_data=None, show_real_time_price=True):
    """
    Create an equity curve plot with real-time price line.
    
    Args:
        equity_data: List of equity data points [{"timestamp": str, "equity": float}]
        price_data: List of price data points [{"timestamp": str, "price": float}]
        show_real_time_price: Whether to show current spot price
    """
    fig = go.Figure()
    
    # Ensure we have data before trying to plot
    if not equity_data or len(equity_data) == 0:
        # If no data, create an empty figure with proper axes
        fig.update_layout(
            title='Portfolio Value Over Time (No Data)',
            xaxis_title='Time',
            yaxis_title='Portfolio Value (USDT)',
            template='plotly_white',
            height=500
        )
        return fig
    
    # Extract timestamps and equity values
    timestamps = [pd.to_datetime(entry['timestamp']) for entry in equity_data if 'timestamp' in entry]
    equity_values = [entry['equity'] for entry in equity_data if 'equity' in entry]
    
    # Ensure we have valid data after filtering
    if not timestamps or not equity_values or len(timestamps) != len(equity_values):
        # If filtered data is empty or inconsistent, return empty chart
        fig.update_layout(
            title='Portfolio Value Over Time (Invalid Data)',
            xaxis_title='Time',
            yaxis_title='Portfolio Value (USDT)',
            template='plotly_white',
            height=500
        )
        return fig
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    # Add price if available
    if price_data and len(price_data) > 0:
        # Extract price timestamps and values (with validation)
        price_timestamps = [pd.to_datetime(entry['timestamp']) for entry in price_data if 'timestamp' in entry]
        price_values = [entry['price'] for entry in price_data if 'price' in entry]
        
        if price_timestamps and price_values and len(price_timestamps) == len(price_values):
            # Add price curve on secondary y-axis
            fig.add_trace(go.Scatter(
                x=price_timestamps,
                y=price_values,
                mode='lines',
                name='BTC Price',
                yaxis='y2',
                line=dict(color='orange', width=1.5)
            ))
            
            # If show_real_time_price is enabled and we have access to the latest price
            if show_real_time_price:
                try:
                    # Import the function locally to avoid circular imports
                    from utils.binance_data import fetch_current_price
                    current_price = fetch_current_price()
                    
                    # Add a horizontal line for the current price
                    fig.add_shape(
                        type="line",
                        x0=min(timestamps),
                        y0=current_price,
                        x1=max(timestamps),
                        y1=current_price,
                        yref="y2",
                        line=dict(
                            color="red",
                            width=1,
                            dash="dash",
                        ),
                    )
                    
                    # Add an annotation for the current price
                    fig.add_annotation(
                        x=max(timestamps),
                        y=current_price,
                        xref="x",
                        yref="y2",
                        text=f"Current: ${current_price:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        ax=50,
                        ay=0,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        font=dict(color="red")
                    )
                except Exception as e:
                    # Silently handle any errors fetching the current price
                    print(f"Error adding real-time price: {e}")
                    pass
    
    # Configure the layout for dual y-axis
    fig.update_layout(
        yaxis=dict(title='Portfolio Value (USDT)'),
        yaxis2=dict(
            title='BTC Price (USDT)',
            overlaying='y',
            side='right'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Time',
        yaxis_title='Portfolio Value (USDT)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=500
    )
    
    return fig

def create_trade_history_table(trade_history):
    """Create a formatted trade history table."""
    if not trade_history:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(trade_history)
    
    # Reformat the timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add profit percentage if it's a sell
    if 'profit_loss' in df.columns:
        df['profit_pct'] = df.apply(
            lambda row: f"{(row['profit_loss'] / (row['amount_usd'] - row['profit_loss']) * 100):.2f}%" 
            if row['type'] == 'SELL' and row['profit_loss'] != 0 else "", 
            axis=1
        )
    
    # Reorder and select columns
    columns = ['timestamp', 'type', 'price', 'amount_usd', 'asset_amount', 'fee']
    if 'profit_loss' in df.columns:
        columns.extend(['profit_loss', 'profit_pct'])
    
    return df[columns]

def bot_worker(api_key, api_secret, use_testnet, config, stop_event, status_data):
    """Worker function to run the trading bot in a separate thread."""
    try:
        # Initialize the bot
        bot = StrategyBot(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet,
            initial_capital=config['initial_capital'],
            trading_pair=config['trading_pair'],
            timeframe=config['timeframe'],
            strategy_type=config['strategy_type'],
            strategy_params=config.get('strategy_params', {})
        )
        
        # Update status
        status_data['status'] = 'running'
        status_data['start_time'] = datetime.now().isoformat()
        status_data['message'] = 'Bot initialized and running'
        
        # Calculate the end time
        end_time = time.time() + (config['duration_minutes'] * 60)
        iteration = 0
        
        # Run the bot until the stop event is set or the duration is reached
        while not stop_event.is_set() and time.time() < end_time:
            iteration += 1
            
            # Run a single iteration
            logger.info(f"Running iteration {iteration}")
            success = bot.run_simulation_iteration()
            
            # Update status data
            status_data['last_update'] = datetime.now().isoformat()
            status_data['iterations'] = iteration
            status_data['portfolio_value'] = bot.calculate_portfolio_value()
            status_data['cash_balance'] = bot.cash_balance
            status_data['positions'] = len(bot.positions)
            status_data['trade_count'] = bot.total_trades
            
            # Get the full performance summary
            status_data['performance'] = bot.get_performance_summary()
            
            # Store trade history and equity curve
            status_data['trade_history'] = bot.trade_history
            status_data['equity_curve'] = bot.equity_curve
            
            # Check if we're stopping
            if stop_event.is_set():
                break
                
            # Wait for the next iteration (5 seconds)
            interval_seconds = config.get('interval_seconds', 5)
            for _ in range(interval_seconds):
                if stop_event.is_set():
                    break
                time.sleep(1)
        
        # Save final results
        bot.save_trade_history()
        bot.save_equity_curve()
        
        # Update status
        status_data['status'] = 'completed'
        status_data['end_time'] = datetime.now().isoformat()
        status_data['message'] = 'Bot completed successfully'
        
    except Exception as e:
        # Update status with error
        status_data['status'] = 'error'
        status_data['end_time'] = datetime.now().isoformat()
        status_data['message'] = f'Error: {str(e)}'
        
        # Log the error
        logger.error(f"Error in bot worker: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function to run the trading bot UI."""
    
    st.title("Cryptocurrency Trading Bot")
    st.write("Run trading strategies with real-time market data from Binance")
    
    # Check if a bot is already running
    if 'bot_thread' not in st.session_state:
        st.session_state.bot_thread = None
        st.session_state.stop_event = threading.Event()
        st.session_state.bot_status = {
            'status': 'idle',
            'message': 'Bot not running',
            'trade_history': [],
            'equity_curve': []
        }
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Bot Configuration", "Live Status", "Results & Analysis"])
    
    with tab1:
        st.header("Configure Trading Bot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Configuration")
            
            # Environment selection (testnet or live)
            use_testnet = st.checkbox("Use Binance Testnet (Recommended for testing)", value=True)
            
            # API credentials
            api_key = st.text_input("Binance API Key", 
                                   value="iWpFe1DXXxkVogNVMzCjjMpKrohrGgWbD4OGV4jFsmvs6EyKAS7lJ3hwttxtSeUD" if use_testnet else "")
            
            api_secret = st.text_input("Binance API Secret", type="password",
                                      value="UNBjYHOgWcT35gPKKBGMilczJOumARTO271PWlbVtEolOUEgfeS5bg09ZbT8n04B" if use_testnet else "")
            
            # Test connection button
            if st.button("Test Connection"):
                if not api_key or not api_secret:
                    st.error("Please enter API key and secret")
                else:
                    with st.spinner("Testing connection to Binance API..."):
                        connection_ok = test_binance_connection(api_key, api_secret, use_testnet)
                        
                        if connection_ok:
                            st.success("Connection successful! You can now configure and run the bot.")
                        else:
                            st.error("Failed to connect to Binance API. Please check your credentials.")
        
        with col2:
            st.subheader("Trading Parameters")
            
            # Trading parameters
            trading_pair = st.selectbox("Trading Pair", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"], index=0)
            
            timeframe = st.selectbox("Timeframe", 
                                    ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
                                    index=3)  # Default to 15m
            
            initial_capital = st.number_input("Initial Capital (USDT)", 
                                             min_value=10.0, max_value=10000.0, value=1000.0, step=100.0,
                                             format="%.2f")
            
            duration_minutes = st.number_input("Bot Run Duration (minutes)",
                                              min_value=5, max_value=1440, value=60, step=5)
            
            interval_seconds = st.number_input("Check Interval (seconds)",
                                             min_value=5, max_value=300, value=60, step=5)
        
        st.subheader("V5 Strategy")
        st.info("This interface exclusively uses the V5 strategy with enhanced risk management and multi-bot capabilities.")
        
        # Only V5 strategy is supported
        strategy_type = "v5_strategy"
        
        # V5 Strategy parameters
        st.subheader("Strategy Parameters")
        
        # Use a form for better organization
        with st.form("v5_strategy_form"):
            # Basic parameters
            st.subheader("Basic Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                investment_pct = st.number_input("Investment Amount (%)", 
                                               min_value=1.0, max_value=100.0, value=50.0, step=1.0,
                                               help="Percentage of available capital to invest on each buy",
                                               format="%.2f")
            
            with col2:
                price_drop_threshold = st.number_input("Price Drop Threshold (%)", 
                                                     min_value=0.5, max_value=10.0, value=2.0, step=0.1,
                                                     help="Minimum price decrease required to trigger a buy",
                                                     format="%.2f")
            
            with col3:
                profit_threshold = st.number_input("Profit Threshold (%)", 
                                                 min_value=0.5, max_value=10.0, value=4.0, step=0.1,
                                                 help="Target profit percentage that triggers selling",
                                                 format="%.2f")
        
            # Risk Management
            st.subheader("Risk Management")
            col1, col2 = st.columns(2)
            
            with col1:
                stop_loss_fixed_pct = st.number_input("Fixed Stop-Loss (%)", 
                                                    min_value=5.0, max_value=50.0, value=50.0, step=1.0,
                                                    help="Fixed stop-loss percentage",
                                                    format="%.2f")
                
                trailing_stop_pct = st.number_input("Trailing Stop (%)", 
                                                  min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                                                  help="Trailing stop percentage below highest price",
                                                  format="%.2f")
                
            with col2:
                max_position_duration = st.number_input("Max Position Duration (days)", 
                                                      min_value=7, max_value=365, value=180, step=1,
                                                      help="Maximum days to hold a position")
                
                min_trade_size = st.number_input("Minimum Trade Size (USDT)", 
                                               min_value=10.0, max_value=200.0, value=10.0, step=1.0,
                                               help="Minimum order size in USDT",
                                               format="%.2f")
                
                trading_fee_pct = st.number_input("Trading Fee (%)", 
                                               min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                                               help="Trading fee percentage",
                                               format="%.2f")
            
            # Advanced Settings
            st.subheader("Advanced Settings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                partial_profit_taking = st.checkbox("Partial Profit Taking", 
                                                  value=False,
                                                  help="Enable taking partial profits")
                
                profit_taking_percentage = st.number_input("Profit Taking Percentage (%)", 
                                                         min_value=10.0, max_value=100.0, value=100.0, step=5.0,
                                                         help="Percentage of position to sell on profit",
                                                         format="%.2f")
                
            with col2:
                use_trailing_stop_main = st.checkbox("Use Trailing Stop", 
                                                   value=True,
                                                   help="Enable trailing stop feature")
                
                trailing_stop_activation_pct = st.number_input("Trailing Stop Activation (%)", 
                                                             min_value=0.5, max_value=50.0, value=50.0, step=0.5,
                                                             help="Profit percentage to activate trailing stop",
                                                             format="%.2f")
                
            with col3:
                number_of_bots = st.number_input("Number of Bots", 
                                               min_value=1, max_value=10, value=6, step=1,
                                               help="Number of independent trading bots to run")
                
                uniform_deployment_days = st.number_input("Uniform Deployment Days", 
                                                        min_value=0, max_value=365, value=90, step=1,
                                                        help="Days to spread bot deployment (0 = all at once)")
                
                # Submit button for the form
                submit_button = st.form_submit_button("Apply Settings")
        
        # Start/Stop Bot buttons
        st.subheader("Bot Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.bot_thread is None or not st.session_state.bot_thread.is_alive():
                if st.button("Start Trading Bot"):
                    # Validate inputs
                    if not api_key or not api_secret:
                        st.error("Please enter API key and secret")
                    else:
                        # Reset stop event
                        st.session_state.stop_event.clear()
                        
                        # Prepare configuration
                        config = {
                            'initial_capital': initial_capital,
                            'trading_pair': trading_pair,
                            'timeframe': timeframe,
                            'strategy_type': strategy_type,
                            'duration_minutes': duration_minutes,
                            'interval_seconds': interval_seconds,
                            'strategy_params': {
                                'investment_pct': investment_pct,
                                'price_drop_threshold': price_drop_threshold,
                                'profit_threshold': profit_threshold,
                            }
                        }
                        
                        # Add V5 strategy-specific parameters
                        config['strategy_params'].update({
                            'stop_loss_fixed_pct': stop_loss_fixed_pct,
                            'trailing_stop_pct': trailing_stop_pct,
                            'max_position_duration': max_position_duration,
                            'min_trade_size': min_trade_size,
                            'trading_fee_pct': trading_fee_pct,
                            'partial_profit_taking': partial_profit_taking,
                            'profit_taking_percentage': profit_taking_percentage,
                            'use_trailing_stop_main': use_trailing_stop_main,
                            'trailing_stop_activation_pct': trailing_stop_activation_pct, 
                            'number_of_bots': number_of_bots,
                            'uniform_deployment_days': uniform_deployment_days
                        })
                        
                        # Initialize bot status
                        st.session_state.bot_status = {
                            'status': 'starting',
                            'message': 'Bot is starting...',
                            'config': config,
                            'trade_history': [],
                            'equity_curve': []
                        }
                        
                        # Start the bot in a new thread
                        st.session_state.bot_thread = threading.Thread(
                            target=bot_worker,
                            args=(api_key, api_secret, use_testnet, config, 
                                  st.session_state.stop_event, st.session_state.bot_status)
                        )
                        st.session_state.bot_thread.start()
                        
                        st.success("Bot started! Go to the Live Status tab to monitor performance.")
        
        with col2:
            if st.session_state.bot_thread is not None and st.session_state.bot_thread.is_alive():
                if st.button("Stop Trading Bot"):
                    st.session_state.stop_event.set()
                    st.success("Stopping the bot... This may take a few seconds.")
    
    with tab2:
        st.header("Live Trading Status")
        
        # Refresh button
        if st.button("Refresh Status"):
            st.rerun()
        
        # Show bot status
        status = st.session_state.bot_status.get('status', 'idle')
        message = st.session_state.bot_status.get('message', '')
        
        if status == 'idle':
            st.info("Bot is not running. Configure and start the bot from the Configuration tab.")
        elif status == 'starting':
            st.info("Bot is starting... Please wait.")
        elif status == 'running':
            st.success("Bot is running")
            
            # Display current stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Value", 
                         f"${st.session_state.bot_status.get('portfolio_value', 0):.2f}")
            
            with col2:
                st.metric("Cash Balance", 
                         f"${st.session_state.bot_status.get('cash_balance', 0):.2f}")
            
            with col3:
                st.metric("Open Positions", 
                         st.session_state.bot_status.get('positions', 0))
            
            with col4:
                st.metric("Total Trades", 
                         st.session_state.bot_status.get('trade_count', 0))
            
            # Display performance metrics if available
            if 'performance' in st.session_state.bot_status:
                perf = st.session_state.bot_status['performance']
                
                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Return", 
                             f"${perf.get('total_return', 0):.2f} ({perf.get('pct_return', 0):.2f}%)")
                
                with col2:
                    st.metric("Max Drawdown", 
                             f"{perf.get('max_drawdown', 0):.2f}%")
                
                with col3:
                    st.metric("Win Rate", 
                             f"{perf.get('win_rate', 0):.2f}%")
            
            # Display equity curve
            if 'equity_curve' in st.session_state.bot_status and st.session_state.bot_status['equity_curve']:
                st.subheader("Equity Curve")
                # Get price data from the bot status or fetch current market data
                try:
                    # Try to get market data with actual price data
                    from utils.binance_data import get_market_data
                    # Use trading pair from bot status if available
                    trading_pair = (st.session_state.bot_status.get('config', {}).get('trading_pair', 'BTCUSDT') 
                                   if 'config' in st.session_state.bot_status else 'BTCUSDT')
                    timeframe = (st.session_state.bot_status.get('config', {}).get('timeframe', '15m')
                               if 'config' in st.session_state.bot_status else '15m')
                    
                    market_data = get_market_data(
                        symbol=trading_pair,
                        interval=timeframe
                    )
                    # Create price data array in the format expected by create_equity_chart
                    price_data = []
                    if market_data and 'klines' in market_data:
                        for kline in market_data['klines'][-30:]:  # Get last 30 candles
                            price_data.append({
                                'timestamp': kline[0],  # Open time
                                'price': float(kline[4])  # Close price
                            })
                    
                    fig = create_equity_chart(
                        st.session_state.bot_status['equity_curve'],
                        price_data if price_data else st.session_state.bot_status['equity_curve'],
                        show_real_time_price=True
                    )
                except Exception as e:
                    # Fallback to original implementation
                    print(f"Error getting market data for chart: {e}")
                    fig = create_equity_chart(
                        st.session_state.bot_status['equity_curve'],
                        st.session_state.bot_status['equity_curve'],
                        show_real_time_price=False
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display recent trades
            if 'trade_history' in st.session_state.bot_status and st.session_state.bot_status['trade_history']:
                st.subheader("Recent Trades")
                trade_df = create_trade_history_table(st.session_state.bot_status['trade_history'])
                if not trade_df.empty:
                    st.dataframe(trade_df, use_container_width=True)
        
        elif status == 'completed':
            st.success("Bot completed successfully")
            
            # Display performance summary
            if 'performance' in st.session_state.bot_status:
                perf = st.session_state.bot_status['performance']
                
                st.subheader("Final Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Initial Capital", f"${perf.get('initial_capital', 0):.2f}")
                    st.metric("Final Value", f"${perf.get('current_portfolio_value', 0):.2f}")
                    st.metric("Total Return", f"${perf.get('total_return', 0):.2f} ({perf.get('pct_return', 0):.2f}%)")
                
                with col2:
                    st.metric("Total Trades", perf.get('total_trades', 0))
                    st.metric("Winning Trades", perf.get('winning_trades', 0))
                    st.metric("Win Rate", f"{perf.get('win_rate', 0):.2f}%")
        
        elif status == 'error':
            st.error(f"Bot encountered an error: {message}")
    
    with tab3:
        st.header("Results & Analysis")
        
        # Display final results and analytics if available
        if ('trade_history' in st.session_state.bot_status and 
            st.session_state.bot_status['trade_history'] and
            'equity_curve' in st.session_state.bot_status and 
            st.session_state.bot_status['equity_curve']):
            
            # Equity curve
            st.subheader("Equity Curve")
            # Get price data from the bot status or fetch current market data
            try:
                # Try to get market data with actual price data
                from utils.binance_data import get_market_data
                # Use trading pair from bot status if available
                trading_pair = (st.session_state.bot_status.get('config', {}).get('trading_pair', 'BTCUSDT') 
                               if 'config' in st.session_state.bot_status else 'BTCUSDT')
                timeframe = (st.session_state.bot_status.get('config', {}).get('timeframe', '15m')
                           if 'config' in st.session_state.bot_status else '15m')
                
                market_data = get_market_data(symbol=trading_pair, interval=timeframe)
                
                # Create price data array in the format expected by create_equity_chart
                price_data = []
                if market_data and 'klines' in market_data:
                    for kline in market_data['klines'][-30:]:  # Get last 30 candles
                        price_data.append({
                            'timestamp': kline[0],  # Open time
                            'price': float(kline[4])  # Close price
                        })
                
                fig = create_equity_chart(
                    st.session_state.bot_status['equity_curve'],
                    price_data if price_data else st.session_state.bot_status['equity_curve'],
                    show_real_time_price=True
                )
            except Exception as e:
                # Fallback to original implementation
                print(f"Error getting market data for chart: {e}")
                fig = create_equity_chart(
                    st.session_state.bot_status['equity_curve'],
                    st.session_state.bot_status['equity_curve'],
                    show_real_time_price=False
                )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade history
            st.subheader("All Trades")
            trade_df = create_trade_history_table(st.session_state.bot_status['trade_history'])
            if not trade_df.empty:
                st.dataframe(trade_df, use_container_width=True)
                
                # Download button for trade history
                csv = trade_df.to_csv(index=False)
                st.download_button(
                    label="Download Trade History CSV",
                    data=csv,
                    file_name="trade_history.csv",
                    mime="text/csv"
                )
        else:
            st.info("No results available yet. Run the bot first to see analysis.")

if __name__ == "__main__":
    main()