"""
V7 Strategy Dashboard

This page provides an enhanced version of the V5 Strategy with additional time configuration options
for each bot, allowing custom start times and operating intervals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time as time_module
from datetime import datetime, date, timedelta, time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_fetcher import DataFetcher
from utils.data_validator import DataValidator
from utils.results_manager import ResultsManager
from backtesting.backtester import Backtester
from visualization.performance_plots import PerformancePlots
from strategy.optimized_alternative_dca_v5 import OptimizedAlternativeDCAv5
from strategy.optimized_alternative_dca_v6 import ALLOWED_INTERVALS

st.set_page_config(
    page_title="V7 Strategy - Bitcoin Backtesting Platform",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data_files' not in st.session_state:
    st.session_state.data_files = []
if 'selected_data' not in st.session_state:
    st.session_state.selected_data = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'v7_bots_config' not in st.session_state:
    st.session_state.v7_bots_config = []

# Initialize data fetcher
data_fetcher = DataFetcher()

# Load custom CSS
st.markdown("""
<style>
    .section-header {
        font-size: 1.5rem;
        color: #f7931a;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #f7931a;
        color: white;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.8rem;
        color: gray;
    }
    .param-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown('<p class="section-header">V7 Time-Optimized Strategy Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
Enhanced version of the V5 Strategy with precision time configuration for each bot.
Configure custom start dates, start times, and operation intervals for maximum flexibility and performance.
""")

# Sidebar for data selection and strategy parameters
with st.sidebar:
    st.header("Backtest Configuration")
    
    # Data selection section
    st.subheader("1. Choose Your Bitcoin Price Data")
    
    # If data files list is empty, load it
    if not st.session_state.data_files:
        st.session_state.data_files = data_fetcher.list_saved_data()
    
    if not st.session_state.data_files:
        st.warning("📋 No price data available. Please visit the Data Manager page to download Bitcoin price history first.")
    else:
        # Create a dataframe for selection
        df_files = pd.DataFrame(st.session_state.data_files)
        
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="margin-bottom: 5px;"><strong>📊 Select Bitcoin Price Data for Your Backtest</strong></p>
        <p style="font-size: 0.9em; margin-bottom: 0;">Choose from your available data sources below. The source includes the exchange name and timeframe (e.g., 1h = hourly data).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File selection with improved formatting
        selected_file = st.selectbox(
            "Bitcoin Price Source",
            options=df_files['filename'].tolist(),
            format_func=lambda x: f"{df_files.loc[df_files['filename'] == x, 'exchange'].iloc[0].capitalize()} • {df_files.loc[df_files['filename'] == x, 'symbol'].iloc[0]} • {df_files.loc[df_files['filename'] == x, 'timeframe'].iloc[0]} data"
        )
        
        # Get the filepath
        filepath = df_files[df_files['filename'] == selected_file]['filepath'].iloc[0]
        
        # Load the data if not already in cache
        if filepath in st.session_state.data_cache:
            df = st.session_state.data_cache[filepath]
        else:
            df = data_fetcher.load_data(filepath)
            if df is not None:
                st.session_state.data_cache[filepath] = df
        
        if df is None or df.empty:
            st.error(f"❌ Could not load data from {selected_file}. The file may be corrupted.")
        else:
            # Store in session state
            st.session_state.selected_data = df
            
            # Calculate date range and points
            start_date = df.index.min().date()
            end_date = df.index.max().date()
            data_points = len(df)
            date_range = (end_date - start_date).days
            
            # Show enhanced data info
            st.markdown(f"""
            <div style="background-color: #e6ffe6; padding: 10px; border-radius: 5px;">
            <p style="margin-bottom: 5px;"><strong>✅ Data Successfully Loaded</strong></p>
            <p style="font-size: 0.9em; margin-bottom: 0;">
            <span style="font-weight: bold;">Date Range:</span> {start_date} to {end_date} ({date_range} days)<br>
            <span style="font-weight: bold;">Data Points:</span> {data_points} price records<br>
            <span style="font-weight: bold;">Source:</span> {df_files.loc[df_files['filename'] == selected_file, 'exchange'].iloc[0].capitalize()}
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date range selection
            st.subheader("Backtest Date Range")
            date_range_option = st.radio(
                "Select date range",
                ["Use full dataset", "Specify date range"],
                index=0
            )
            
            if date_range_option == "Specify date range":
                min_date = df.index.min().date()
                max_date = df.index.max().date()
                
                custom_start_date = st.date_input(
                    "Start date",
                    value=min_date + timedelta(days=30),
                    min_value=min_date,
                    max_value=max_date - timedelta(days=7)
                )
                
                custom_end_date = st.date_input(
                    "End date",
                    value=max_date,
                    min_value=custom_start_date + timedelta(days=7),
                    max_value=max_date
                )
                
                # Filter data based on selected date range
                filtered_data = df[(df.index.date >= custom_start_date) & (df.index.date <= custom_end_date)].copy()
                
                # Update min_date and max_date to use the filtered data range for bot deployments
                min_date = custom_start_date
                max_date = custom_end_date
                
                st.info(f"Using {len(filtered_data)} data points from {custom_start_date} to {custom_end_date}")
            else:
                filtered_data = df.copy()
                min_date = df.index.min().date()
                max_date = df.index.max().date()
                st.info(f"Using full dataset with {len(filtered_data)} data points")
    
    # Strategy parameters section
    st.subheader("2. Strategy Parameters")
    
    # Now create the form with the appropriate fields
    with st.form("v7_strategy_form"):
        st.markdown("#### Strategy Configuration")
        
        initial_capital = st.number_input(
            "Initial Capital (USDT)",
            min_value=1.0,
            value=1000.0,
            step=1.0,
            help="Starting investment amount in USDT"
        )
        
        # Basic parameters
        col1, col2 = st.columns(2)
        with col1:
            investment_pct = st.number_input(
                "Investment Amount (%)",
                key="v7_investment_pct",
                min_value=0.1,
                value=10.0,
                step=0.1,
                help="Percentage of available capital to invest per buy"
            )
            
            price_drop_threshold = st.number_input(
                "Price Drop Threshold (%)",
                key="v7_price_drop_threshold",
                min_value=0.1,
                value=2.0,
                step=0.1,
                help="Minimum price decrease required to trigger a buy"
            )
        
        with col2:
            profit_threshold = st.number_input(
                "Profit Threshold (%)",
                key="v7_profit_threshold",
                min_value=0.1,
                value=3.0,
                step=0.1,
                help="Target profit percentage that triggers selling"
            )
            
            trading_fee_pct = st.number_input(
                "Trading Fee (%)",
                key="v7_trading_fee_pct",
                min_value=0.0,
                value=0.1,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                help="Exchange trading fee percentage"
            )
            
        # Risk management parameters
        st.markdown("#### Risk Management")
        
        col1, col2 = st.columns(2)
        with col1:
            stop_loss_fixed_pct = st.number_input(
                "Fixed Stop-Loss (%)",
                key="v7_stop_loss_fixed_pct",
                min_value=0.0,
                value=15.0,
                step=0.5,
                help="Maximum allowed loss before triggering stop loss (% below entry)"
            )
            
            trailing_stop_pct = st.number_input(
                "Trailing Stop (%)",
                key="v7_trailing_stop_pct",
                min_value=0.0,
                value=1.5,
                step=0.1,
                help="Trailing stop percentage below highest price reached after entry"
            )
            
            trailing_stop_activation_pct = st.number_input(
                "Trailing Stop Activation (%)",
                key="v7_trailing_stop_activation_pct",
                min_value=0.0,
                value=1.0,
                step=0.1,
                help="Profit level that activates trailing stop"
            )
            
        with col2:
            max_position_duration = st.number_input(
                "Max Position Duration (days)",
                key="v7_max_position_duration",
                min_value=1,
                value=180,
                step=1,
                help="Maximum duration to hold a position before forced exit"
            )
            
            use_trailing_stop_main = st.checkbox(
                "Use Trailing Stop as Primary Exit",
                key="v7_use_trailing_stop_main",
                value=True,
                help="If checked, trailing stop will be the primary exit method instead of fixed profit target"
            )
            
            # Simple checkbox for enabling partial profit taking 
            enable_partial_profit_taking = st.checkbox(
                "Enable Partial Profit Taking",
                key="v7_enable_partial_profit",
                value=False,
                help="If checked, only a portion of the position will be sold when profit target is reached"
            )
            
            # Always show the percentage field, but it's only relevant when partial profit taking is enabled
            partial_profit_taking_pct = st.number_input(
                "Profit Taking Percentage (%)",
                key="v7_partial_profit_pct",
                min_value=1.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                help="Percentage of position to sell when taking profits"
            )
            
            # Minimum trade size control - was previously hard-coded
            min_trade_size = st.number_input(
                "Minimum Trade Size (USDT)",
                key="v7_min_trade_size",
                min_value=1.0,
                value=10.0,
                step=1.0,
                help="Minimum USD value for a trade to be executed"
            )
        
        # Multi-bot configuration
        st.markdown("#### Multi-Bot Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            number_of_bots = st.number_input(
                "Number of Bots",
                key="v7_number_of_bots",
                min_value=1,
                value=3,
                max_value=10,
                step=1,
                help="Number of independent bots to run in parallel"
            )
        
        # Bot timing strategy options
        with col2:
            # Use the saved timing strategy from session state if available
            default_strategy_index = 0
            if 'v7_bot_timing_strategy' in st.session_state:
                strategy_options = ["Uniformly spaced", "Evenly distribute across dataset", "Custom bot timing"]
                if st.session_state.v7_bot_timing_strategy in strategy_options:
                    default_strategy_index = strategy_options.index(st.session_state.v7_bot_timing_strategy)
            
            bot_timing_strategy = st.selectbox(
                "Bot Timing Strategy",
                options=["Uniformly spaced", "Evenly distribute across dataset", "Custom bot timing"],
                index=default_strategy_index,
                help="How to space out the bot start dates"
            )
            
            if bot_timing_strategy == "Uniformly spaced":
                # Set default value from session state if available
                default_days = 30
                if 'v7_uniform_deployment_days_value' in st.session_state:
                    default_days = st.session_state.v7_uniform_deployment_days_value
                
                # Use a widget key that's different from the value storage key
                uniform_deployment_days = st.number_input(
                    "Days Between Bot Starts",
                    min_value=1,
                    value=default_days,
                    step=1,
                    help="Number of days between each bot's start date",
                    key="v7_uniform_deployment_days_input"
                )
                
                # Store the value in a different session state key from the widget key
                st.session_state.v7_uniform_deployment_days_value = uniform_deployment_days
                st.info(f"Bots will be deployed {uniform_deployment_days} days apart, starting from the beginning of the dataset.")
            
        # Submit button
        submitted = st.form_submit_button("Prepare Strategy")
        
        # Save settings to session state when form is submitted
        if submitted:
            # Save both timing strategy and deployment days to session state
            st.session_state.v7_bot_timing_strategy = bot_timing_strategy
            
            if bot_timing_strategy == "Uniformly spaced":
                # Store in both keys for backward compatibility
                st.session_state.v7_uniform_deployment_days_value = uniform_deployment_days
                # We need to use this one for other parts of the code that may reference it
                st.session_state.v7_uniform_deployment_days = uniform_deployment_days
                
                # Clear bot start date keys to force recalculation with new deployment days
                for i in range(10):  # Clear for up to 10 bots
                    key = f"v7_bot_{i+1}_start_date"
                    if key in st.session_state:
                        del st.session_state[key]
                
            # Clear any existing bot configuration to force regeneration
            if 'v7_bots_config' in st.session_state:
                st.session_state.v7_bots_config = []
                
            # Pre-calculate and store the new bot configuration with the updated settings
            # This ensures the dates are updated immediately when "Prepare Strategy" is clicked
            new_bot_config = []
            
            # Get date range from data
            if 'selected_data' in st.session_state and st.session_state.selected_data is not None:
                df = st.session_state.selected_data
                min_date = df.index.min().date()
                max_date = df.index.max().date()
                
                # Generate proper start dates for each bot based on the timing strategy
                for i in range(number_of_bots):
                    # Calculate default date based on bot timing strategy
                    if bot_timing_strategy == "Uniformly spaced":
                        # Get the days between value from session state
                        days_between = st.session_state.get('v7_uniform_deployment_days_value', 30)
                        default_date = min_date + timedelta(days=min(i * days_between, (max_date - min_date).days - 30))
                    elif bot_timing_strategy == "Evenly distribute across dataset":
                        bot_spacing = (max_date - min_date).days / max(1, number_of_bots - 1) if number_of_bots > 1 else 0
                        default_date = min_date + timedelta(days=int(i * bot_spacing))
                    else:
                        default_date = min_date
                        
                    # Set default time
                    default_hour = (9 + i * 2) % 24
                    default_time = time(hour=default_hour, minute=0)
                    
                    # Set default interval
                    interval_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                    default_intervals = {
                        0: 2,  # Default first bot to 15m
                        1: 4,  # Default second bot to 1h 
                        2: 1,  # Default third bot to 5m
                    }
                    default_idx = default_intervals.get(i, i % len(interval_options))
                    default_interval = interval_options[default_idx]
                    
                    # Add to configuration
                    new_bot_config.append({
                        "bot_id": i+1,
                        "start_date": default_date.strftime('%Y-%m-%d'),
                        "start_time": default_time.strftime('%H:%M'),
                        "interval": default_interval
                    })
                
                # Store the pre-calculated configuration
                st.session_state.v7_bots_config = new_bot_config
                
            # Signal to the UI that the bot configuration has been updated
            st.session_state.v7_bot_config_updated = True
            
# Main section - based on form submission
if 'selected_data' in st.session_state and st.session_state.selected_data is not None:
    df = st.session_state.selected_data

    # Configure custom bot timing 
    if 'v7_number_of_bots' in st.session_state:
        number_of_bots = st.session_state.v7_number_of_bots
        
        st.markdown('<p class="section-header">🔄 Custom Bot Timing Configuration</p>', unsafe_allow_html=True)
        st.info("Configure the start date, start time, and operating interval for each bot")
        
        # Get date range from data
        min_date = df.index.min().date()
        max_date = df.index.max().date()
            
        # Create bot configuration form
        with st.form("bot_timing_form"):
            bot_config = []
            
            # Reset cached bot start dates if the days between bots changed
            # This ensures the default dates reflect the new spacing setting
            if 'v7_form_days_between' not in st.session_state or st.session_state.get('v7_form_days_between') != st.session_state.get('v7_uniform_deployment_days_value', 30):
                # Clear previous bot start dates
                for i in range(10):  # Clear for up to 10 bots
                    key = f"v7_bot_{i+1}_start_date"
                    if key in st.session_state:
                        del st.session_state[key]
                # Store current days_between value to track future changes
                st.session_state.v7_form_days_between = st.session_state.get('v7_uniform_deployment_days_value', 30)
            
            # Create columns for titles
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Start Date")
            with col2:
                st.markdown("#### Start Time")
            with col3:
                st.markdown("#### Operating Interval")
                
            for i in range(number_of_bots):
                st.markdown(f"### Bot {i+1}")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Calculate default date based on bot timing strategy
                    if bot_timing_strategy == "Uniformly spaced":
                        # Use a consistent approach to get the days between bots
                        days_between = st.session_state.get('v7_uniform_deployment_days_value', 30)
                        default_date = min_date + timedelta(days=min(i * days_between, (max_date - min_date).days - 30))
                    elif bot_timing_strategy == "Evenly distribute across dataset":
                        bot_spacing = (max_date - min_date).days / max(1, number_of_bots - 1) if number_of_bots > 1 else 0
                        default_date = min_date + timedelta(days=int(i * bot_spacing))
                    else:
                        default_date = min_date
                        
                    # Bot start date
                    bot_start_date = st.date_input(
                        f"Bot {i+1} Start Date",
                        value=default_date,
                        min_value=min_date,
                        max_value=max_date - timedelta(days=30),  # At least 30 days of data
                        key=f"v7_bot_{i+1}_start_date"
                    )
                
                with col2:
                    # Bot start time - distribute across the day for diversity
                    default_hour = (9 + i * 2) % 24
                    bot_start_time = st.time_input(
                        f"Bot {i+1} Start Time",
                        value=time(hour=default_hour, minute=0),
                        key=f"v7_bot_{i+1}_start_time"
                    )
                
                with col3:
                    # Bot operating interval with diverse defaults
                    interval_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                    default_intervals = {
                        0: 2,  # Default first bot to 15m
                        1: 4,  # Default second bot to 1h 
                        2: 1,  # Default third bot to 5m
                    }
                    default_idx = default_intervals.get(i, i % len(interval_options))
                    
                    bot_interval = st.selectbox(
                        f"Bot {i+1} Operating Interval",
                        options=interval_options,
                        index=default_idx,
                        key=f"v7_bot_{i+1}_interval"
                    )
                
                # Store configuration
                bot_config.append({
                    "bot_id": i+1,
                    "start_date": bot_start_date.strftime('%Y-%m-%d'),
                    "start_time": bot_start_time.strftime('%H:%M'),
                    "interval": bot_interval
                })
            
            # Submit button
            bot_timing_submitted = st.form_submit_button("Save Bot Configuration")
            
            if bot_timing_submitted:
                # Store bot configuration in session state
                st.session_state.v7_bots_config = bot_config
                st.success("Bot timing configuration saved successfully!")
                
    # Run backtest button and results
    st.markdown('<p class="section-header">Run V7 Strategy Backtest</p>', unsafe_allow_html=True)
    
    # Initialize bot configuration if needed
    if 'v7_bots_config' not in st.session_state:
        st.session_state.v7_bots_config = []
    
    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        run_backtest = st.button("🚀 Run Backtest", key="run_v7_backtest")
    
    with run_col2:
        if 'v7_bots_config' in st.session_state and len(st.session_state.v7_bots_config) > 0:
            st.success(f"{number_of_bots} bots configured and ready for backtesting!")
        else:
            st.warning("Please configure bot timing settings first.")
    
    if run_backtest and df is not None and len(st.session_state.v7_bots_config) > 0:
        # Display a loading message and progress bar
        st.info("Running V7 strategy backtest with custom bot timing...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define progress callback function
        def update_progress(percent, message):
            progress_bar.progress(percent / 100)
            status_text.text(message)
        
        # Prepare parameters for the strategy
        strategy_params = {
            'initial_capital': st.session_state.get('v7_initial_capital', 1000.0),
            'investment_pct': st.session_state.get('v7_investment_pct', 10.0),
            'price_drop_threshold': st.session_state.get('v7_price_drop_threshold', 2.0),
            'profit_threshold': st.session_state.get('v7_profit_threshold', 3.0),
            'trading_fee_pct': st.session_state.get('v7_trading_fee_pct', 0.1),
            'stop_loss_fixed_pct': st.session_state.get('v7_stop_loss_fixed_pct', 15.0),
            'trailing_stop_pct': st.session_state.get('v7_trailing_stop_pct', 1.5),
            'use_trailing_stop_main': st.session_state.get('v7_use_trailing_stop_main', True),
            'max_position_duration': st.session_state.get('v7_max_position_duration', 180),
            'number_of_bots': number_of_bots,
            'bots_config': st.session_state.v7_bots_config,
            # Include parameters that were previously hard-coded
            'min_trade_size': st.session_state.get('v7_min_trade_size', 10.0),
            'enable_partial_profit_taking': st.session_state.get('v7_enable_partial_profit', False),
            'partial_profit_taking_pct': st.session_state.get('v7_partial_profit_pct', 50.0),
            'trailing_stop_activation_pct': st.session_state.get('v7_trailing_stop_activation_pct', 1.0)
        }
        
        try:
            # Initialize backtester with progress updates
            backtester = Backtester(update_progress)
            
            # Initialize the strategy using OptimizedAlternativeDCAv5 as the base
            # but with V7's advanced time configuration
            strategy = OptimizedAlternativeDCAv5(
                initial_capital=strategy_params['initial_capital'],
                investment_pct=strategy_params['investment_pct'],
                price_drop_threshold=strategy_params['price_drop_threshold'],
                profit_threshold=strategy_params['profit_threshold'],
                trading_fee_pct=strategy_params['trading_fee_pct'],
                stop_loss_fixed_pct=strategy_params['stop_loss_fixed_pct'],
                trailing_stop_pct=strategy_params['trailing_stop_pct'],
                max_position_duration=strategy_params['max_position_duration'],
                min_trade_size=strategy_params.get('min_trade_size', 10.0),
                partial_profit_taking=strategy_params.get('enable_partial_profit_taking', False),
                profit_taking_percentage=strategy_params.get('partial_profit_taking_pct', 50.0),
                use_trailing_stop_main=strategy_params.get('use_trailing_stop_main', True),
                trailing_stop_activation_pct=strategy_params.get('trailing_stop_activation_pct', 1.0),
                number_of_bots=strategy_params['number_of_bots'],
                # Extract starting dates and times from bot configurations
                bots_starting_dates=[bot['start_date'] for bot in strategy_params['bots_config']],
                bots_starting_times=[bot['start_time'] for bot in strategy_params['bots_config']]
            )
            
            # Set up the backtester with the strategy
            backtester.set_strategy(strategy)
            
            # Make sure we have data
            if 'filtered_data' not in locals() or filtered_data is None or filtered_data.empty:
                # Fetch data if not already available
                data_fetcher = DataFetcher()
                filtered_data = data_fetcher.fetch_historical_data('binance', 'BTC/USDT', '1h', days=365)
            
            # Load data into backtester - use filtered data based on selected date range
            backtester.set_data(filtered_data)
            
            # Run the actual backtest with progress updates
            results = backtester.run(progress_callback=update_progress)
            
            # Store results for visualization
            st.session_state.v7_backtest_results = results
            
            # Also store a copy of the results to keep them around after saving
            st.session_state.v7_backtest_results_persistent = results.copy()
            
            # Update final progress
            progress_bar.progress(100)
            status_text.text("Backtest completed successfully!")
            
            # Display a success message
            st.success("V7 Strategy backtest with custom timing configuration completed!")
            
            # Display backtest results
            st.markdown('<p class="section-header">Backtest Results</p>', unsafe_allow_html=True)
            
            # Create a multi-tab interface for the results matching backtest.py structure exactly
            results_tab1, results_tab2, results_tab3, results_tab4 = st.tabs(["Summary", "Equity & Drawdown", "Trades", "Detailed Analysis"])
            
            # Use actual backtest results instead of mock data
            if 'v7_backtest_results' in st.session_state and st.session_state.v7_backtest_results is not None:
                backtest_results = st.session_state.v7_backtest_results
            elif 'v7_backtest_results_persistent' in st.session_state and st.session_state.v7_backtest_results_persistent is not None:
                # Use the persistent copy if the original was cleared
                backtest_results = st.session_state.v7_backtest_results_persistent
                # Restore the main results from persistent copy
                st.session_state.v7_backtest_results = st.session_state.v7_backtest_results_persistent
            else:
                # Default values if no results available yet
                backtest_results = {
                    'return_pct': 0.0,
                    'outperformance': 0.0,
                    'max_drawdown_pct': 0.0,
                    'final_equity': strategy_params['initial_capital'],
                    'initial_capital': strategy_params['initial_capital'],
                    'trades_count': 0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            with results_tab1:
                st.markdown("### Backtest Results Summary")
                
                # Key metrics in columns - with USDT and BTC-based returns
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "USDT Return",
                        f"{backtest_results['return_pct']:.2f}%",
                        f"{backtest_results.get('outperformance', 0):.2f}% vs B&H"
                    )
                
                # Calculate BTC-based return with improved robustness
                btc_return = None
                
                # Let's create a dedicated helper function to calculate BTC returns
                def calculate_btc_based_return(backtest_results):
                    """
                    Calculates the return in BTC terms instead of USD.
                    This shows how the strategy performed compared to simply holding BTC.
                    """
                    # Approach #1: Calculate from trade data if available
                    if 'trades' in backtest_results and len(backtest_results['trades']) > 0 and 'equity_curve' in backtest_results:
                        trades_df = backtest_results['trades']
                        equity_df = backtest_results['equity_curve']
                        initial_capital = backtest_results.get('initial_capital', 0)
                        
                        # Make sure we have price data to work with
                        if not equity_df.empty and 'close' in equity_df.columns:
                            start_price = equity_df['close'].iloc[0]
                            end_price = equity_df['close'].iloc[-1]
                            
                            # Calculate BTC holdings from trades if possible
                            if any(col in trades_df.columns for col in ['btc_amount', 'amount', 'amount_btc']):
                                # Find the BTC amount column
                                btc_column = None
                                for col_name in ['btc_amount', 'amount_btc', 'amount']:
                                    if col_name in trades_df.columns:
                                        btc_column = col_name
                                        break
                                
                                if btc_column:
                                    # Define a function to determine if a trade is a buy
                                    def is_buy_trade(row):
                                        if 'action' in row and isinstance(row['action'], str):
                                            return row['action'].upper() == 'BUY'
                                        elif 'type' in row and isinstance(row['type'], str):
                                            return row['type'].upper() == 'BUY'
                                        return False
                                    
                                    # Define a function to determine if a trade is a sell
                                    def is_sell_trade(row):
                                        if 'action' in row and isinstance(row['action'], str):
                                            return row['action'].upper() == 'SELL'
                                        elif 'type' in row and isinstance(row['type'], str):
                                            return row['type'].upper() == 'SELL'
                                        return False
                                    
                                    # Calculate total BTC from buys and sells
                                    total_btc_bought = 0
                                    total_btc_sold = 0
                                    
                                    for _, trade in trades_df.iterrows():
                                        btc_amount = trade.get(btc_column, 0)
                                        if pd.notna(btc_amount):
                                            if is_buy_trade(trade):
                                                total_btc_bought += btc_amount
                                            elif is_sell_trade(trade):
                                                total_btc_sold += btc_amount
                                    
                                    # Calculate final BTC holdings
                                    final_btc = total_btc_bought - total_btc_sold
                                    
                                    # If we have valid BTC amount, calculate equivalent BTC from Buy & Hold
                                    if final_btc > 0 and start_price > 0:
                                        equivalent_btc = initial_capital / start_price
                                        if equivalent_btc > 0:
                                            return ((final_btc / equivalent_btc) - 1) * 100
                            
                            # Approach #2: If we can't calculate from individual trade amounts, 
                            # use the overall equity
                            final_equity = backtest_results.get('final_equity', 0)
                            if final_equity > 0 and initial_capital > 0 and start_price > 0 and end_price > 0:
                                # How much BTC we could have bought at the start
                                initial_btc_value = initial_capital / start_price
                                # How much BTC our final equity is worth
                                final_btc_value = final_equity / end_price
                                
                                if initial_btc_value > 0:
                                    return ((final_btc_value / initial_btc_value) - 1) * 100
                    
                    # Approach #3: Direct comparison from equity curve if available
                    if 'equity_curve' in backtest_results and not backtest_results['equity_curve'].empty:
                        equity_df = backtest_results['equity_curve']
                        
                        if 'equity' in equity_df.columns and 'close' in equity_df.columns:
                            # Get start and end values
                            start_equity = equity_df['equity'].iloc[0]
                            end_equity = equity_df['equity'].iloc[-1]
                            start_price = equity_df['close'].iloc[0]
                            end_price = equity_df['close'].iloc[-1]
                            
                            # Calculate BTC values
                            if start_price > 0 and end_price > 0:
                                start_btc_value = start_equity / start_price
                                end_btc_value = end_equity / end_price
                                
                                if start_btc_value > 0:
                                    return ((end_btc_value / start_btc_value) - 1) * 100
                    
                    # If all methods fail, return None
                    return None
                
                # Get the BTC return using our helper function
                try:
                    btc_return = calculate_btc_based_return(backtest_results)
                except Exception as e:
                    # Log error but don't crash
                    st.error(f"Error calculating BTC return: {str(e)}")
                    btc_return = None
                
                with col2:
                    if btc_return is not None:
                        st.metric(
                            "BTC Return",
                            f"{btc_return:.2f}%",
                            "vs holding BTC"
                        )
                    else:
                        st.metric(
                            "BTC Return",
                            "N/A",
                            "Data unavailable"
                        )
                
                with col3:
                    st.metric(
                        "Max Drawdown",
                        f"{backtest_results['max_drawdown_pct']:.2f}%"
                    )
                
                with col4:
                    st.metric(
                        "Final Equity",
                        f"${backtest_results['final_equity']:.2f}",
                        f"${backtest_results['final_equity'] - backtest_results['initial_capital']:.2f}"
                    )
                
                with col5:
                    st.metric(
                        "Total Trades",
                        f"{backtest_results['trades_count']}"
                    )
                
                # Market Summary for Selected Period
                st.markdown("### Market Summary for Selected Period")
                
                # Calculate market metrics if we have price data
                if 'filtered_data' in locals() and filtered_data is not None and not filtered_data.empty and 'close' in filtered_data.columns:
                    # Price change over period
                    start_price = filtered_data['close'].iloc[0]
                    end_price = filtered_data['close'].iloc[-1]
                    price_change_pct = ((end_price / start_price) - 1) * 100
                    
                    # Calculate volatility (standard deviation of daily returns)
                    daily_returns = filtered_data['close'].pct_change().dropna() * 100
                    volatility = daily_returns.std()
                    
                    # Determine market type based on trend and volatility
                    if price_change_pct > 5:
                        market_type = "Bullish" if volatility < 3 else "Volatile Bullish"
                    elif price_change_pct < -5:
                        market_type = "Bearish" if volatility < 3 else "Volatile Bearish"
                    else:
                        market_type = "Sideways" if volatility < 2 else "Volatile Sideways"
                    
                    # Display market metrics
                    market_col1, market_col2, market_col3 = st.columns(3)
                    
                    with market_col1:
                        st.metric("Price Change", f"{price_change_pct:.2f}%")
                        
                    with market_col2:
                        st.metric("Volatility", f"{volatility:.2f}%")
                        
                    with market_col3:
                        st.metric("Market Type", market_type)
                else:
                    st.info("No price data available to calculate market metrics.")
                
                # Performance Metrics (same as in backtest.py)
                st.markdown("### Performance Metrics")
                
                # Get trade metrics if available
                if (('v7_backtest_results' in st.session_state and st.session_state.v7_backtest_results is not None) or
                    ('v7_backtest_results_persistent' in st.session_state and st.session_state.v7_backtest_results_persistent is not None)):
                    
                    # Calculate annualized return
                    if 'equity_curve' in backtest_results and not backtest_results['equity_curve'].empty:
                        days = (backtest_results['equity_curve'].index[-1] - backtest_results['equity_curve'].index[0]).days
                        if days > 0:
                            annualized_return = ((1 + backtest_results['return_pct']/100) ** (365/days) - 1) * 100
                        else:
                            annualized_return = 0
                    else:
                        annualized_return = 0
                    
                    # Get sharpe and sortino ratio if available
                    sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
                    sortino_ratio = backtest_results.get('sortino_ratio', 0)
                    
                    # Get buy & hold return
                    buy_hold_return = backtest_results.get('buy_hold_return_pct', 0)
                    
                    # Calculate alpha
                    alpha = backtest_results.get('outperformance', 0)
                    
                    # Get win rate and profit factor if available
                    if 'trades' in backtest_results and len(backtest_results['trades']) > 0:
                        trades = backtest_results['trades']
                        if 'profit_pct' in trades.columns:
                            win_rate = (trades['profit_pct'] > 0).mean() * 100
                            avg_profit = trades['profit_pct'].mean()
                            
                            # Calculate profit factor
                            profitable_trades = trades[trades['profit_pct'] > 0]
                            losing_trades = trades[trades['profit_pct'] <= 0]
                            
                            total_profit = profitable_trades['profit_pct'].sum() if len(profitable_trades) > 0 else 0
                            total_loss = abs(losing_trades['profit_pct'].sum()) if len(losing_trades) > 0 else 1  # Avoid div by zero
                            
                            profit_factor = total_profit / total_loss if total_loss > 0 else total_profit
                        else:
                            win_rate = 0
                            avg_profit = 0
                            profit_factor = 0
                    else:
                        win_rate = 0
                        avg_profit = 0
                        profit_factor = 0
                    
                    # Create dataframes for metrics
                    return_metrics = pd.DataFrame({
                        'Metric': ['Total Return', 'Annualized Return', 'Buy & Hold Return', 'Alpha (vs. Buy & Hold)'],
                        'Value': [f"{backtest_results['return_pct']:.2f}%", 
                                 f"{annualized_return:.2f}%", 
                                 f"{buy_hold_return:.2f}%", 
                                 f"{alpha:.2f}%"]
                    })
                    
                    risk_metrics = pd.DataFrame({
                        'Metric': ['Maximum Drawdown', 'Sharpe Ratio', 'Sortino Ratio'],
                        'Value': [f"{backtest_results['max_drawdown_pct']:.2f}%", 
                                 f"{sharpe_ratio:.2f}", 
                                 f"{sortino_ratio:.2f}"]
                    })
                    
                    trade_metrics = pd.DataFrame({
                        'Metric': ['Total Trades', 'Win Rate', 'Average Profit', 'Profit Factor'],
                        'Value': [f"{backtest_results['trades_count']}", 
                                 f"{win_rate:.2f}%", 
                                 f"{avg_profit:.2f}%", 
                                 f"{profit_factor:.2f}"]
                    })
                    
                    financial_metrics = pd.DataFrame({
                        'Metric': ['Initial Capital', 'Final Equity'],
                        'Value': [f"${backtest_results['initial_capital']:.2f}", 
                                 f"${backtest_results['final_equity']:.2f}"]
                    })
                    
                    # Display metrics in a nice layout
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.markdown("#### Return Metrics")
                        st.dataframe(return_metrics, hide_index=True)
                        
                        st.markdown("#### Risk Metrics")
                        st.dataframe(risk_metrics, hide_index=True)
                        
                    with metrics_col2:
                        st.markdown("#### Trades Metrics")
                        st.dataframe(trade_metrics, hide_index=True)
                        
                        st.markdown("#### Financial Metrics")
                        st.dataframe(financial_metrics, hide_index=True)
                        
                    # Add Save Results button
                    if (('v7_backtest_results' in st.session_state and st.session_state.v7_backtest_results is not None) or
                        ('v7_backtest_results_persistent' in st.session_state and st.session_state.v7_backtest_results_persistent is not None)):
                        # Create a JSON string representation of the results
                        import json
                        import io
                        
                        # Filter the results to exclude large DataFrame values for download
                        download_results = {
                            'strategy': 'V7 Strategy',
                            'parameters': strategy_params,
                            'performance': {
                                'return_pct': backtest_results['return_pct'],
                                'max_drawdown_pct': backtest_results['max_drawdown_pct'],
                                'sharpe_ratio': sharpe_ratio,
                                'sortino_ratio': sortino_ratio,
                                'win_rate': win_rate,
                                'profit_factor': profit_factor,
                                'total_trades': backtest_results['trades_count'],
                                'buy_hold_return': buy_hold_return,
                                'alpha': alpha,
                                'initial_capital': backtest_results['initial_capital'],
                                'final_equity': backtest_results['final_equity']
                            },
                            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Convert to JSON string
                        results_json = json.dumps(download_results, indent=2)
                        
                        # Create columns for save and download
                        save_col1, save_col2 = st.columns(2)
                        
                        with save_col1:
                            # Use a unique key for the button to prevent rerun issues
                            save_button_key = "save_results_project_v7_" + str(pd.Timestamp.now().timestamp())
                            
                            # Store the save status in session state to survive page refresh
                            if 'save_status' not in st.session_state:
                                st.session_state.save_status = {}
                            
                            # Create a button that directly saves without refreshing the page
                            if st.button("Save Results to Project", key=save_button_key):
                                try:
                                    import os
                                    results_dir = "results"
                                    if not os.path.exists(results_dir):
                                        os.makedirs(results_dir)
                                    
                                    # Generate timestamp for the filename
                                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
                                    
                                    # Create a filename with timestamp
                                    filename = f"v7_strategy_backtest_{timestamp}.json"
                                    filepath = os.path.join(results_dir, filename)
                                    
                                    # Save to file
                                    with open(filepath, 'w') as f:
                                        f.write(results_json)
                                    
                                    # Store the saved file info in session state for future reference
                                    if 'saved_v7_results' not in st.session_state:
                                        st.session_state.saved_v7_results = []
                                    
                                    st.session_state.saved_v7_results.append({
                                        'filepath': filepath,
                                        'timestamp': timestamp,
                                        'filename': filename
                                    })
                                    
                                    # Store the fact that we have saved results
                                    st.session_state.v7_results_saved = True
                                    
                                    # Let the user know where the results were saved
                                    st.success(f"Results saved to {filepath}")
                                    
                                    # Create a persistent copy of the complete results
                                    # This will be used to maintain state across pages
                                    st.session_state.v7_backtest_results_persistent = st.session_state.v7_backtest_results
                                    
                                    # Save full results data with DataFrames to session state
                                    # Convert DataFrames to records format for JSON compatibility
                                    full_results_data = download_results.copy()
                                    
                                    # If equity_curve exists in the results, add it in records format
                                    if 'equity_curve' in backtest_results:
                                        full_results_data['equity_curve'] = backtest_results['equity_curve'].to_dict(orient='records') if not backtest_results['equity_curve'].empty else []
                                    
                                    # If trades exists in the results, add it in records format    
                                    if 'trades' in backtest_results:
                                        full_results_data['trades'] = backtest_results['trades'].to_dict(orient='records') if not backtest_results['trades'].empty else []
                                    
                                    # Store complete results data in session state
                                    if 'saved_v7_results_data' not in st.session_state:
                                        st.session_state.saved_v7_results_data = {}
                                    
                                    result_key = f"v7_strategy_backtest_{timestamp}"
                                    st.session_state.saved_v7_results_data[result_key] = full_results_data
                                    
                                except Exception as e:
                                    st.error(f"Error saving results: {str(e)}")
                            
                            # If results have been saved, add a message confirming it
                            elif 'v7_results_saved' in st.session_state and st.session_state.v7_results_saved:
                                st.info("Results have been saved! You can find them in the Results page.")
                        
                        with save_col2:
                            # Add download button for user to download locally
                            st.download_button(
                                "Download Results as JSON",
                                data=results_json,
                                file_name=f"v7_strategy_backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json"
                            )
                else:
                    st.info("Run a backtest to see detailed performance metrics.")
                
                # Strategy parameters summary - matching backtest.py format
                st.markdown("### Strategy Parameters")
                
                # Parameters table
                params_df = pd.DataFrame({
                    'Parameter': [
                        'Strategy Type',
                        'Initial Capital',
                        'Investment Amount',
                        'Price Drop Threshold',
                        'Profit Threshold',
                        'Fixed Stop-Loss',
                        'Trailing Stop',
                        'Trailing Stop Activation',
                        'Use Trailing Stop as Primary',
                        'Max Position Duration',
                        'Number of Bots',
                        'Trading Fee',
                        'Start Date',
                        'End Date',
                        'Backtest Duration'
                    ],
                    'Value': [
                        'V7 Strategy',
                        f"${strategy_params['initial_capital']:.2f}",
                        f"{strategy_params['investment_pct']:.1f}%",
                        f"{strategy_params['price_drop_threshold']:.1f}%",
                        f"{strategy_params['profit_threshold']:.1f}%",
                        f"{strategy_params['stop_loss_fixed_pct']:.1f}%",
                        f"{strategy_params['trailing_stop_pct']:.1f}%",
                        f"After {backtest_results['return_pct'] / 2:.1f}% profit",
                        f"{'Yes' if strategy_params['use_trailing_stop_main'] else 'No'}",
                        f"{strategy_params['max_position_duration']} days",
                        f"{len(st.session_state.v7_bots_config)}",
                        f"{strategy_params['trading_fee_pct']:.2f}%",
                        df.index.min().strftime('%Y-%m-%d'),
                        df.index.max().strftime('%Y-%m-%d'),
                        f"{(df.index.max().date() - df.index.min().date()).days} days"
                    ]
                })
                
                # Display parameters table 
                st.dataframe(params_df, use_container_width=True, hide_index=True)
            
            with results_tab2:
                st.markdown("### Equity Curve and Drawdown Analysis")
                
                # Store the backtest results in session state to ensure they persist
                if 'v7_backtest_results' in st.session_state:
                    # Update the persistent copy whenever we have results available
                    st.session_state.v7_backtest_results_persistent = st.session_state.v7_backtest_results
                
                # Define a callback function to toggle normalization without page refresh
                def toggle_normalize():
                    st.session_state.normalize_equity_value = not st.session_state.normalize_equity_value
                
                # Initialize normalize_equity in session state if not present
                if 'normalize_equity_value' not in st.session_state:
                    st.session_state.normalize_equity_value = True
                
                # Create a simple checkbox that uses the callback for state change
                normalize_equity = st.checkbox(
                    "Normalize equity curves to 100%", 
                    value=st.session_state.normalize_equity_value,
                    key="normalize_equity_checkbox",
                    on_change=toggle_normalize
                )
                
                # Equity curve
                st.markdown("#### Strategy Equity vs Buy & Hold")
                
                # Use backtest results from persistent storage
                if (('v7_backtest_results_persistent' in st.session_state and 
                     st.session_state.v7_backtest_results_persistent is not None and 
                     'equity_curve' in st.session_state.v7_backtest_results_persistent)):
                    # Use persistent equity curve data to avoid losing results
                    equity_df = st.session_state.v7_backtest_results_persistent['equity_curve'].copy()
                elif ('v7_backtest_results' in st.session_state and 
                     st.session_state.v7_backtest_results is not None and 
                     'equity_curve' in st.session_state.v7_backtest_results):
                    # Fallback to non-persistent results if needed
                    equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                    
                    # Make sure we have the required columns
                    if 'equity' not in equity_df.columns and 'portfolio_value' in equity_df.columns:
                        equity_df['equity'] = equity_df['portfolio_value']
                    
                    # Make sure we have buy & hold data
                    if 'buy_hold_equity' not in equity_df.columns and 'buy_hold_value' in equity_df.columns:
                        equity_df['buy_hold_equity'] = equity_df['buy_hold_value']
                    elif 'buy_hold_equity' not in equity_df.columns:
                        # Create buy & hold data if not available
                        equity_df['buy_hold_equity'] = equity_df['equity'].iloc[0] * (equity_df['close'] / equity_df['close'].iloc[0])
                else:
                    # Fallback to placeholder data if actual results not available
                    st.warning("No actual backtest results found. Please run a backtest first.")
                    
                    # Get date range from data
                    if 'df' in locals() and df is not None and not df.empty:
                        start_date = df.index.min()
                        end_date = df.index.max()
                    else:
                        start_date = pd.to_datetime('2025-01-01')
                        end_date = pd.to_datetime('2025-03-15')
                    
                    # Create placeholder data for visualization
                    equity_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    equity_df = pd.DataFrame(index=equity_dates)
                    equity_df['equity'] = strategy_params['initial_capital']
                    equity_df['buy_hold_equity'] = strategy_params['initial_capital']
                
                # Create equity figure
                equity_fig = go.Figure()
                
                # Get initial values for potential normalization
                # Important: Both strategy and buy & hold must use the same initial capital for fair comparison
                initial_capital = strategy_params.get('initial_capital', equity_df['equity'].iloc[0] if not equity_df.empty else 1000.0)
                
                # For normalization, we need to ensure both curves start from the same point (100%)
                # Prepare y values based on normalization setting (using session state value)
                if st.session_state.normalize_equity_value:
                    # Normalize to percentage change from initial value (100 = starting point)
                    # CRITICAL: Use the same initial_capital for both to ensure fair comparison
                    strategy_y = equity_df['equity'] / initial_capital * 100
                    buyhold_y = equity_df['buy_hold_equity'] / initial_capital * 100
                    y_axis_title = 'Percentage of Initial Capital (%)'
                else:
                    # Use actual equity values
                    strategy_y = equity_df['equity']
                    buyhold_y = equity_df['buy_hold_equity']
                    y_axis_title = 'Equity Value (USDT)'
                
                # Add equity curve line
                equity_fig.add_trace(
                    go.Scatter(
                        x=equity_df.index,
                        y=strategy_y,
                        name='V7 Strategy',
                        line=dict(color='#1f77b4', width=2)
                    )
                )
                
                # Add buy and hold curve
                equity_fig.add_trace(
                    go.Scatter(
                        x=equity_df.index,
                        y=buyhold_y,
                        name='Buy & Hold',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    )
                )
                
                # Format the chart
                equity_fig.update_layout(
                    title='Equity Curve: Strategy vs Buy & Hold',
                    xaxis_title='Date',
                    yaxis_title=y_axis_title,  # Use the title based on normalization setting
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(equity_fig, use_container_width=True, key="v7_equity_curve")
                
                # Bot comparison chart
                st.markdown("## Individual Bot Performance")
                
                # Create a blue info box with explanation text matching the screenshot
                st.markdown("""
                <div style="background-color: #f0f5ff; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <p style="margin: 0; color: #1F4287;">This chart shows the performance of each individual bot. When normalized, all bots start at 100% of their individual capital regardless of when they were deployed, allowing direct comparison of relative performance between bots.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create the bot comparison figure
                bot_fig = go.Figure()
                
                # Add a line for each bot
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                
                # Check if we have actual bot results from the backtest
                if ('v7_backtest_results' in st.session_state and 
                    st.session_state.v7_backtest_results is not None and 
                    'trades' in st.session_state.v7_backtest_results and
                    'equity_curve' in st.session_state.v7_backtest_results):
                    
                    # Use actual bot data from backtest results
                    has_valid_bot_data = False
                    
                    try:
                        # Get trades data
                        trades_df = st.session_state.v7_backtest_results['trades']
                        equity_curve = st.session_state.v7_backtest_results['equity_curve']
                        
                        # Check if we have bot_id or bot_number column in trades
                        if 'bot_id' in trades_df.columns or 'bot_number' in trades_df.columns:
                            # Use bot_id or bot_number column
                            bot_column = 'bot_id' if 'bot_id' in trades_df.columns else 'bot_number'
                            
                            # Get unique bot IDs
                            unique_bots = trades_df[bot_column].unique()
                            
                            # For each bot, create a line on the chart
                            for i, bot_id in enumerate(unique_bots):
                                # Filter trades for this bot
                                bot_trades = trades_df[trades_df[bot_column] == bot_id]
                                
                                # Create a simplified equity curve for this bot (starting with 1000 and adding profits)
                                if len(bot_trades) > 0:
                                    # Get bot config to show interval in legend
                                    bot_config = next((b for b in st.session_state.v7_bots_config if str(b['bot_id']) == str(bot_id)), None)
                                    interval = bot_config['interval'] if bot_config else 'Unknown'
                                    
                                    # Calculate equity values for each trade date
                                    date_column = None
                                    for col in ['buy_date', 'timestamp', 'date']:
                                        if col in bot_trades.columns:
                                            date_column = col
                                            break
                                    
                                    if date_column is not None:
                                        # Start with initial capital from strategy params (or default 1000 per bot)
                                        bot_equity = strategy_params.get('initial_capital_per_bot', 1000.0)
                                        equity_points = []
                                        
                                        # Add initial point (use first date)
                                        first_date = pd.to_datetime(bot_trades[date_column].min())
                                        if not pd.isnull(first_date):
                                            # Get bot config to find start date
                                            bot_start_date = first_date
                                            if bot_config and 'start_date' in bot_config:
                                                try:
                                                    # Parse date from bot configuration
                                                    config_start_date = pd.to_datetime(bot_config['start_date'])
                                                    # Use configuration date if valid
                                                    if not pd.isnull(config_start_date):
                                                        bot_start_date = config_start_date
                                                except:
                                                    # If parsing fails, use the first trade date
                                                    pass
                                            
                                            # Add the initial point using the correct start date
                                            equity_points.append({
                                                'date': bot_start_date,
                                                'equity': bot_equity
                                            })
                                        
                                        # Sort trades by date
                                        bot_trades = bot_trades.sort_values(date_column)
                                        
                                        # Calculate equity after each trade
                                        for _, trade in bot_trades.iterrows():
                                            date = pd.to_datetime(trade[date_column])
                                            if pd.isnull(date):
                                                continue
                                                
                                            # Get profit from transaction if available
                                            if 'profit_amount' in trade and pd.notna(trade['profit_amount']):
                                                profit_amount = trade['profit_amount']
                                            elif 'profit_pct' in trade and pd.notna(trade['profit_pct']):
                                                # Calculate profit amount from percentage
                                                profit_amount = (trade['profit_pct'] / 100) * bot_equity
                                            else:
                                                profit_amount = 0
                                                
                                            # Update equity
                                            bot_equity += profit_amount
                                            
                                            # Add this point to our equity curve
                                            equity_points.append({
                                                'date': date,
                                                'equity': bot_equity
                                            })
                                        
                                        # Add final point (use last date from the equity curve)
                                        if equity_curve is not None and not equity_curve.empty:
                                            last_date = equity_curve.index.max()
                                            if not pd.isnull(last_date):
                                                equity_points.append({
                                                    'date': last_date,
                                                    'equity': bot_equity
                                                })
                                        
                                        # Convert to DataFrame if we have points
                                        if equity_points:
                                            bot_equity_df = pd.DataFrame(equity_points)
                                            bot_equity_df = bot_equity_df.set_index('date')
                                            
                                            # Prepare values based on normalization setting
                                            if st.session_state.normalize_equity_value:
                                                # Normalize to percentage of initial (100%)
                                                initial_bot_equity = bot_equity_df['equity'].iloc[0]
                                                bot_y_values = bot_equity_df['equity'] / initial_bot_equity * 100
                                                name_suffix = "% (norm)"
                                            else:
                                                # Use actual values
                                                bot_y_values = bot_equity_df['equity']
                                                name_suffix = "USDT"
                                                
                                            # Add line to chart - ensuring it only starts from the bot's actual start date
                                            # This is the key change to make each bot's line start at its deployment date
                                            # Improved line styling to match screenshot
                                            bot_fig.add_trace(
                                                go.Scatter(
                                                    x=bot_equity_df.index,
                                                    y=bot_y_values,
                                                    name=f"Bot {bot_id} ({interval}) - {name_suffix}",
                                                    line=dict(
                                                        color=colors[i % len(colors)],
                                                        width=2,
                                                        shape='spline',  # Smooth curve
                                                        smoothing=0.3    # Moderate smoothing
                                                    ),
                                                    showlegend=True,
                                                    mode='lines'
                                                )
                                            )
                                            has_valid_bot_data = True
                                            
                            # If we didn't get any valid bot data from trades, try using the main equity curve
                            if not has_valid_bot_data and equity_curve is not None and not equity_curve.empty:
                                # Check if 'equity' column exists
                                if 'equity' in equity_curve.columns:
                                    # Prepare values based on normalization setting
                                    if st.session_state.normalize_equity_value:
                                        # Normalize to percentage of initial (100%)
                                        initial_equity = equity_curve['equity'].iloc[0]
                                        y_values = equity_curve['equity'] / initial_equity * 100
                                        name_suffix = "% (norm)"
                                    else:
                                        # Use actual values
                                        y_values = equity_curve['equity']
                                        name_suffix = "USDT"
                                    
                                    # Add overall equity curve as a fallback
                                    bot_fig.add_trace(
                                        go.Scatter(
                                            x=equity_curve.index,
                                            y=y_values,
                                            name=f"Combined Strategy - {name_suffix}",
                                            line=dict(color=colors[0], width=2)
                                        )
                                    )
                                    has_valid_bot_data = True
                    except Exception as e:
                        st.warning(f"Error processing bot performance data: {str(e)}")
                    
                    if not has_valid_bot_data:
                        st.info("No valid bot performance data found. Run the backtest to see actual bot performance.")
                        
                        # Create a placeholder chart with a message
                        placeholder_x = [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-03-01')]
                        placeholder_y = [1000, 1000]  # Flat line
                        
                        bot_fig.add_trace(
                            go.Scatter(
                                x=placeholder_x,
                                y=placeholder_y,
                                name="No Valid Bot Data",
                                line=dict(color='#cccccc', dash='dot'),
                                mode="lines"
                            )
                        )
                        
                else:
                    # Display a message when no bot data is available
                    st.info("No individual bot performance data available yet. Run the backtest to see actual bot performance.")
                    
                    # Create a placeholder chart with a message
                    placeholder_x = [pd.Timestamp('2025-01-01'), pd.Timestamp('2025-03-01')]
                    placeholder_y = [1000, 1000]  # Flat line
                    
                    bot_fig.add_trace(
                        go.Scatter(
                            x=placeholder_x,
                            y=placeholder_y,
                            name="No Bot Data Yet",
                            line=dict(color='#cccccc', dash='dot'),
                            mode="lines"
                        )
                    )
                
                # Format the bot comparison chart to match the screenshot
                bot_fig.update_layout(
                    title='Individual Bot Performance Comparison (Normalized to 100%)',
                    xaxis_title='Date',
                    yaxis_title='Percentage of Initial Capital (%)',
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    # Add light grid lines to match screenshot
                    plot_bgcolor='white',
                    yaxis=dict(
                        gridcolor='#e6e6e6',
                        zeroline=True,
                        zerolinecolor='#e6e6e6',
                        showline=True,
                        linewidth=1,
                        linecolor='#e6e6e6',
                    ),
                    xaxis=dict(
                        gridcolor='#e6e6e6',
                        showline=True,
                        linewidth=1,
                        linecolor='#e6e6e6',
                    )
                )
                
                # Update axes styling to match the screenshot
                bot_fig.update_xaxes(
                    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-03-01')],
                    autorange=True,
                    showgrid=True,
                    gridcolor='#e6e6e6',
                    tickfont=dict(size=10),
                    title_font=dict(size=12),
                    mirror=True,
                    linecolor='#e6e6e6',
                    zeroline=True,
                    zerolinecolor='#e6e6e6'
                )
                bot_fig.update_yaxes(
                    range=[90, 120],  # Better range for normalized values around 100%
                    autorange=True,
                    showgrid=True,
                    gridcolor='#e6e6e6',
                    tickfont=dict(size=10),
                    title_font=dict(size=12),
                    mirror=True,
                    linecolor='#e6e6e6'
                )
                
                st.plotly_chart(bot_fig, use_container_width=True, key="v7_bot_comparison")
                
                # Bot configuration summary - as in the original template
                st.subheader("Bot Configuration Summary")
                
                for bot in st.session_state.v7_bots_config:
                    st.markdown(f"""
                    <div class="param-container">
                        <strong>Bot {bot['bot_id']}</strong><br>
                        Start Date: {bot['start_date']}<br>
                        Start Time: {bot['start_time']}<br>
                        Operating Interval: {bot['interval']}<br>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Drawdown chart
                st.markdown("### Drawdown Chart")
                
                # Calculate drawdown values (simplified)
                equity_df['peak'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
                
                # Create drawdown figure
                drawdown_fig = go.Figure()
                
                # Add drawdown area
                drawdown_fig.add_trace(
                    go.Scatter(
                        x=equity_df.index,
                        y=equity_df['drawdown'],
                        name='Drawdown',
                        fill='tozeroy',
                        line=dict(color='#d62728')
                    )
                )
                
                # Format the drawdown chart
                drawdown_fig.update_layout(
                    title='Strategy Drawdown',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    template='plotly_white',
                    yaxis=dict(
                        autorange='reversed',  # Invert y-axis so drawdowns go down
                    ),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(drawdown_fig, use_container_width=True, key="v7_drawdown_chart")
                
                # Monthly returns heatmap - matching backtest.py format
                st.markdown("### Monthly Returns")
                
                # Use the same implementation as in the Detailed Analysis tab
                if ('v7_backtest_results' in st.session_state and 
                    st.session_state.v7_backtest_results is not None and 
                    'equity_curve' in st.session_state.v7_backtest_results):
                    
                    # Use actual equity curve data from the backtest
                    results_equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                    
                    # Calculate monthly returns from the equity data
                    # Direct calculation for return if data is short (1 month or less)
                    if len(results_equity_df) >= 2:
                        # Calculate overall return directly
                        initial_value = results_equity_df['equity'].iloc[0]
                        final_value = results_equity_df['equity'].iloc[-1]
                        total_return_pct = (final_value / initial_value - 1) * 100
                        
                        # Get the month and year for the current data
                        start_date = results_equity_df.index[0]
                        end_date = results_equity_df.index[-1]
                        
                        # Always create at least one monthly return even if within the same month
                        # First check if we have more than one month
                        if (end_date.year > start_date.year) or (end_date.month > start_date.month):
                            # Use standard monthly resampling if spanning multiple months
                            monthly_equity = results_equity_df['equity'].resample('ME').last()
                            # Calculate the monthly percentage change
                            monthly_pct_change = monthly_equity.pct_change() * 100
                        else:
                            # If all data is within one month, create direct return as a monthly value
                            # Create a series with one value for the current month
                            current_month = end_date.strftime('%Y-%m')
                            monthly_equity = pd.Series(
                                [initial_value, final_value],
                                index=[pd.to_datetime(f"{current_month}-01"), end_date]
                            )
                            # Create a single monthly return
                            monthly_pct_change = pd.Series([total_return_pct], index=[end_date])
                    else:
                        # Not enough data for calculations
                        monthly_equity = pd.Series()
                        monthly_pct_change = pd.Series()
                    
                    # Create a DataFrame suitable for the heatmap
                    monthly_returns = []
                    
                    for date, value in monthly_pct_change.items():
                        if not pd.isna(value):  # Skip the first value, which is NaN
                            monthly_returns.append({
                                'year': date.year,
                                'month': date.strftime('%b'),
                                'return': value
                            })
                    
                    # Only continue if we have monthly returns data
                    if monthly_returns:
                        monthly_df = pd.DataFrame(monthly_returns)
                        monthly_pivot = monthly_df.pivot(index='year', columns='month', values='return')
                    else:
                        # Calculate overall return if we have no monthly data but have equity data
                        if len(results_equity_df) >= 2:
                            initial_equity = results_equity_df['equity'].iloc[0]
                            final_equity = results_equity_df['equity'].iloc[-1]
                            total_return_pct = (final_equity / initial_equity - 1) * 100
                            
                            # Create a single month return
                            current_date = results_equity_df.index[-1]
                            month_str = current_date.strftime('%b')
                            year = current_date.year
                            
                            # Create a DataFrame with a single value
                            monthly_df = pd.DataFrame([{
                                'year': year, 
                                'month': month_str, 
                                'return': total_return_pct
                            }])
                            monthly_pivot = monthly_df.pivot(index='year', columns='month', values='return')
                        else:
                            st.info("Not enough data to calculate returns.")
                            monthly_pivot = None
                else:
                    # If no backtest results, just show a message
                    st.info("No monthly return data available yet. Run the backtest to see monthly returns.")
                    monthly_pivot = None
                
                # Create heatmap using plotly
                if monthly_pivot is not None and not monthly_pivot.empty and len(monthly_pivot) > 0:
                    # Ensure data is in the right format with numeric values
                    try:
                        # Check if values are valid
                        if np.all(np.isfinite(monthly_pivot.values)):
                            monthly_fig = go.Figure(data=go.Heatmap(
                                z=monthly_pivot.values,
                                x=monthly_pivot.columns,
                                y=monthly_pivot.index,
                                colorscale='RdBu',
                                zmid=0,
                                text=[[f"{val:.2f}%" for val in row] for row in monthly_pivot.values],
                                texttemplate="%{text}",
                                colorbar=dict(title='Return %')
                            ))
                        else:
                            # Create empty figure for invalid data
                            monthly_fig = go.Figure()
                            st.info("No valid monthly return data to display.")
                    except (ValueError, TypeError) as e:
                        # Create empty figure if error occurs
                        monthly_fig = go.Figure()
                        st.info(f"Could not create monthly returns chart: {str(e)}")
                else:
                    # Create empty figure for empty data
                    monthly_fig = go.Figure()
                    st.info("No monthly return data available yet.")
                
                monthly_fig.update_layout(
                    title='Monthly Returns (%)',
                    xaxis_title='Month',
                    yaxis_title='Year',
                    template='plotly_white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                # Add a default empty heatmap if there's nothing to show
                if len(monthly_fig.data) == 0:
                    # Default data in case of empty figure
                    default_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    default_years = [2024, 2025]
                    default_z = [[0 for _ in range(len(default_months))] for _ in range(len(default_years))]
                    default_text = [[f"0.00%" for _ in range(len(default_months))] for _ in range(len(default_years))]
                    
                    monthly_fig.add_trace(go.Heatmap(
                        z=default_z,
                        x=default_months,
                        y=default_years,
                        colorscale='RdBu',
                        zmid=0,
                        text=default_text,
                        texttemplate="%{text}",
                        colorbar=dict(title='Return %'),
                        hoverinfo="text"
                    ))
                    
                    # Add note about empty data
                    monthly_fig.add_annotation(
                        text="No return data yet",
                        x=0.5, 
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16)
                    )
                
                st.plotly_chart(monthly_fig, use_container_width=True, key="v7_monthly_returns")
            
            with results_tab3:
                st.markdown("### Trade Analysis")
                
                # Trade statistics - calculated from actual trade data
                trade_analysis = {
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0
                }
                
                # Check if we have actual trades data to calculate trade statistics
                if ('v7_backtest_results' in st.session_state and 
                    st.session_state.v7_backtest_results is not None and 
                    'trades' in st.session_state.v7_backtest_results):
                    
                    trades = st.session_state.v7_backtest_results['trades']
                    
                    # Calculate win rate if we have profit_pct data
                    if 'profit_pct' in trades.columns and len(trades) > 0:
                        # Count profitable trades (profit > 0)
                        profitable_trades = trades[trades['profit_pct'] > 0]
                        if len(trades) > 0:
                            trade_analysis["win_rate"] = (len(profitable_trades) / len(trades)) * 100
                        
                        # Calculate average profit
                        if 'profit_pct' in trades.columns:
                            profit_values = trades['profit_pct'].dropna()
                            if len(profit_values) > 0:
                                trade_analysis["avg_profit"] = profit_values.mean()
                            
                            # Find best and worst trades
                            if len(profit_values) > 0:
                                trade_analysis["best_trade"] = profit_values.max()
                                trade_analysis["worst_trade"] = profit_values.min()
                
                # Display trade metrics in 4 columns exactly like backtest.py
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Win Rate", f"{trade_analysis.get('win_rate', 0):.1f}%")
                
                with col2:
                    st.metric("Avg. Profit", f"{trade_analysis.get('avg_profit', 0):.2f}%")
                
                with col3:
                    st.metric("Best Trade", f"{trade_analysis.get('best_trade', 0):.2f}%")
                
                with col4:
                    st.metric("Worst Trade", f"{trade_analysis.get('worst_trade', 0):.2f}%")
                
                # Create trades chart to show where trades happened on the price chart
                st.markdown("### Price Chart with Trade Markers")
                
                # Create price and trades chart
                trades_fig = go.Figure()
                
                # Check if we have actual price data from the backtest
                if ('v7_backtest_results' in st.session_state and 
                    st.session_state.v7_backtest_results is not None and 
                    'equity_curve' in st.session_state.v7_backtest_results and
                    'close' in st.session_state.v7_backtest_results['equity_curve'].columns):
                    
                    # Use real price data from backtest
                    price_df = st.session_state.v7_backtest_results['equity_curve']
                    
                    # Make sure the price data index and values are valid
                    if (price_df is not None and not price_df.empty and 
                        not price_df.index.isnull().any() and 
                        not price_df['close'].isnull().all() and 
                        not np.isinf(price_df['close']).any()):
                        
                        # Add price line to chart using actual data
                        trades_fig.add_trace(
                            go.Scatter(
                                x=price_df.index,
                                y=price_df['close'],
                                name='BTC Price',
                                line=dict(color='#1f77b4', width=1)
                            )
                        )
                    else:
                        st.warning("Price data contains invalid values. Cannot create price chart.")
                else:
                    # Use actual price data from the dataset
                    # Add price line to chart using actual data if df is valid
                    if 'df' in locals() and df is not None and not df.empty and 'close' in df.columns:
                        # Make sure the dataset price data index and values are valid
                        if (not df.index.isnull().any() and 
                            not df['close'].isnull().all() and 
                            not np.isinf(df['close']).any()):
                            
                            trades_fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['close'],
                                    name='BTC Price',
                                    line=dict(color='#1f77b4', width=1)
                                )
                            )
                        else:
                            st.warning("Dataset price data contains invalid values. Cannot create price chart.")
                    else:
                        st.info("No valid price data available for chart.")
                
                # Check if we have actual trades data from the backtest
                if ('v7_backtest_results' in st.session_state and 
                    st.session_state.v7_backtest_results is not None and 
                    'trades' in st.session_state.v7_backtest_results):
                    
                    # Use actual trades data from backtest
                    trades_df = st.session_state.v7_backtest_results['trades'].copy()
                    
                    # Make sure required columns exist
                    if 'timestamp' not in trades_df.columns:
                        if 'date' in trades_df.columns:
                            trades_df['timestamp'] = trades_df['date']
                        elif 'buy_date' in trades_df.columns:
                            trades_df['timestamp'] = trades_df['buy_date']
                    
                    if 'action' not in trades_df.columns and 'type' in trades_df.columns:
                        trades_df['action'] = trades_df['type'].str.upper()
                        
                    if 'bot_number' not in trades_df.columns and 'bot_id' in trades_df.columns:
                        trades_df['bot_number'] = trades_df['bot_id']
                        
                    # Parse timestamp if needed and ensure it exists
                    if 'timestamp' in trades_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                    else:
                        # Add a default timestamp using the index
                        trades_df['timestamp'] = pd.Timestamp('2025-03-19')
                else:
                    # Instead of using placeholder trade data, we'll display an empty trades DataFrame
                    # with the same structure as the actual trades data
                    st.info("No trades data available yet. Run the backtest to see trades.")
                    
                    # Create an empty DataFrame with the expected columns
                    trades_df = pd.DataFrame(columns=[
                        'timestamp', 'bot_id', 'action', 'price', 'amount_btc', 
                        'amount_usd', 'profit_pct', 'holding_period', 'bot_number'
                    ])
                    
                    # Add a single row to use the actual data range if available
                    if 'df' in locals() and df is not None and not df.empty:
                        try:
                            # Add a single empty row with the first date from the dataset
                            # This ensures the chart has the right date range
                            # Make sure the index is valid before using it
                            if not df.index.empty and pd.notna(df.index.min()):
                                empty_row = {
                                    'timestamp': df.index.min(),
                                    'bot_id': 0,
                                    'action': 'NONE',
                                    'price': df['close'].iloc[0] if 'close' in df.columns and len(df) > 0 else 0.0,
                                    'amount_btc': 0.0,
                                    'amount_usd': 0.0,
                                    'profit_pct': 0.0,
                                    'holding_period': 0,
                                    'bot_number': 0
                                }
                                trades_df = pd.DataFrame([empty_row])
                            else:
                                # Create an empty placeholder with today's date if index is not valid
                                empty_row = {
                                    'timestamp': pd.Timestamp('2025-03-19'),
                                    'bot_id': 0,
                                    'action': 'NONE',
                                    'price': 0.0,
                                    'amount_btc': 0.0,
                                    'amount_usd': 0.0,
                                    'profit_pct': 0.0,
                                    'holding_period': 0,
                                    'bot_number': 0
                                }
                                trades_df = pd.DataFrame([empty_row])
                        except Exception as e:
                            st.warning(f"Could not create placeholder trade data: {str(e)}")
                
                # Debug the trades dataframe to understand available columns and values
                # Debug info - commented out for production
                # st.write("Available columns in trades_df:", list(trades_df.columns))
                
                # Prepare action column with consistent values 
                if 'action' in trades_df.columns:
                    # Ensure consistent capitalization
                    trades_df['action'] = trades_df['action'].str.upper()
                elif 'type' in trades_df.columns:
                    # Create action column from type
                    trades_df['action'] = trades_df['type'].str.upper()
                
                # Add Buy markers - use case-insensitive comparison with proper handling
                # First, determine available timestamp column
                timestamp_column = None
                for col in ['timestamp', 'date', 'buy_date', 'sell_date', 'trade_date']:
                    if col in trades_df.columns:
                        timestamp_column = col
                        break
                
                if timestamp_column is not None:
                    # Add debug info for timestamp column
                    st.session_state['debug_timestamp_column'] = timestamp_column
                    
                    # Create unified action field using any available action column
                    has_action = False
                    for action_col in ['action', 'type']:
                        if action_col in trades_df.columns:
                            trades_df['unified_action'] = trades_df[action_col].astype(str).str.upper()
                            has_action = True
                            break
                    
                    if not has_action and 'buy_price' in trades_df.columns:
                        # If we don't have an action column but have buy_price, infer action
                        trades_df['unified_action'] = 'BUY'
                        if 'sell_price' in trades_df.columns:
                            # Mark rows with non-null sell_price as SELL
                            sell_mask = pd.notna(trades_df['sell_price'])
                            trades_df.loc[sell_mask, 'unified_action'] = 'SELL'
                    
                    # Process BUY trades
                    try:
                        # Filter buy trades
                        buy_trades = trades_df[trades_df['unified_action'] == 'BUY'].copy()
                        
                        if len(buy_trades) > 0 and timestamp_column in buy_trades.columns and 'price' in buy_trades.columns:
                            # Verify we have valid data to plot
                            buy_trades = buy_trades.dropna(subset=[timestamp_column, 'price'])
                            
                            if len(buy_trades) > 0:
                                # Convert timestamp to datetime if it's not already
                                buy_trades[timestamp_column] = pd.to_datetime(buy_trades[timestamp_column])
                                
                                # Check that timestamp and price values are valid (not NaN or infinite)
                                valid_timestamp = pd.notna(buy_trades[timestamp_column]).all()
                                valid_price = pd.notna(buy_trades['price']).all() and np.isfinite(buy_trades['price']).all()
                                
                                if valid_timestamp and valid_price:
                                    trades_fig.add_trace(
                                        go.Scatter(
                                            x=buy_trades[timestamp_column],
                                            y=buy_trades['price'],
                                            mode='markers',
                                            name='Buy',
                                            marker=dict(
                                                symbol='triangle-up',
                                                size=10,  # Slightly larger for visibility
                                                color='green',
                                                line=dict(width=1, color='darkgreen')
                                            ),
                                            hovertemplate='%{x}<br>Buy @ $%{y:.2f}<extra></extra>'
                                        )
                                    )
                    except Exception as e:
                        st.error(f"Error adding buy markers: {str(e)}")
                
                    # Process SELL trades
                    try:
                        # Filter sell trades
                        sell_trades = trades_df[trades_df['unified_action'] == 'SELL'].copy()
                        
                        if len(sell_trades) > 0 and timestamp_column in sell_trades.columns and 'price' in sell_trades.columns:
                            # Verify we have valid data to plot
                            sell_trades = sell_trades.dropna(subset=[timestamp_column, 'price'])
                            
                            if len(sell_trades) > 0:
                                # Convert timestamp to datetime if it's not already
                                sell_trades[timestamp_column] = pd.to_datetime(sell_trades[timestamp_column])
                                
                                # Check that timestamp and price values are valid
                                valid_timestamp = pd.notna(sell_trades[timestamp_column]).all()
                                valid_price = pd.notna(sell_trades['price']).all() and np.isfinite(sell_trades['price']).all()
                                
                                if valid_timestamp and valid_price:
                                    # Create text for hover tooltip
                                    if 'profit_pct' in sell_trades.columns:
                                        text_values = sell_trades['profit_pct'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                                    else:
                                        text_values = ["N/A"] * len(sell_trades)
                                        
                                    trades_fig.add_trace(
                                        go.Scatter(
                                            x=sell_trades[timestamp_column],
                                            y=sell_trades['price'],
                                            mode='markers',
                                            name='Sell',
                                            marker=dict(
                                                symbol='triangle-down',
                                                size=10,  # Slightly larger for visibility
                                                color='red',
                                                line=dict(width=1, color='darkred')
                                            ),
                                            hovertemplate='%{x}<br>Sell @ $%{y:.2f}<br>Profit: %{text}%<extra></extra>',
                                            text=text_values
                                        )
                                    )
                    except Exception as e:
                        st.error(f"Error adding sell markers: {str(e)}")
                else:
                    st.warning("Trade visualization: No timestamp column found in trades data.")
                
                # Format chart
                trades_fig.update_layout(
                    title='BTC Price with Trade Markers',
                    xaxis_title='Date',
                    yaxis_title='Price (USDT)',
                    template='plotly_white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                # Fix for "Infinite extent" warnings - ensure xaxis and yaxis have explicit ranges
                # Using fixed ranges without autorange to prevent warnings
                start_date = pd.Timestamp('2025-01-01')
                end_date = pd.Timestamp('2025-03-15')
                min_price = 10000
                max_price = 100000
                
                # Use actual data ranges if available
                if 'df' in locals() and df is not None and not df.empty:
                    if isinstance(df.index, pd.DatetimeIndex):
                        start_date = df.index.min()
                        end_date = df.index.max()
                    
                    if 'close' in df.columns:
                        min_price = max(1000, df['close'].min() * 0.8)  # Ensure positive value with buffer
                        max_price = df['close'].max() * 1.2  # Add buffer
                
                trades_fig.update_xaxes(
                    # Set explicit range and avoid auto-ranging which can cause infinite extent warnings
                    range=[start_date, end_date],
                    rangemode="normal",  # Use normal range mode instead of "tozero"
                    autorange=False  # Disable autoranging to use our explicit range
                )
                trades_fig.update_yaxes(
                    # Set explicit range and avoid auto-ranging which can cause infinite extent warnings
                    range=[min_price, max_price],
                    rangemode="normal",  # Use normal range mode instead of "tozero"
                    autorange=False  # Disable autoranging to use our explicit range
                )
                
                # If no data or trades are available, add a default empty chart message
                if len(trades_fig.data) == 0:
                    # Add a default empty line
                    trades_fig.add_trace(
                        go.Scatter(
                            x=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-03-01')],
                            y=[50000, 50000],
                            name="No Price Data",
                            line=dict(color='#cccccc', dash='dot'),
                            mode="lines"
                        )
                    )
                    
                    # Add annotation message
                    trades_fig.add_annotation(
                        text="No price or trade data available yet",
                        x=0.5, 
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16)
                    )
                
                st.plotly_chart(trades_fig, use_container_width=True, key="v7_trade_markers")
                
                # Display trade history table with proper formatting
                st.markdown("### Trade History")
                
                # Format trades dataframe for display
                display_trades_df = trades_df.copy()
                
                # Ensure required columns exist
                required_columns = ['price', 'amount_usd', 'profit_pct', 'bot_number']
                for col in required_columns:
                    if col not in display_trades_df.columns:
                        display_trades_df[col] = None
                
                # Format price with $ sign (safely)
                if 'price' in display_trades_df.columns and not display_trades_df['price'].isnull().all():
                    display_trades_df['price'] = display_trades_df['price'].apply(
                        lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"
                    )
                
                # Format amount_usd with $ sign (safely)
                if 'amount_usd' in display_trades_df.columns and not display_trades_df['amount_usd'].isnull().all():
                    display_trades_df['amount_usd'] = display_trades_df['amount_usd'].apply(
                        lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"
                    )
                
                # Format profit_pct with % sign where applicable (safely)
                if 'profit_pct' in display_trades_df.columns:
                    display_trades_df['profit_pct'] = display_trades_df['profit_pct'].apply(
                        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                    )
                
                # Format bot_number to be more descriptive (safely)
                if 'bot_number' in display_trades_df.columns and not display_trades_df['bot_number'].isnull().all():
                    # Convert to string first to handle various input types safely
                    display_trades_df['bot_number'] = display_trades_df['bot_number'].astype(str)
                    # Then try to convert to integer format for displaying
                    try:
                        display_trades_df['bot_number'] = display_trades_df['bot_number'].astype(int).map(lambda x: f"Bot #{x}")
                    except (ValueError, TypeError):
                        # If conversion fails, use the string values as is
                        display_trades_df['bot_number'] = "Bot #0"
                
                # Re-order and rename columns for better presentation
                renamed_trades_df = display_trades_df.rename(columns={
                    'timestamp': 'Date & Time',
                    'action': 'Action',
                    'price': 'Price',
                    'amount_btc': 'BTC Amount',
                    'amount_usd': 'USD Value',
                    'profit_pct': 'Profit',
                    'holding_period': 'Days Held',
                    'bot_number': 'Bot'
                })
                
                # Display formatted trades table
                st.dataframe(renamed_trades_df, use_container_width=True)
                
                # Option to download trades as CSV - original data without formatting
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    label="Download Trades as CSV",
                    data=csv,
                    file_name="v7_strategy_trades.csv",
                    mime="text/csv"
                )
                
            with results_tab4:
                # Detailed Analysis tab - exactly matching backtest.py format and content
                st.markdown("### Detailed Performance Analysis")
                
                # Date range selection for the detailed analysis
                detailed_col1, detailed_col2 = st.columns(2)
                
                with detailed_col1:
                    # Use the actual date range from the dataset
                    min_date = df.index.min().date()
                    max_date = df.index.max().date()
                    
                    analysis_start = st.date_input(
                        "Analysis Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="v7_analysis_start_date"
                    )
                
                with detailed_col2:
                    analysis_end = st.date_input(
                        "Analysis End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="v7_analysis_end_date"
                    )
                
                # Filter analysis by date - for display only
                analysis_date_range = f"Analysis Period: {analysis_start} to {analysis_end}"
                st.info(analysis_date_range)
                
                # Equity vs BTC price chart - Exactly like backtest.py
                st.markdown("#### Equity vs BTC Price")
                
                try:
                    # Create a figure with dual y-axes - matching backtest.py
                    equity_fig = go.Figure()
                    
                    # Check if we have backtest results
                    if ('v7_backtest_results' in st.session_state and 
                        st.session_state.v7_backtest_results is not None and 
                        'equity_curve' in st.session_state.v7_backtest_results):
                        
                        # Use actual equity curve data from backtest
                        equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                        
                        # Add strategy equity to chart using actual data
                        if 'equity' in equity_df.columns:
                            equity_fig.add_trace(
                                go.Scatter(
                                    x=equity_df.index,
                                    y=equity_df['equity'],
                                    name='Strategy Equity',
                                    line=dict(color='#1f77b4', width=2)
                                )
                            )
                        
                        # Initialize flag to track if we can add price chart
                        continue_with_chart = True
                        
                        # Check if we have price data in the equity curve or if we need to add it
                        if 'close' not in equity_df.columns:
                            # We need to merge price data from our dataset into the equity curve
                            if 'close' in df.columns:
                                # Make a copy of the price data aligned with equity curve dates
                                price_data = df[['close']].copy()
                                
                                # Ensure both dataframes have datetime index
                                if not isinstance(price_data.index, pd.DatetimeIndex):
                                    price_data.index = pd.to_datetime(price_data.index)
                                
                                if not isinstance(equity_df.index, pd.DatetimeIndex):
                                    equity_df.index = pd.to_datetime(equity_df.index)
                                    
                                # Merge price data into equity dataframe (right join to keep equity dates)
                                # First resample price data to match equity curve frequency
                                equity_freq = pd.infer_freq(equity_df.index)
                                if equity_freq:
                                    price_data = price_data.resample(equity_freq).last()
                                    
                                # Then reindex price data to match equity dates exactly
                                price_data = price_data.reindex(index=equity_df.index, method='ffill')
                                equity_df['close'] = price_data['close']
                            else:
                                # Fallback to original dataset
                                st.warning("Price data not available in expected format")
                                # Skip adding the price data trace
                                continue_with_chart = False
                        
                        # Add BTC price chart only if we have the data
                        if continue_with_chart and 'close' in equity_df.columns:
                            equity_fig.add_trace(
                                go.Scatter(
                                    x=equity_df.index,
                                    y=equity_df['close'],
                                    name='BTC Price',
                                    line=dict(color='#ff7f0e', width=1, dash='dash'),
                                    yaxis='y2'
                                )
                            )
                    else:
                        # Use actual price data from the dataset for both axes
                        # Create a temporary dataframe with the actual price data
                        temp_df = df.copy()
                        
                        # Add a message about no backtest results
                        st.info("No backtest results available yet. Run the backtest to see actual equity curve.")
                        
                        # Add BTC price to chart using actual data
                        equity_fig.add_trace(
                            go.Scatter(
                                x=temp_df.index,
                                y=temp_df['close'],
                                name='BTC Price',
                                line=dict(color='#ff7f0e', width=1, dash='dash'),
                                yaxis='y2'
                            )
                        )
                    
                    # Update layout with secondary y-axis - exactly like backtest.py
                    equity_fig.update_layout(
                        title='Strategy Equity vs. BTC Price',
                        xaxis_title='Date',
                        yaxis_title='Equity (USDT)',
                        yaxis2=dict(
                            title='BTC Price (USDT)',
                            overlaying='y',
                            side='right'
                        ),
                        template='plotly_white',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(equity_fig, use_container_width=True, key="v7_detailed_equity_vs_btc")
                    
                    # Create Cash vs Position Allocation chart - exactly like backtest.py
                    st.markdown("#### Cash vs. Position Allocation")
                    
                    # Check if we have backtest results with cash and position data
                    if ('v7_backtest_results' in st.session_state and 
                        st.session_state.v7_backtest_results is not None and 
                        'equity_curve' in st.session_state.v7_backtest_results and
                        'cash' in st.session_state.v7_backtest_results['equity_curve'].columns and
                        'position_value' in st.session_state.v7_backtest_results['equity_curve'].columns):
                        
                        # Use actual allocation data from backtest
                        allocation_fig = go.Figure()
                        
                        # Add cash area
                        allocation_fig.add_trace(
                            go.Scatter(
                                x=equity_df.index,
                                y=equity_df['cash'],
                                name='Cash',
                                stackgroup='one',
                                line=dict(color='#1f77b4')
                            )
                        )
                        
                        # Add position area
                        allocation_fig.add_trace(
                            go.Scatter(
                                x=equity_df.index,
                                y=equity_df['position_value'],
                                name='BTC Position',
                                stackgroup='one',
                                line=dict(color='#ff7f0e')
                            )
                        )
                    else:
                        # Display message about missing data
                        st.info("No cash and position allocation data available yet. Run the backtest to see allocation data.")
                        
                        # Create an empty figure
                        allocation_fig = go.Figure()
                    
                    # Format chart
                    allocation_fig.update_layout(
                        title='Cash vs. Position Allocation',
                        xaxis_title='Date',
                        yaxis_title='Value (USDT)',
                        template='plotly_white',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(allocation_fig, use_container_width=True, key="v7_allocation_chart")
                    
                    # BTC Holdings Over Time - simplified implementation
                    st.markdown("#### BTC Holdings Over Time")
                    
                    # Create a figure for BTC holdings
                    holdings_fig = go.Figure()
                    
                    # Flag to track if we have valid holdings data
                    has_holdings_data = False
                    
                    try:
                        # Check if we have backtest results with data
                        if ('v7_backtest_results' in st.session_state and 
                            st.session_state.v7_backtest_results is not None):
                            
                            # Method 1: Direct BTC holdings from equity curve
                            holdings_data = None
                            holdings_column = None
                            
                            if 'equity_curve' in st.session_state.v7_backtest_results:
                                equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                                
                                if 'btc_holdings' in equity_df.columns:
                                    holdings_data = equity_df
                                    holdings_column = 'btc_holdings'
                                elif 'holdings' in equity_df.columns:
                                    holdings_data = equity_df
                                    holdings_column = 'holdings'
                                elif 'btc_amount' in equity_df.columns:
                                    holdings_data = equity_df
                                    holdings_column = 'btc_amount'
                            
                            # Check if we found a valid column with holdings data
                            if holdings_data is not None and holdings_column is not None:
                                if not holdings_data[holdings_column].isnull().all():
                                    holdings_fig.add_trace(
                                        go.Scatter(
                                            x=holdings_data.index,
                                            y=holdings_data[holdings_column],
                                            name='BTC Holdings',
                                            line=dict(color='#2ca02c', width=2),
                                            fill='tozeroy'
                                        )
                                    )
                                    has_holdings_data = True
                            
                            # Method 2: Calculate holdings from position_value and close
                            if not has_holdings_data and 'equity_curve' in st.session_state.v7_backtest_results:
                                equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                                
                                if ('position_value' in equity_df.columns and 
                                    'close' in equity_df.columns and 
                                    not equity_df['position_value'].isnull().all()):
                                    
                                    # Avoid division by zero
                                    close_prices = equity_df['close'].copy()
                                    close_prices = close_prices.replace(0, 1)  # Replace zeros with ones
                                    
                                    # Calculate derived holdings
                                    derived_holdings = equity_df['position_value'] / close_prices
                                    derived_holdings = derived_holdings.fillna(0)
                                    derived_holdings = derived_holdings.replace([np.inf, -np.inf], 0)
                                    
                                    if not derived_holdings.isnull().all() and derived_holdings.sum() > 0:
                                        holdings_fig.add_trace(
                                            go.Scatter(
                                                x=equity_df.index,
                                                y=derived_holdings,
                                                name='BTC Holdings (Calculated)',
                                                line=dict(color='#2ca02c', width=2),
                                                fill='tozeroy'
                                            )
                                        )
                                        has_holdings_data = True
                            
                            # Method 3: Calculate from cash and equity
                            if not has_holdings_data and 'equity_curve' in st.session_state.v7_backtest_results:
                                equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                                
                                if ('cash' in equity_df.columns and 
                                    'equity' in equity_df.columns and 
                                    'close' in equity_df.columns):
                                    
                                    # Avoid division by zero
                                    close_prices = equity_df['close'].copy()
                                    close_prices = close_prices.replace(0, 1)  # Replace zeros with ones
                                    
                                    # Calculate position value and then holdings
                                    position_values = equity_df['equity'] - equity_df['cash']
                                    derived_holdings = position_values / close_prices
                                    derived_holdings = derived_holdings.where(derived_holdings >= 0, 0)  # No negative values
                                    derived_holdings = derived_holdings.fillna(0)
                                    
                                    if not derived_holdings.isnull().all() and derived_holdings.sum() > 0:
                                        holdings_fig.add_trace(
                                            go.Scatter(
                                                x=equity_df.index,
                                                y=derived_holdings,
                                                name='BTC Holdings (From Cash/Equity)',
                                                line=dict(color='#2ca02c', width=2),
                                                fill='tozeroy'
                                            )
                                        )
                                        has_holdings_data = True
                            
                            # Method 4: Reconstruct from trades data
                            if not has_holdings_data and 'trades' in st.session_state.v7_backtest_results:
                                trades_df = st.session_state.v7_backtest_results['trades']
                                
                                if not trades_df.empty and 'buy_date' in trades_df.columns and 'quantity' in trades_df.columns and 'type' in trades_df.columns:
                                    # Make sure buy_date is datetime
                                    if trades_df['buy_date'].dtype != 'datetime64[ns]':
                                        trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
                                    
                                    # Sort trades chronologically
                                    trades_df = trades_df.sort_values('buy_date')
                                    
                                    # Get date range spanning all trades
                                    min_date = trades_df['buy_date'].min()
                                    max_date = trades_df['buy_date'].max()
                                    
                                    if pd.notna(min_date) and pd.notna(max_date):
                                        # Create daily date index
                                        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
                                        
                                        # Initialize holdings with zeros
                                        holdings_df = pd.DataFrame(index=date_range)
                                        holdings_df['holdings'] = 0.0
                                        
                                        # Accumulate holdings
                                        total_holdings = 0.0
                                        
                                        for _, trade in trades_df.iterrows():
                                            if trade['type'].lower() == 'buy':
                                                try:
                                                    total_holdings += float(trade['quantity'])
                                                except (ValueError, TypeError):
                                                    # Skip invalid quantities
                                                    continue
                                            elif trade['type'].lower() == 'sell':
                                                try:
                                                    total_holdings -= float(trade['quantity'])
                                                except (ValueError, TypeError):
                                                    # Skip invalid quantities
                                                    continue
                                            
                                            # Set holdings from this date forward
                                            trade_date = pd.to_datetime(trade['buy_date']) 
                                            mask = holdings_df.index >= trade_date
                                            holdings_df.loc[mask, 'holdings'] = total_holdings
                                        
                                        # Add to chart if we have data
                                        if not holdings_df['holdings'].isnull().all() and holdings_df['holdings'].sum() > 0:
                                            holdings_fig.add_trace(
                                                go.Scatter(
                                                    x=holdings_df.index,
                                                    y=holdings_df['holdings'],
                                                    name='BTC Holdings (from Trades)',
                                                    line=dict(color='#2ca02c', width=2),
                                                    fill='tozeroy'
                                                )
                                            )
                                            has_holdings_data = True
                    except Exception as e:
                        st.error(f"Error calculating BTC holdings: {str(e)}")
                                    
                    # If no data was found, show a placeholder
                    if not has_holdings_data:
                        st.info("No BTC holdings data available. Run the backtest to see holdings over time.")
                        
                        # Create a placeholder with the current date range for better visuals
                        current_date = pd.Timestamp.now()
                        start_date = current_date - pd.Timedelta(days=90)
                        end_date = current_date
                        
                        # Empty chart with placebo data points
                        holdings_fig.add_trace(
                            go.Scatter(
                                x=[start_date, end_date],
                                y=[0, 0],
                                name='No Holdings Data',
                                line=dict(color='#cccccc', dash='dot'),
                                mode='lines'
                            )
                        )
                    
                    # Format the chart
                    holdings_fig.update_layout(
                        title='BTC Holdings Over Time',
                        xaxis_title='Date',
                        yaxis_title='BTC Amount',
                        template='plotly_white',
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    # Set explicit ranges to avoid infinite extent warnings
                    
                    # Get price data date range if available
                    if 'df' in locals() and df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
                        x_min = df.index.min()
                        x_max = df.index.max()
                        
                        holdings_fig.update_xaxes(
                            range=[x_min, x_max]
                        )
                    else:
                        # Fallback to current date range
                        curr_date = pd.Timestamp.now()
                        holdings_fig.update_xaxes(
                            range=[curr_date - pd.Timedelta(days=90), curr_date]
                        )
                    
                    # Set y-axis range based on data (if available) or default
                    if has_holdings_data:
                        # Try to get maximum y value from the traces
                        max_y = 0
                        for trace in holdings_fig.data:
                            if hasattr(trace, 'y') and len(trace.y) > 0:
                                trace_max = max([y for y in trace.y if y is not None and not np.isnan(y)], default=0)
                                max_y = max(max_y, trace_max)
                        
                        # Set range with a small buffer (10% extra)
                        if max_y > 0:
                            holdings_fig.update_yaxes(
                                range=[0, max_y * 1.1]
                            )
                        else:
                            holdings_fig.update_yaxes(
                                range=[0, 0.1]
                            )
                    else:
                        holdings_fig.update_yaxes(
                            range=[0, 0.1]
                        )
                    
                    # Display the chart
                    st.plotly_chart(holdings_fig, use_container_width=True, key="v7_btc_holdings_chart")
                    
                    # Return Distribution - exactly like in backtest.py
                    st.markdown("#### Return Distribution")
                    
                    # Check if we have backtest results with equity data
                    if ('v7_backtest_results' in st.session_state and 
                        st.session_state.v7_backtest_results is not None and 
                        'equity_curve' in st.session_state.v7_backtest_results and
                        'equity' in st.session_state.v7_backtest_results['equity_curve'].columns):
                        
                        # Get the equity curve data
                        results_equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                        
                        # Calculate daily returns
                        results_equity_df['daily_return_pct'] = results_equity_df['equity'].pct_change() * 100
                        
                        # Filter out first row (NaN) and any other NaN values
                        daily_returns = results_equity_df['daily_return_pct'].dropna().tolist()
                        
                        if len(daily_returns) > 1:  # Need at least 2 points for a distribution
                            # Create return distribution histogram
                            returns_fig = go.Figure()
                            
                            # Add histogram of returns
                            returns_fig.add_trace(
                                go.Histogram(
                                    x=daily_returns,
                                    name='Daily Returns',
                                    nbinsx=30,
                                    histnorm='probability',
                                    marker=dict(color='#1f77b4')
                                )
                            )
                            
                            # Calculate normal distribution curve
                            mean_return = np.mean(daily_returns)
                            std_return = np.std(daily_returns)
                            
                            # Generate x values for the normal curve
                            x_curve = np.linspace(min(daily_returns), max(daily_returns), 100)
                            
                            # Calculate normal distribution values
                            from scipy.stats import norm
                            y_curve = norm.pdf(x_curve, mean_return, std_return)
                            
                            # Scale y values to match histogram probability density
                            bin_count = 30
                            y_curve_scaled = y_curve * (max(daily_returns) - min(daily_returns)) / bin_count
                            
                            # Add normal curve
                            returns_fig.add_trace(
                                go.Scatter(
                                    x=x_curve,
                                    y=y_curve_scaled,
                                    name='Normal Distribution',
                                    line=dict(color='#ff7f0e', width=2)
                                )
                            )
                            
                            # Format chart
                            returns_fig.update_layout(
                                title=f'Daily Return Distribution (Mean: {mean_return:.2f}%, Std Dev: {std_return:.2f}%)',
                                xaxis_title='Daily Return (%)',
                                yaxis_title='Probability',
                                template='plotly_white',
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            st.plotly_chart(returns_fig, use_container_width=True, key="v7_returns_distribution")
                        else:
                            st.info("Not enough daily return data to create return distribution.")
                    else:
                        st.info("No return distribution data available yet. Run the backtest to see return distribution.")
                    
                    # Monthly Returns Heatmap - matching backtest.py
                    st.markdown("#### Monthly Returns")
                    
                    # Check if we have backtest results
                    if ('v7_backtest_results' in st.session_state and 
                        st.session_state.v7_backtest_results is not None and 
                        'equity_curve' in st.session_state.v7_backtest_results):
                        
                        # Use actual equity curve data from the backtest
                        results_equity_df = st.session_state.v7_backtest_results['equity_curve'].copy()
                        
                        # Calculate monthly returns from the equity data
                        # Need to account for capital injections
                        
                        # First check if we have information about cash injections
                        if 'cash_injections' in results_equity_df.columns:
                            # If we have cash injections column, we need to adjust returns calculation
                            # Start with end-of-month equity values
                            monthly_equity = results_equity_df['equity'].resample('ME').last()
                            # Get monthly cash injections
                            monthly_injections = results_equity_df['cash_injections'].resample('ME').sum()
                            
                            # Calculate actual returns by removing effect of injections
                            monthly_pct_change = []
                            prev_equity = None
                            
                            for date, equity in monthly_equity.items():
                                if prev_equity is not None:
                                    # Get injections for this month
                                    injection = monthly_injections.get(date, 0)
                                    # Calculate actual growth excluding injections
                                    actual_growth = equity - prev_equity - injection
                                    # Calculate percentage based on previous equity
                                    pct_change = (actual_growth / prev_equity) * 100
                                    monthly_pct_change.append((date, pct_change))
                                prev_equity = equity
                            
                            # Convert to Series
                            if monthly_pct_change:
                                monthly_pct_change = pd.Series({date: val for date, val in monthly_pct_change})
                            else:
                                monthly_pct_change = pd.Series()
                        else:
                            # Standard calculation without accounting for injections
                            # Use 'ME' instead of deprecated 'M' frequency
                            monthly_equity = results_equity_df['equity'].resample('ME').last()
                            monthly_pct_change = monthly_equity.pct_change() * 100
                        
                        # Create a DataFrame suitable for the heatmap
                        monthly_returns = []
                        
                        for date, value in monthly_pct_change.items():
                            if not pd.isna(value):  # Skip the first value, which is NaN
                                # Using month number to ensure proper sorting
                                monthly_returns.append({
                                    'year': date.year,
                                    'month': date.month,  # Use month number for sorting
                                    'month_name': date.strftime('%b'),  # Keep month name for display
                                    'return': value
                                })
                        
                        # Only continue if we have monthly returns data
                        if monthly_returns:
                            monthly_df = pd.DataFrame(monthly_returns)
                            monthly_pivot = monthly_df.pivot(index='year', columns='month', values='return')
                        else:
                            st.info("Not enough monthly data to create returns heatmap.")
                            monthly_pivot = None
                    else:
                        # If no backtest results, just show a message
                        st.info("No monthly return data available yet. Run the backtest to see monthly returns.")
                        monthly_pivot = None
                    
                    # Only create the heatmap if we have data
                    if monthly_pivot is not None and not monthly_pivot.empty:
                        # Create a mapping of month numbers to month names
                        month_names = {
                            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                        }
                        
                        # Sort the columns (months) to ensure they're in calendar order
                        monthly_pivot = monthly_pivot.sort_index(axis=1)
                        
                        # Create heatmap using plotly with proper month labels
                        monthly_fig = go.Figure(data=go.Heatmap(
                            z=monthly_pivot.values,
                            x=[month_names[m] for m in monthly_pivot.columns],  # Use month names for display
                            y=[str(int(year)) for year in monthly_pivot.index],  # Convert to integer years for display
                            colorscale='RdBu',
                            zmid=0,
                            text=[[f"{val:.2f}%" for val in row] for row in monthly_pivot.values],
                            texttemplate="%{text}",
                            colorbar=dict(title='Return %')
                        ))
                        
                        monthly_fig.update_layout(
                            title='Monthly Returns (%)',
                            xaxis_title='Month',
                            yaxis_title='Year',
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(monthly_fig, use_container_width=True, key="v7_monthly_returns_heatmap")
                    
                except Exception as fig_error:
                    st.warning(f"Could not display detailed analysis charts: {str(fig_error)}")
                
        except Exception as e:
            st.error(f"An error occurred during backtesting: {str(e)}")
            progress_bar.progress(100)
    
    elif run_backtest:
        if df is None:
            st.error("No data available for backtesting. Please select a valid data source.")
        elif len(st.session_state.v7_bots_config) == 0:
            st.error("Please configure bot timing settings before running the backtest.")
else:
    st.info("Please select a data source from the sidebar to continue.")