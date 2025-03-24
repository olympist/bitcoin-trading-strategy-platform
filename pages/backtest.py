import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, date, timedelta, time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_fetcher import DataFetcher
from strategy.variable_timeframe_dca import VariableTimeframeDCA
from strategy.alternative_dca import AlternativeDCA
from strategy.alternative_dca_v3 import AlternativeDCAv3
from strategy.optimized_alternative_dca_v4 import OptimizedAlternativeDCAv4
from strategy.optimized_alternative_dca_v5 import OptimizedAlternativeDCAv5
from strategy.optimized_alternative_dca_v6 import OptimizedAlternativeDCAv6, ALLOWED_INTERVALS
from strategy.fixed_investment_dca import FixedInvestmentDCA
from strategy.liquidity_managed_dca import LiquidityManagedDCA
from strategy.interval_dca import IntervalDCA
from strategy.hybrid_dca import HybridDCA
from strategy.optimized_alternative_dca import OptimizedAlternativeDCA
from backtesting.backtester import Backtester
from visualization.performance_plots import PerformancePlots

st.set_page_config(
    page_title="Backtest - Bitcoin Backtesting Platform",
    page_icon="📈",
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
st.markdown('<p class="section-header">Bitcoin DCA Strategy Backtesting</p>', unsafe_allow_html=True)
st.markdown("Backtest multiple Dollar Cost Averaging (DCA) strategies on historical Bitcoin price data, including Original DCA, Alternative DCA, Fixed Investment DCA, Liquidity-Managed Dynamic DCA, and Interval-based DCA.")

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
    
    # Strategy parameters section
    st.subheader("2. Strategy Parameters")
    
    # Initialize strategy_type in session state if it doesn't exist
    if 'strategy_type' not in st.session_state:
        st.session_state.strategy_type = "Optimized Alternative DCA v5"
    
    # Strategy selection outside the form to allow the UI to update
    strategy_type = st.selectbox(
        "Strategy Type",
        options=["Original DCA", "Alternative DCA", "Alternative DCA v2.0", "Alternative DCA v3.0", "Optimized Alternative DCA v4", "Optimized Alternative DCA v5", "Optimized Alternative DCA v6", "Fixed Investment DCA", "Liquidity-Managed DCA", "Interval DCA", "Hybrid DCA"],
        help="Select which DCA strategy variant to use for backtesting",
        key="strategy_type"  # This connects it to session state
    )
    
    # Now create the form with the appropriate fields based on strategy_type
    with st.form("strategy_params_form"):
        initial_capital = st.number_input(
            "Initial Capital (USDT)",
            min_value=1.0,
            value=1000.0,
            step=1.0,
            help="Starting investment amount in USDT"
        )
        
        # Show different parameter inputs based on strategy type
        if strategy_type == "Liquidity-Managed DCA":
            # Liquidity-Managed DCA specific parameters
            st.markdown("#### Strategy Parameters")
            st.markdown("Configure the investment amounts for different market conditions:")
            
            invest_increase = st.number_input(
                "X: Invest When Price Increases (%)",
                min_value=0.1, 
                value=0.5, 
                step=0.1,
                help="Percentage of initial capital to invest when price increases"
            )
            
            invest_flat = st.number_input(
                "Y: Invest When Price is Flat (%)",
                min_value=0.1, 
                value=1.0, 
                step=0.1,
                help="Percentage of initial capital to invest when price stays flat"
            )
            
            invest_drop_significant = st.number_input(
                "Z: Invest On Significant Drops (%)",
                min_value=0.1, 
                value=3.0, 
                step=0.1,
                help="Percentage of initial capital to invest on significant price drops (>= drop threshold)"
            )
            
            invest_drop_non_significant = st.number_input(
                "W: Invest On Non-Significant Drops (%)",
                min_value=0.1, 
                value=1.5, 
                step=0.1,
                help="Percentage of initial capital to invest on smaller price drops (< drop threshold)"
            )
            
            # Common parameters for Liquidity-Managed DCA
            st.markdown("#### Threshold Settings")
            
            drop_threshold = st.number_input(
                "U: Drop Threshold (%)",
                min_value=0.1, 
                value=1.5, 
                step=0.1,
                help="Threshold percentage (U%) for defining a significant drop"
            )
            
            exit_profit_margin = st.number_input(
                "K: Exit Profit Target (%)",
                min_value=0.1, 
                value=2.0, 
                step=0.1,
                help="Target profit percentage (K%) that triggers selling"
            )
                
            # Initialize placeholder variables for other strategy types
            investment_pct = 10.0
            price_drop_threshold = 2.0
            profit_threshold = 3.0
            dca_amount = 40.0
            interval_minutes = 15
            profit_target = 2.0
            
        elif strategy_type == "Interval DCA":
            # Interval DCA specific parameters
            st.markdown("#### Time-Based DCA Parameters")
            st.markdown("Configure the regular investment schedule and profit target:")
            
            dca_amount = st.number_input(
                "DCA Investment Amount (USDT)",
                min_value=1.0, 
                value=40.0, 
                step=1.0,
                help="Fixed amount to invest at each interval"
            )
            
            interval_minutes = st.number_input(
                "Interval Between Investments (minutes)",
                min_value=1, 
                value=15, 
                step=1,
                help="Time interval between regular investments"
            )
            
            profit_target = st.number_input(
                "Profit Target (%)",
                min_value=0.1, 
                value=2.0, 
                step=0.1,
                help="Profit percentage that triggers selling"
            )
            
            # Initialize placeholder variables for other strategy types
            investment_pct = 10.0
            price_drop_threshold = 2.0
            profit_threshold = 3.0
            invest_increase = 0.5
            invest_flat = 1.0
            invest_drop_significant = 3.0
            invest_drop_non_significant = 1.5
            drop_threshold = 1.5
            exit_profit_margin = 2.0
            reserve_pct = 25.0
            extra_buy_pct = 50.0
            stop_loss_pct = 10.0
            
        elif strategy_type == "Alternative DCA v3.0":
            # Alternative DCA v3.0 specific parameters
            st.markdown("#### Alternative DCA v3.0 Parameters")
            st.markdown("Configure the enhanced strategy with simplified risk management features:")
            
            # Basic parameters
            col1, col2 = st.columns(2)
            with col1:
                investment_pct = st.number_input(
                    "Investment Amount (%)",
                    min_value=0.1,
                    value=10.0,
                    step=0.1,
                    help="Percentage of available capital to invest per buy"
                )
                
                price_drop_threshold = st.number_input(
                    "Price Drop Threshold (%)",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    help="Minimum price decrease required to trigger a buy"
                )
                
                profit_threshold = st.number_input(
                    "Profit Threshold (%)",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="Target profit percentage that triggers selling"
                )
            
            with col2:
                stop_loss_fixed_pct = st.number_input(
                    "Fixed Stop-Loss (%)",
                    min_value=0.0,
                    value=15.0,
                    step=0.5,
                    help="Maximum allowed loss before triggering stop loss (% below entry)"
                )
                
                trailing_stop_pct = st.number_input(
                    "Trailing Stop (%)",
                    min_value=0.0,
                    value=1.5,
                    step=0.1,
                    help="Trailing stop percentage below highest price reached after entry"
                )
                
                max_position_duration = st.number_input(
                    "Max Position Duration (days)",
                    min_value=1,
                    value=180,
                    step=1,
                    help="Maximum duration to hold a position before forced exit"
                )
                
                # No trend filter for Alternative DCA v3.0
                
        elif strategy_type == "Optimized Alternative DCA v4":
            # Optimized Alternative DCA v4 specific parameters
            st.markdown("#### Optimized Alternative DCA v4 Parameters")
            st.markdown("Configure the simplified strategy with enhanced risk management features:")
            
            # Basic parameters
            col1, col2 = st.columns(2)
            with col1:
                investment_pct = st.number_input(
                    "Investment Amount (%)",
                    min_value=0.1,
                    value=10.0,
                    step=0.1,
                    key="v4_investment_pct",
                    help="Percentage of available capital to invest per buy"
                )
                
                price_drop_threshold = st.number_input(
                    "Price Drop Threshold (%)",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    key="v4_price_drop_threshold",
                    help="Minimum price decrease required to trigger a buy"
                )
                
                profit_threshold = st.number_input(
                    "Profit Threshold (%)",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    key="v4_profit_threshold",
                    help="Target profit percentage that triggers selling"
                )
            
            with col2:
                stop_loss_fixed_pct = st.number_input(
                    "Fixed Stop-Loss (%)",
                    key="v4_stop_loss_fixed_pct",
                    min_value=0.0,
                    value=15.0,
                    step=0.5,
                    help="Maximum allowed loss before triggering stop loss (% below entry)"
                )
                
                trailing_stop_pct = st.number_input(
                    "Trailing Stop (%)",
                    key="v4_trailing_stop_pct",
                    min_value=0.0,
                    value=1.5,
                    step=0.1,
                    help="Trailing stop percentage below highest price reached after entry"
                )
                
                max_position_duration = st.number_input(
                    "Max Position Duration (days)",
                    key="v4_max_position_duration",
                    min_value=1,
                    value=180,
                    step=1,
                    help="Maximum duration to hold a position before forced exit"
                )
                

            st.markdown("Configure the minute-based multi-bot strategy with enhanced risk management features:")
            
            # Basic parameters
            col1, col2 = st.columns(2)
            with col1:
                investment_pct = st.number_input(
                    "Investment Amount (%)",
                    min_value=0.1,
                    value=10.0,
                    step=0.1,
                    help="Percentage of available capital to invest per buy"
                )
                
                price_drop_threshold = st.number_input(
                    "Price Drop Threshold (%)",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    help="Minimum price decrease required to trigger a buy"
                )
            
            with col2:
                profit_threshold = st.number_input(
                    "Profit Threshold (%)",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="Target profit percentage that triggers selling"
                )
                
                trading_fee_pct = st.number_input(
                    "Trading Fee (%)",
                    min_value=0.0,
                    value=0.1,
                    step=0.01,
                    help="Trading fee percentage (e.g., 0.1 for 0.1%)"
                )
            
            # Risk management parameters
            st.markdown("##### Risk Management Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                stop_loss_fixed_pct = st.number_input(
                    "Fixed Stop-Loss (%)",
                    min_value=1.0,
                    value=15.0,
                    step=0.5,
                    help="Fixed stop-loss percentage from entry price"
                )
            
            with col2:
                trailing_stop_pct = st.number_input(
                    "Trailing Stop (%)",
                    min_value=0.1,
                    value=1.5,
                    step=0.1,
                    help="Trailing stop percentage below highest price since entry"
                )
            
            with col3:
                max_position_minutes = st.number_input(
                    "Max Position Duration (minutes)",
                    min_value=1,
                    value=180,
                    step=1,
                    help="Maximum number of minutes to hold a position"
                )
                
            # Multi-bot configuration
            st.markdown("##### Multi-Bot Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                number_of_bots = st.number_input(
                    "Number of Bots",
                    min_value=1,
                    value=3,
                    step=1,
                    help="Number of bots to run under the minute-based umbrella strategy"
                )
                
            with col2:
                st.markdown("**Bot Deployment Strategy**")
                uniform_deployment = st.checkbox(
                    "Uniform deployment across period",
                    value=True,
                    help="When checked, bots will be deployed at equal intervals throughout the backtest period"
                )
                
                if uniform_deployment and number_of_bots > 1:
                    uniform_deployment_minutes = st.number_input(
                        "Minutes between bots",
                        min_value=1,
                        # Removed artificial limitation
                        value=60,
                        step=1,
                        help="Number of minutes between each bot's start time (0 = auto calculate based on period length)"
                    )
                else:
                    pass  # No special parameters for other cases
                    
        elif strategy_type == "Optimized Alternative DCA v5":
            # Optimized Alternative DCA v5 specific parameters
            st.markdown("#### Optimized Alternative DCA v5 Parameters (Multi-Bot Umbrella Strategy)")
            st.markdown("Configure the multi-bot umbrella strategy with enhanced risk management features:")
            
            # Basic parameters
            col1, col2 = st.columns(2)
            with col1:
                investment_pct = st.number_input(
                    "Investment Amount (%)",
                    key="v5_investment_pct",
                    min_value=0.1,
                    value=10.0,
                    step=0.1,
                    help="Percentage of available capital to invest per buy"
                )
                
                price_drop_threshold = st.number_input(
                    "Price Drop Threshold (%)",
                    key="v5_price_drop_threshold",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    help="Minimum price decrease required to trigger a buy"
                )
            
            with col2:
                profit_threshold = st.number_input(
                    "Profit Threshold (%)",
                    key="v5_profit_threshold",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="Target profit percentage that triggers selling"
                )
                
                trading_fee_pct = st.number_input(
                    "Trading Fee (%)",
                    key="v5_trading_fee_pct",
                    min_value=0.0,
                    value=0.1,
                    step=0.01,
                    help="Trading fee percentage (e.g., 0.1 for 0.1%)"
                )
            
            # Risk management parameters
            st.markdown("##### Risk Management Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                stop_loss_fixed_pct = st.number_input(
                    "Fixed Stop-Loss (%)",
                    key="v5_stop_loss_fixed_pct",
                    min_value=1.0,
                    value=15.0,
                    step=0.5,
                    help="Fixed stop-loss percentage from entry price"
                )
            
            with col2:
                trailing_stop_pct = st.number_input(
                    "Trailing Stop (%)",
                    key="v5_trailing_stop_pct",
                    min_value=0.1,
                    value=1.5,
                    step=0.1,
                    help="Trailing stop percentage below highest price since entry"
                )
                
                # Trailing stop activation threshold
                trailing_stop_activation_pct = st.number_input(
                    "Trailing Stop Activation (%)",
                    key="v5_trailing_stop_activation_pct",
                    min_value=0.1,
                    value=1.5,
                    step=0.1,
                    help="Profit percentage required to activate trailing stop"
                )
                
                # Trailing stop as primary mechanism
                use_trailing_stop_main = st.checkbox(
                    "Use Trailing Stop as Primary",
                    key="v5_use_trailing_stop_main",
                    value=True,
                    help="If enabled, trailing stop becomes the primary profit-protection mechanism"
                )
            
            with col3:
                max_position_duration = st.number_input(
                    "Max Position Duration (days)",
                    key="v5_max_position_duration",
                    min_value=1,
                    value=180,
                    step=1,
                    help="Maximum number of days to hold a position"
                )
                
                # Add partial profit taking options
                partial_profit_taking = st.checkbox(
                    "Enable Partial Profit Taking",
                    key="v5_partial_profit_taking",
                    value=False,
                    help="If enabled, only a portion of position will be sold when profit target is hit"
                )
                
                profit_taking_percentage = st.number_input(
                    "Profit Taking Percentage (%)",
                    key="v5_profit_taking_percentage",
                    min_value=10.0,
                    max_value=100.0,
                    value=50.0,
                    step=10.0,
                    help="Percentage of position to sell when taking partial profits (10-100%)"
                )
                
            # Multi-bot configuration
            st.markdown("##### Multi-Bot Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                number_of_bots = st.number_input(
                    "Number of Bots",
                    key="v5_number_of_bots",
                    min_value=1,
                    value=3,
                    step=1,
                    help="Number of bots to run under the umbrella strategy (no upper limit)"
                )
                
            with col2:
                st.markdown("**Bot Deployment Strategy**")
                deployment_options = st.radio(
                    "Deployment method",
                    options=["Fixed days between bots", "Evenly distribute across dataset", "Custom start dates"],
                    key="v5_deployment_method",
                    help="Choose how to deploy multiple bots across the trading period"
                )
                
                if deployment_options == "Fixed days between bots" and number_of_bots > 1:
                    uniform_deployment = True
                    auto_distribute = False
                    uniform_deployment_days = st.number_input(
                        "Days between bots",
                        min_value=1,
                        
                        value=30,
                        step=1,
                        key="v5_uniform_deployment_days",
                        help="Fixed number of days between each bot's start date"
                    )
                    
                    # Set the v5_custom_start_dates flag to also show date inputs
                    # This allows us to automatically populate the dates based on fixed spacing
                    st.session_state['v5_custom_start_dates'] = True
                    
                    # Also store the uniform_deployment_days for reference when setting date inputs
                    st.session_state['v5_fixed_days'] = uniform_deployment_days
                elif deployment_options == "Evenly distribute across dataset" and number_of_bots > 1:
                    uniform_deployment = True
                    auto_distribute = True
                    uniform_deployment_days = 0  # Special value to indicate auto-calculation
                    st.info("Bots will be automatically distributed evenly across the entire dataset period.")
                else:
                    # Custom start dates option
                    uniform_deployment = False
                    auto_distribute = False
                    uniform_deployment_days = None
                    
                    # Add custom start dates configuration right here
                    st.markdown("##### Custom Start Dates for Each Bot")
                    st.info("Please complete other parameters and run the backtest. You'll be able to set custom start dates on the main form.")
                    
                    # Initialize v5_custom_start_dates if it doesn't exist in session state
                    if 'v5_custom_start_dates' not in st.session_state:
                        st.session_state['v5_custom_start_dates'] = True
                    else:
                        # Set it to True since user selected the custom start dates option
                        st.session_state['v5_custom_start_dates'] = True
        
        elif strategy_type == "Optimized Alternative DCA v6":
            # Optimized Alternative DCA v6 specific parameters
            st.markdown("#### Optimized Alternative DCA v6 Parameters (Time-Based Multi-Bot Strategy)")
            st.markdown("Configure the multi-bot strategy with timeframe resampling features:")
            
            # Information about the new resampling feature
            st.info("""
            **New in v6: Timeframe Resampling**
            
            This strategy enhances the v5 multi-bot capability with timeframe resampling, allowing you to control when bots can execute trades.
            By setting a specific bot action interval (e.g., 15min, 1h, 4h), you can:
            - Reduce overtrading by restricting trade execution to specific timeframes
            - Better align your strategy with market cycles
            - Potentially improve performance by focusing on more significant price movements
            
            All risk management features from v5 are maintained with this added control over execution timing.
            """)
            
            # Basic parameters
            col1, col2 = st.columns(2)
            with col1:
                investment_pct = st.number_input(
                    "Investment Amount (%)",
                    key="v6_investment_pct",
                    min_value=0.1,
                    value=10.0,
                    step=0.1,
                    help="Percentage of available capital to invest per buy"
                )
                
                price_drop_threshold = st.number_input(
                    "Price Drop Threshold (%)",
                    key="v6_price_drop_threshold",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    help="Minimum price decrease required to trigger a buy"
                )
            
            with col2:
                profit_threshold = st.number_input(
                    "Profit Threshold (%)",
                    key="v6_profit_threshold",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="Target profit percentage that triggers selling"
                )
                
                trading_fee_pct = st.number_input(
                    "Trading Fee (%)",
                    key="v6_trading_fee_pct",
                    min_value=0.0,
                    value=0.1,
                    step=0.01,
                    help="Trading fee percentage (e.g., 0.1 for 0.1%)"
                )
            
            # Risk management parameters
            st.markdown("##### Risk Management Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                stop_loss_fixed_pct = st.number_input(
                    "Fixed Stop-Loss (%)",
                    key="v6_stop_loss_fixed_pct",
                    min_value=1.0,
                    value=15.0,
                    step=0.5,
                    help="Fixed stop-loss percentage from entry price"
                )
            
            with col2:
                trailing_stop_pct = st.number_input(
                    "Trailing Stop (%)",
                    key="v6_trailing_stop_pct",
                    min_value=0.1,
                    value=1.5,
                    step=0.1,
                    help="Trailing stop percentage below highest price since entry"
                )
                
                # Trailing stop activation threshold
                trailing_stop_activation_pct = st.number_input(
                    "Trailing Stop Activation (%)",
                    key="v6_trailing_stop_activation_pct",
                    min_value=0.1,
                    value=1.5,
                    step=0.1,
                    help="Profit percentage required to activate trailing stop"
                )
                
                # Trailing stop as primary mechanism
                use_trailing_stop_main = st.checkbox(
                    "Use Trailing Stop as Primary",
                    key="v6_use_trailing_stop_main",
                    value=True,
                    help="If enabled, trailing stop becomes the primary profit-protection mechanism"
                )
            
            with col3:
                max_position_duration = st.number_input(
                    "Max Position Duration (days)",
                    key="v6_max_position_duration",
                    min_value=1,
                    value=180,
                    step=1,
                    help="Maximum days to hold a position before forced exit"
                )
                
                # Partial profit taking settings
                partial_profit_taking = st.checkbox(
                    "Enable Partial Profit Taking",
                    key="v6_partial_profit_taking",
                    value=False,
                    help="If enabled, only a portion of the position will be sold at profit target"
                )
                
                # Create a container for the profit taking percentage - this helps ensure UI updates properly
                profit_taking_container = st.container()
                
                with profit_taking_container:
                    if partial_profit_taking:
                        st.markdown("##### Configure Partial Profit Taking")
                        profit_taking_percentage = st.number_input(
                            "Profit Taking % of Position",
                            key="v6_profit_taking_percentage",
                            min_value=10.0,
                            max_value=90.0,
                            value=50.0,
                            step=10.0,
                            help="Percentage of position to sell when profit target is hit"
                        )
                        # Add an explanatory note
                        st.caption("This percentage of your position will be sold when the profit target is reached. The remainder continues to ride the trend.")
                    else:
                        profit_taking_percentage = 100.0
                    
            # Timeframe selection - new in v6
            st.markdown("##### Bot Action Interval (Timeframe Resampling)")
            st.markdown("Configure how frequently your bots evaluate the market and potentially execute trades:")
            
            bot_action_interval_choice = st.selectbox(
                "Bot Action Interval",
                options=list(ALLOWED_INTERVALS.keys()),
                index=2,  # Default to 15min
                key="v6_bot_action_interval",
                help="Choose how frequently bots should evaluate and potentially execute trades"
            )
            
            st.info(f"Setting bots to only act at {bot_action_interval_choice} intervals. This will resample the data to execute trades only at these intervals, reducing overtrading.")
            
            # Multi-bot configuration
            st.markdown("##### Multi-Bot Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                number_of_bots = st.number_input(
                    "Number of Bots",
                    key="v6_number_of_bots",
                    min_value=1,
                    value=3,
                    step=1,
                    help="Number of bots to run in the umbrella strategy"
                )
                
            with col2:
                st.markdown("**Bot Deployment Strategy**")
                uniform_deployment = st.checkbox(
                    "Uniform deployment across period",
                    key="v6_uniform_deployment",
                    value=True,
                    help="When checked, bots will be deployed at equal intervals throughout the backtest period"
                )
                
                if uniform_deployment and number_of_bots > 1:
                    uniform_deployment_days = st.number_input(
                        "Days between bots",
                        key="v6_uniform_deployment_days",
                        min_value=1,
                        value=30,
                        step=1,
                        help="Number of days between each bot's start time"
                    )
                else:
                    # Set to None when not using uniform deployment
                    # This will handle bot deployment differently in the strategy
                    uniform_deployment_days = None
                
            # Cooldown Parameters - New in V6
            st.markdown("##### Cooldown Parameters")
            st.markdown("Configure trading cooldown periods to prevent over-trading:")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                profit_cooldown_threshold_pct = st.number_input(
                    "Profit Cooldown Threshold (%)",
                    key="v6_profit_cooldown_threshold_pct",
                    min_value=1.0,
                    max_value=100.0,
                    value=50.0,
                    step=5.0,
                    help="Percentage of profit threshold at which to reset cooldown"
                )
            
            with col2:
                min_time_between_trades_minutes = st.number_input(
                    "Min Time Between Trades (min)",
                    key="v6_min_time_between_trades_minutes",
                    min_value=1,
                    max_value=1440,
                    value=120,
                    step=10,
                    help="Minimum time in minutes between same-type trades (buy-to-buy or sell-to-sell)"
                )
            
            with col3:
                min_time_between_buy_sell_minutes = st.number_input(
                    "Min Time Between Buy/Sell (min)",
                    key="v6_min_time_between_buy_sell_minutes",
                    min_value=1,
                    max_value=1440,
                    value=30,
                    step=5,
                    help="Minimum time in minutes between a buy and a sell operation"
                )
                
            # Advanced bot configuration (only show if not using uniform deployment)
            if not uniform_deployment and number_of_bots > 1:
                st.markdown("##### Bot Start Times (Custom Deployment)")
                
                # Add market-based timing option
                market_timing = st.radio(
                    "Bot Timing Strategy",
                    ["Custom Times", "Market Opening Times"],
                    index=0,
                    key="v5_market_timing",
                    help="Select timing strategy for your bots"
                )
                
                # Create lists to store the custom start dates and times
                bots_starting_dates = []
                bots_starting_times = []
                
                if market_timing == "Market Opening Times":
                    st.markdown("**Market Opening Time Distribution**")
                    st.info("Bots will be distributed to start at major market opening times across your selected date range")
                    
                    market_times = {
                        "Asian Market (Tokyo)": "00:00",
                        "European Market (London)": "08:00",
                        "US Market (New York)": "13:30"
                    }
                    
                    # Let user select which markets to use
                    selected_markets = st.multiselect(
                        "Select markets for bot deployment",
                        options=list(market_times.keys()),
                        default=["Asian Market (Tokyo)", "US Market (New York)"],
                        key="v5_selected_markets",
                        help="Select which market opening times to use for bot deployments"
                    )
                    
                    if not selected_markets:
                        st.warning("Please select at least one market")
                        selected_markets = ["Asian Market (Tokyo)"]
                    
                    # Calculate date distribution
                    date_range = (end_date - start_date).days
                    if date_range <= 0:
                        date_range = 1
                    
                    days_between = max(1, date_range // len(selected_markets))
                    
                    # Create the distribution of dates and market times
                    current_date = start_date
                    market_index = 0
                    
                    for i in range(number_of_bots):
                        # Calculate the date for this bot (distribute evenly)
                        if i > 0 and i % len(selected_markets) == 0:
                            current_date += timedelta(days=days_between)
                        
                        # Get the market for this bot (cycle through selected markets)
                        market = selected_markets[market_index % len(selected_markets)]
                        market_time = market_times[market]
                        
                        # Add to lists
                        bots_starting_dates.append(current_date.strftime('%Y-%m-%d'))
                        bots_starting_times.append(market_time)
                        
                        # Show the assignment to the user
                        st.write(f"Bot {i+1}: {current_date.strftime('%Y-%m-%d')} at {market_time} ({market})")
                        
                        market_index += 1
                
                else:  # Custom Times
                    st.markdown("Define custom starting dates and times for each bot")
                    
                    # Create rows of inputs for each bot
                    for i in range(number_of_bots):
                        col1, col2 = st.columns(2)
                        with col1:
                            # Calculate evenly spaced dates within the available range
                            total_days = (end_date - start_date).days
                            spacing = total_days / max(1, number_of_bots - 1) if number_of_bots > 1 else 0
                            default_date = start_date + timedelta(days=min(int(i * spacing), total_days))
                            
                            bot_start_date = st.date_input(
                                f"Bot {i+1} Start Date",
                                value=default_date,  # Use safe default within range
                                key=f"v5_bot_{i+1}_start_date"
                            )
                            bots_starting_dates.append(bot_start_date.strftime('%Y-%m-%d'))
                        
                        with col2:
                            bot_start_time = st.time_input(
                                f"Bot {i+1} Start Time",
                                value=time(hour=12, minute=0),  # Default to noon
                                key=f"v5_bot_{i+1}_start_time"
                            )
                            bots_starting_times.append(bot_start_time.strftime('%H:%M'))
            else:
                # For uniform deployment, set starting dates to None to let the strategy handle it
                bots_starting_dates = None
                bots_starting_times = None
                
        elif strategy_type == "Alternative DCA v2.0":
            # Alternative DCA v2.0 specific parameters
            st.markdown("#### Alternative DCA v2.0 Parameters")
            st.markdown("Configure the optimized strategy with enhanced risk management features:")
            
            # Basic parameters
            col1, col2 = st.columns(2)
            with col1:
                investment_pct = st.number_input(
                    "Investment Amount (%)",
                    min_value=0.1,
                    value=10.0,
                    step=0.1,
                    help="Percentage of available capital to invest per buy"
                )
                
                price_drop_threshold = st.number_input(
                    "Price Drop Threshold (%)",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    help="Minimum price decrease required to trigger a buy"
                )
                
                profit_threshold = st.number_input(
                    "Profit Threshold (%)",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="Target profit percentage that triggers selling"
                )
            
            with col2:
                stop_loss_fixed_pct = st.number_input(
                    "Fixed Stop-Loss (%)",
                    min_value=0.0,
                    value=15.0,
                    step=0.5,
                    help="Maximum allowed loss before triggering stop loss (% below entry)"
                )
                
                trailing_stop_pct = st.number_input(
                    "Trailing Stop (%)",
                    min_value=0.0,
                    value=1.5,
                    step=0.1,
                    help="Trailing stop percentage below highest price reached after entry"
                )
                
                max_position_duration = st.number_input(
                    "Max Position Duration (days)",
                    min_value=1,
                    value=180,
                    step=1,
                    help="Maximum duration to hold a position before forced exit"
                )
            
            # Advanced parameters
            st.markdown("#### Advanced Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                trend_filter_ma_period = st.number_input(
                    "Trend Filter MA Period (days)",
                    min_value=1,
                    value=50,
                    step=1,
                    help="Moving average period for trend filtering"
                )
                
                volatility_atr_window = st.number_input(
                    "ATR Volatility Window (days)",
                    min_value=1,
                    value=14,
                    step=1,
                    help="ATR calculation window for volatility measurement"
                )
                
                atr_stop_multiplier = st.number_input(
                    "ATR Stop-Loss Multiplier",
                    min_value=0.1,
                    value=3.0,
                    step=0.1,
                    help="Multiplier for ATR-based stop loss distance"
                )
            
            with col2:
                min_order_size = st.number_input(
                    "Minimum Order Size (USDT)",
                    min_value=1.0,
                    value=50.0,
                    step=1.0,
                    help="Minimum amount for executing an order"
                )
                
                order_aggregation_threshold = st.number_input(
                    "Order Aggregation Threshold (%)",
                    min_value=0.1,
                    value=2.0,
                    step=0.1,
                    help="Threshold for aggregating orders placed close together"
                )
                
                trailing_stop_activation_pct = st.number_input(
                    "Trailing Stop Activation Threshold (%)",
                    min_value=10.0,
                    
                    value=50.0,
                    step=5.0,
                    help="Percentage price increase needed to activate trailing stop (e.g., 1.5% profit to start trailing)"
                )
            
            # Initialize placeholder variables for other strategy types
            invest_increase = 0.5
            invest_flat = 1.0
            invest_drop_significant = 3.0
            invest_drop_non_significant = 1.5
            drop_threshold = 1.5
            exit_profit_margin = 2.0
            dca_amount = 40.0
            interval_minutes = 15
            profit_target = 2.0
            reserve_pct = 25.0
            extra_buy_pct = 50.0
            stop_loss_pct = 10.0

        elif strategy_type == "Hybrid DCA":
            # Hybrid DCA specific parameters
            st.markdown("#### Hybrid DCA Parameters")
            st.markdown("Configure the strategy with reserve capital and extra buys on dips:")
            
            dca_amount = st.number_input(
                "Regular DCA Investment Amount (USDT)",
                min_value=1.0, 
                value=50.0, 
                step=1.0,
                help="Fixed amount to invest at each interval"
            )
            
            interval_minutes = st.number_input(
                "Interval Between Investments (minutes)",
                min_value=1, 
                value=15, 
                step=1,
                help="Time interval between regular investments"
            )
            
            reserve_pct = st.number_input(
                "Reserve Capital Percentage (%)",
                min_value=0.0,
                value=25.0,
                step=1.0,
                help="Percentage of capital to set aside as reserve for market dips"
            )
            
            drop_threshold = st.number_input(
                "Price Drop Threshold (%)",
                min_value=0.1,
                value=0.5,
                step=0.1,
                help="Price drop percentage that triggers extra buying"
            )
            
            extra_buy_pct = st.number_input(
                "Extra Buy Amount (%)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                help="Additional percentage of regular DCA amount to buy on dips"
            )
            
            profit_target = st.number_input(
                "Profit Target (%)",
                min_value=0.1, 
                value=2.0, 
                step=0.1,
                help="Profit percentage that triggers selling"
            )
            
            stop_loss_pct = st.number_input(
                "Stop Loss (%)",
                min_value=0.0,
                value=10.0,
                step=0.5,
                help="Maximum allowed loss before triggering a stop loss (0 to disable)"
            )
            
            # Initialize placeholder variables for other strategy types
            investment_pct = 10.0
            price_drop_threshold = 2.0
            profit_threshold = 3.0
            invest_increase = 0.5
            invest_flat = 1.0
            invest_drop_significant = 3.0
            invest_drop_non_significant = 1.5
            exit_profit_margin = 2.0
            
        else:
            # Standard parameters for other strategies
            investment_pct = st.number_input(
                "Investment Amount (%)",
                min_value=0.1,
                value=10.0,
                step=0.1,
                help="Percentage of available capital to invest per buy"
            )
            
            price_drop_threshold = st.number_input(
                "Price Drop Threshold (%)",
                min_value=0.1,
                value=2.0,
                step=0.1,
                help="Minimum price decrease required to trigger a buy"
            )
            
            profit_threshold = st.number_input(
                "Profit Threshold (%)",
                min_value=0.1,
                value=3.0,
                step=0.1,
                help="Target profit percentage that triggers selling"
            )
            
            # Initialize placeholder variables for Liquidity-Managed DCA
            invest_increase = 0.5  # Percentage of initial capital (0.5%)
            invest_flat = 1.0  # Percentage of initial capital (1.0%)
            invest_drop_significant = 3.0  # Percentage of initial capital (3.0%)
            invest_drop_non_significant = 1.5  # Percentage of initial capital (1.5%)
            drop_threshold = 1.5  # Percentage drop threshold (1.5%)
            exit_profit_margin = 2.0  # Profit percentage that triggers selling (2.0%)
            
            # Initialize placeholder variables for Interval DCA
            dca_amount = 40.0  # Fixed amount to invest at each interval
            interval_minutes = 15  # Time interval between investments
            profit_target = 2.0  # Profit percentage that triggers selling
            
            # Initialize placeholder variables for risk management strategies
            stop_loss_fixed_pct = 15.0  # Stop loss percentage
            trailing_stop_pct = 1.5  # Trailing stop percentage
            max_position_duration = 180  # Maximum position duration in days
        
        trading_fee_pct = st.number_input(
            "Trading Fee (%)",
            min_value=0.0,
            value=0.1,
            step=0.01,
            help="Trading fee percentage"
        )
        
        run_backtest_button = st.form_submit_button("Run Backtest")
    
    # Strategy parameters completed in sidebar

# Main content area
if st.session_state.selected_data is None:
    st.warning("Please select a data file from the sidebar to begin.")
else:
    df = st.session_state.selected_data
    
    # Date range selection in the main content area
    st.markdown('<p class="section-header">📅 Backtest Date Range</p>', unsafe_allow_html=True)
    st.markdown("Select the time period for your backtest. Different market conditions can significantly impact strategy performance.")
    
    date_min = df.index.min().date()
    date_max = df.index.max().date()
    
    # Create a date range container with border
    st.markdown("""
    <style>
    .date-range-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    .date-info {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .available-range {
        font-size: 14px;
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="date-range-container">', unsafe_allow_html=True)
    
    # Show available date range in the dataset
    st.markdown(f'<p class="available-range">Available data range: {date_min} to {date_max}</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date_min,
            min_value=date_min,
            max_value=date_max,
            help="The starting date of your backtest period"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date_max,
            min_value=date_min,
            max_value=date_max,
            help="The ending date of your backtest period"
        )
    
    # Display selected date range information
    duration_days = (end_date - start_date).days
    
    # Get the number of data points in the selected range
    data_points = len(df[(df.index.date >= start_date) & (df.index.date <= end_date)])
    
    # Calculate months as well for longer periods
    months = duration_days / 30.44  # Average days per month
    
    # Format the date range info based on duration
    if duration_days > 60:
        duration_text = f"{duration_days} days ({months:.1f} months)"
    else:
        duration_text = f"{duration_days} days"
        
    st.markdown(f'<p class="date-info">Selected Period: {start_date} to {end_date}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="date-info">Duration: {duration_text}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="date-info">Data Points: {data_points}</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market condition summary for selected period
    if data_points > 0:
        # Convert date objects to pandas Timestamp with time at start/end of day (same as filtering above)
        start_timestamp = pd.Timestamp(start_date).floor('D')  # Beginning of day
        end_timestamp = pd.Timestamp(end_date).ceil('D') - pd.Timedelta(microseconds=1)  # End of day
        
        # Get filtered data using the improved timestamp filtering
        period_data = df[(df.index >= start_timestamp) & (df.index <= end_timestamp)]
        
        if len(period_data) > 0:
            start_price = period_data.iloc[0]['close']
            end_price = period_data.iloc[-1]['close']
            price_change_pct = ((end_price - start_price) / start_price) * 100
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = period_data['close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            # Debug info for market stats
            st.write(f"Debug - Market stats calculated on {len(period_data)} data points from {period_data.index.min()} to {period_data.index.max()}")
            st.write(f"Debug - Price change: {price_change_pct:.2f}% (from ${start_price:.2f} to ${end_price:.2f})")
        else:
            price_change_pct = 0
            volatility = 0
            st.warning("No data available for market summary calculations.")
        
        # Determine market condition based on price change
        if price_change_pct > 20:
            market_condition = "Strong Bull Market"
            emoji = "🚀"
        elif price_change_pct > 5:
            market_condition = "Bull Market"
            emoji = "📈"
        elif price_change_pct > -5:
            market_condition = "Sideways Market"
            emoji = "↔️"
        elif price_change_pct > -20:
            market_condition = "Bear Market"
            emoji = "📉"
        else:
            market_condition = "Strong Bear Market"
            emoji = "🧸"
            
        # Create a market summary container
        st.markdown("""
        <style>
        .market-summary {
            margin-top: 15px;
            padding: 10px;
            border-radius: 5px;
            background-color: #e6f3ff;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="market-summary">', unsafe_allow_html=True)
        st.markdown(f"### {emoji} Market Summary for Selected Period")
        
        # Display market metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price Change", f"{price_change_pct:.2f}%")
        with col2:
            st.metric("Volatility", f"{volatility:.2f}%")
        with col3:
            st.metric("Market Type", f"{market_condition}")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        # Debug date range selection
        st.write(f"Debug - Selected date range: {start_date} to {end_date}")
        
        # Convert date objects to pandas Timestamp with time at start/end of day
        start_timestamp = pd.Timestamp(start_date).floor('D')  # Beginning of day
        end_timestamp = pd.Timestamp(end_date).ceil('D') - pd.Timedelta(microseconds=1)  # End of day
        
        st.write(f"Debug - Date range as timestamps: {start_timestamp} to {end_timestamp}")
        
        # Apply date filter to the data using timestamp comparison
        mask = (df.index >= start_timestamp) & (df.index <= end_timestamp)
        filtered_df = df.loc[mask].copy()  # Use copy to avoid SettingWithCopyWarning
        
        st.write(f"Debug - Original data range: {df.index.min()} to {df.index.max()} ({len(df)} points)")
        
        if len(filtered_df) == 0:
            st.error("No data in selected date range.")
        else:
            st.write(f"Debug - Filtered data range: {filtered_df.index.min()} to {filtered_df.index.max()} ({len(filtered_df)} points)")
            
            # Ready for backtest section
            st.markdown("---")
            
            # Custom start dates for V5 if selected in sidebar
            if strategy_type == "Optimized Alternative DCA v5" and 'v5_custom_start_dates' in st.session_state and st.session_state['v5_custom_start_dates']:
                st.markdown('<p class="section-header">🔄 Custom Start Dates for Each Bot</p>', unsafe_allow_html=True)
                st.info("Configure custom start dates and times for each bot in the V5 multi-bot strategy:")
                
                bots_starting_dates = []
                bots_starting_times = []
                
                # Create columns for the bot configuration
                for i in range(number_of_bots):
                    col1, col2 = st.columns(2)
                    with col1:
                        # Calculate dates based on fixed days or evenly spaced if using custom dates
                        total_days = (end_date - start_date).days
                        
                        # Check if using fixed days between bots
                        if 'v5_fixed_days' in st.session_state:
                            # Use the fixed days value from the settings
                            fixed_days = st.session_state['v5_fixed_days']
                            default_date = start_date + timedelta(days=min(i * fixed_days, total_days))
                        else:
                            # Default to evenly spacing bots across the date range
                            spacing = total_days / max(1, number_of_bots - 1) if number_of_bots > 1 else 0
                            default_date = start_date + timedelta(days=min(int(i * spacing), total_days))
                        
                        bot_start_date = st.date_input(
                            f"Bot {i+1} Start Date",
                            value=default_date,  # Use safe default within range
                            key=f"v5_main_bot_{i+1}_start_date",
                            min_value=start_date,
                            max_value=end_date
                        )
                        bots_starting_dates.append(bot_start_date.strftime('%Y-%m-%d'))
                    
                    with col2:
                        bot_start_time = st.time_input(
                            f"Bot {i+1} Start Time",
                            value=time(hour=12, minute=0),  # Default to noon
                            key=f"v5_main_bot_{i+1}_start_time"
                        )
                        bots_starting_times.append(bot_start_time.strftime('%H:%M'))
                        
                # Store these in session state so we can use them when running the backtest
                st.session_state['v5_bots_starting_dates'] = bots_starting_dates
                st.session_state['v5_bots_starting_times'] = bots_starting_times
            
            st.markdown('<p class="section-header">Execute Backtest</p>', unsafe_allow_html=True)
            st.markdown(f"Ready to run backtest with **{strategy_type}** strategy on **{len(filtered_df)}** data points from **{filtered_df.index.min().strftime('%Y-%m-%d %H:%M')}** to **{filtered_df.index.max().strftime('%Y-%m-%d %H:%M')}**.")
            
            # Run backtest if button clicked
            if run_backtest_button:
                try:
                    # Create strategy instance based on selection
                    if strategy_type == "Original DCA":
                        strategy = VariableTimeframeDCA(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct
                        )
                    elif strategy_type == "Fixed Investment DCA":
                        strategy = FixedInvestmentDCA(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct
                        )
                    elif strategy_type == "Liquidity-Managed DCA":
                        # Use the parameters from the form
                        strategy = LiquidityManagedDCA(
                            initial_capital=initial_capital,
                            exit_profit_margin=exit_profit_margin,
                            invest_increase=invest_increase,
                            invest_flat=invest_flat,
                            invest_drop_significant=invest_drop_significant,
                            invest_drop_non_significant=invest_drop_non_significant,
                            drop_threshold=drop_threshold,
                            trading_fee_pct=trading_fee_pct
                        )
                    elif strategy_type == "Interval DCA":
                        # Use the parameters from the form for Interval DCA
                        strategy = IntervalDCA(
                            initial_capital=initial_capital,
                            dca_amount=dca_amount,
                            profit_target=profit_target,
                            interval_minutes=interval_minutes,
                            trading_fee_pct=trading_fee_pct
                        )
                    elif strategy_type == "Hybrid DCA":
                        # Use the parameters from the form for Hybrid DCA
                        strategy = HybridDCA(
                            initial_capital=initial_capital,
                            dca_amount=dca_amount,
                            profit_target=profit_target,
                            interval_minutes=interval_minutes,
                            drop_threshold=drop_threshold,
                            extra_buy_pct=extra_buy_pct,
                            stop_loss_pct=stop_loss_pct,
                            trading_fee_pct=trading_fee_pct,
                            reserve_pct=reserve_pct
                        )
                    elif strategy_type == "Alternative DCA v2.0":
                        # Use the parameters from the form for Alternative DCA v2.0
                        strategy = OptimizedAlternativeDCA(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct,
                            stop_loss_fixed_pct=stop_loss_fixed_pct,
                            atr_stop_multiplier=atr_stop_multiplier,
                            trailing_stop_pct=trailing_stop_pct,
                            max_position_duration=max_position_duration,
                            trend_filter_ma_period=trend_filter_ma_period,
                            volatility_atr_window=volatility_atr_window,
                            min_order_size=min_order_size,
                            order_aggregation_threshold=order_aggregation_threshold,
                            trailing_stop_activation_pct=trailing_stop_activation_pct
                        )
                    elif strategy_type == "Alternative DCA v3.0":
                        # Use the parameters for Alternative DCA v3.0
                        strategy = AlternativeDCAv3(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct,
                            stop_loss_fixed_pct=stop_loss_fixed_pct,
                            trailing_stop_pct=trailing_stop_pct,
                            max_position_duration=max_position_duration
                        )
                    elif strategy_type == "Optimized Alternative DCA v4":
                        # Use the parameters for Optimized Alternative DCA v4
                        strategy = OptimizedAlternativeDCAv4(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct,
                            stop_loss_fixed_pct=stop_loss_fixed_pct,
                            trailing_stop_pct=trailing_stop_pct,
                            max_position_duration=max_position_duration
                        )
                    # Optimized Alternative DCA v5.1 has been removed
                    elif strategy_type == "Optimized Alternative DCA v5":
                        # Use the parameters for Optimized Alternative DCA v5 with multi-bot support
                        
                        # Initialize bots_starting_dates and bots_starting_times with safe defaults
                        if 'bots_starting_dates' not in locals():
                            bots_starting_dates = []
                        
                        if 'bots_starting_times' not in locals():
                            bots_starting_times = []
                            
                        # For uniform deployment, generate the dates automatically if needed
                        if uniform_deployment and number_of_bots > 1:
                            # Clear any previous values
                            bots_starting_dates = []
                            bots_starting_times = []
                            
                            # Calculate date spacing for uniform deployment
                            date_range = (end_date - start_date).days
                            if date_range <= 0:
                                date_range = 1
                                
                            day_step = max(1, date_range // number_of_bots) if uniform_deployment_days is None else uniform_deployment_days
                            current_date = start_date
                            
                            # Generate uniform dates and default times
                            for i in range(number_of_bots):
                                bot_date = current_date + timedelta(days=i * day_step)
                                if bot_date > end_date:
                                    bot_date = end_date
                                bots_starting_dates.append(bot_date.strftime("%Y-%m-%d"))
                                bots_starting_times.append("00:00")  # Default time
                        
                        # Ensure we have at least empty arrays
                        if bots_starting_dates is None:
                            bots_starting_dates = []
                        
                        if bots_starting_times is None:
                            bots_starting_times = []
                            
                        # Check if we have custom start dates from the form
                        if 'v5_custom_start_dates' in st.session_state and st.session_state['v5_custom_start_dates']:
                            if 'v5_bots_starting_dates' in st.session_state and len(st.session_state['v5_bots_starting_dates']) > 0:
                                bots_starting_dates = st.session_state['v5_bots_starting_dates']
                                
                            if 'v5_bots_starting_times' in st.session_state and len(st.session_state['v5_bots_starting_times']) > 0:
                                bots_starting_times = st.session_state['v5_bots_starting_times']
                            
                        strategy = OptimizedAlternativeDCAv5(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct,
                            stop_loss_fixed_pct=stop_loss_fixed_pct,
                            trailing_stop_pct=trailing_stop_pct,
                            trailing_stop_activation_pct=trailing_stop_activation_pct,
                            use_trailing_stop_main=use_trailing_stop_main,
                            max_position_duration=max_position_duration,
                            partial_profit_taking=partial_profit_taking,
                            profit_taking_percentage=profit_taking_percentage,
                            number_of_bots=number_of_bots,
                            bots_starting_dates=bots_starting_dates if number_of_bots > 1 else None,
                            bots_starting_times=bots_starting_times if number_of_bots > 1 else None,
                            uniform_deployment_days=uniform_deployment_days if number_of_bots > 1 else None
                        )
                    elif strategy_type == "Optimized Alternative DCA v6":
                        # Use the parameters for Optimized Alternative DCA v6 with timeframe resampling
                        
                        # Initialize bots_starting_dates and bots_starting_times with safe defaults
                        if 'bots_starting_dates' not in locals():
                            bots_starting_dates = []
                        
                        if 'bots_starting_times' not in locals():
                            bots_starting_times = []
                        
                        # Create the v6 strategy with timeframe resampling
                        strategy = OptimizedAlternativeDCAv6(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct,
                            stop_loss_fixed_pct=stop_loss_fixed_pct,
                            trailing_stop_pct=trailing_stop_pct,
                            trailing_stop_activation_pct=trailing_stop_activation_pct,
                            use_trailing_stop_main=use_trailing_stop_main,
                            max_position_duration=max_position_duration,
                            partial_profit_taking=partial_profit_taking,
                            profit_taking_percentage=profit_taking_percentage,
                            number_of_bots=number_of_bots,
                            bots_starting_dates=bots_starting_dates if number_of_bots > 1 else None,
                            bots_starting_times=bots_starting_times if number_of_bots > 1 else None,
                            uniform_deployment_days=uniform_deployment_days if number_of_bots > 1 else None,
                            bot_action_interval=bot_action_interval_choice,
                            # Cooldown parameters
                            profit_cooldown_threshold_pct=profit_cooldown_threshold_pct,
                            min_time_between_trades_minutes=min_time_between_trades_minutes,
                            min_time_between_buy_sell_minutes=min_time_between_buy_sell_minutes
                        )
                    else:  # Alternative DCA
                        strategy = AlternativeDCA(
                            initial_capital=initial_capital,
                            investment_pct=investment_pct,
                            price_drop_threshold=price_drop_threshold,
                            profit_threshold=profit_threshold,
                            trading_fee_pct=trading_fee_pct
                        )
                    
                    # Log parameters to debug
                    st.write(f"Debug: Strategy created with parameters for {strategy_type}")
                    if strategy_type == "Liquidity-Managed DCA":
                        st.write(f"Parameters: Initial capital = {initial_capital}, Exit profit = {exit_profit_margin}%, X = {invest_increase}, Y = {invest_flat}, Z = {invest_drop_significant}, W = {invest_drop_non_significant}, U = {drop_threshold}%, Trading fee = {trading_fee_pct}%")
                    elif strategy_type == "Interval DCA":
                        st.write(f"Parameters: Initial capital = {initial_capital}, DCA amount = ${dca_amount}, Interval = {interval_minutes} minutes, Profit target = {profit_target}%, Trading fee = {trading_fee_pct}%")
                    elif strategy_type == "Hybrid DCA":
                        st.write(f"Parameters: Initial capital = {initial_capital}, DCA amount = ${dca_amount}, Interval = {interval_minutes} minutes, Reserve = {reserve_pct}%, Drop threshold = {drop_threshold}%, Extra buy = {extra_buy_pct}%, Profit target = {profit_target}%, Stop loss = {stop_loss_pct}%, Trading fee = {trading_fee_pct}%")
                    elif strategy_type == "Alternative DCA v3.0":
                        st.write(f"Parameters: Initial capital = {initial_capital}, Investment = {investment_pct}%, Price drop = {price_drop_threshold}%, Profit = {profit_threshold}%, Fixed stop-loss = {stop_loss_fixed_pct}%, Trailing stop = {trailing_stop_pct}%, Max position duration = {max_position_duration} days, Trading fee = {trading_fee_pct}%")
                    elif strategy_type == "Optimized Alternative DCA v4":
                        st.write(f"Parameters: Initial capital = {initial_capital}, Investment = {investment_pct}%, Price drop = {price_drop_threshold}%, Profit = {profit_threshold}%, Fixed stop-loss = {stop_loss_fixed_pct}%, Trailing stop = {trailing_stop_pct}%, Max position duration = {max_position_duration} days, Trading fee = {trading_fee_pct}%")
                    # Optimized Alternative DCA v5.1 has been removed
                    elif strategy_type == "Optimized Alternative DCA v5":
                        st.write(f"Parameters: Initial capital = {initial_capital}, Investment = {investment_pct}%, Price drop = {price_drop_threshold}%, Profit = {profit_threshold}%, Fixed stop-loss = {stop_loss_fixed_pct}%, Trailing stop = {trailing_stop_pct}%")
                        st.write(f"Advanced parameters: Trailing stop activation = {trailing_stop_activation_pct}%, Use trailing stop as primary = {'Yes' if use_trailing_stop_main else 'No'}, Partial profit taking = {'Enabled' if partial_profit_taking else 'Disabled'}, Profit taking percentage = {profit_taking_percentage if partial_profit_taking else 'N/A'}%")
                        st.write(f"Multi-bot configuration: {number_of_bots} bots running as umbrella strategy, Max position duration = {max_position_duration} days, Trading fee = {trading_fee_pct}%")
                        
                        # Display custom start dates if available
                        if 'v5_custom_start_dates' in st.session_state and st.session_state['v5_custom_start_dates'] and 'v5_bots_starting_dates' in st.session_state:
                            st.info("**Custom bot start dates:**")
                            for i in range(min(number_of_bots, len(st.session_state.get('v5_bots_starting_dates', [])))):
                                date_str = st.session_state['v5_bots_starting_dates'][i] if i < len(st.session_state['v5_bots_starting_dates']) else 'Default'
                                time_str = st.session_state['v5_bots_starting_times'][i] if i < len(st.session_state['v5_bots_starting_times']) else '00:00'
                                st.write(f"Bot {i+1}: {date_str} at {time_str}")
                    elif strategy_type == "Optimized Alternative DCA v6":
                        st.write(f"Parameters: Initial capital = {initial_capital}, Investment = {investment_pct}%, Price drop = {price_drop_threshold}%, Profit = {profit_threshold}%, Fixed stop-loss = {stop_loss_fixed_pct}%, Trailing stop = {trailing_stop_pct}%")
                        st.write(f"Advanced parameters: Trailing stop activation = {trailing_stop_activation_pct}%, Use trailing stop as primary = {'Yes' if use_trailing_stop_main else 'No'}, Partial profit taking = {'Enabled' if partial_profit_taking else 'Disabled'}, Profit taking percentage = {profit_taking_percentage if partial_profit_taking else 'N/A'}%")
                        st.write(f"Multi-bot configuration: {number_of_bots} bots running as umbrella strategy, Max position duration = {max_position_duration} days, Trading fee = {trading_fee_pct}%")
                        st.write(f"Cooldown settings: Profit threshold = {profit_cooldown_threshold_pct}%, Time between trades = {min_time_between_trades_minutes} min, Time between buy/sell = {min_time_between_buy_sell_minutes} min")
                        interval_explanation = ""
                        if bot_action_interval_choice == "1min":
                            interval_explanation = "bots check and execute trades every 1 minute"
                        elif bot_action_interval_choice == "5min":
                            interval_explanation = "bots check and execute trades every 5 minutes"
                        elif bot_action_interval_choice == "15min":
                            interval_explanation = "bots check and execute trades every 15 minutes"
                        elif bot_action_interval_choice == "30min":
                            interval_explanation = "bots check and execute trades every 30 minutes"
                        elif bot_action_interval_choice == "1h":
                            interval_explanation = "bots check and execute trades every hour"
                        elif bot_action_interval_choice == "4h":
                            interval_explanation = "bots check and execute trades every 4 hours"
                        elif bot_action_interval_choice == "1d":
                            interval_explanation = "bots check and execute trades once per day"
                        
                        st.write(f"Timeframe resampling: Bot action interval = {bot_action_interval_choice} ({interval_explanation})")
                        st.info(f"⏰ **Time-Based Trading:** With the {bot_action_interval_choice} interval, your bots only execute trades at specific timeframes, reducing overtrading and potentially improving strategy performance.")
                    elif strategy_type == "Alternative DCA v2.0":
                        st.write(f"Parameters: Initial capital = {initial_capital}, Investment = {investment_pct}%, Price drop = {price_drop_threshold}%, Profit = {profit_threshold}%, Fixed stop-loss = {stop_loss_fixed_pct}%, Trailing stop = {trailing_stop_pct}%, ATR multiplier = {atr_stop_multiplier}, Trading fee = {trading_fee_pct}%")
                        st.write(f"Advanced parameters: MA period = {trend_filter_ma_period}, ATR window = {volatility_atr_window}, Max position duration = {max_position_duration} days, Min order size = ${min_order_size}, Trailing stop activation = {trailing_stop_activation_pct}%, Order threshold = {order_aggregation_threshold}%")
                    else:
                        st.write(f"Parameters: Initial capital = {initial_capital}, Investment = {investment_pct}%, Price drop = {price_drop_threshold}%, Profit = {profit_threshold}%, Trading fee = {trading_fee_pct}%")
                
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"An error occurred creating the strategy: {str(e)}")
                    st.code(error_details, language="python")
                    st.stop()  # Stop execution
                
                # Create backtester and store it in session state for later use
                backtester = Backtester(data=filtered_df, strategy=strategy)
                st.session_state.backtester = backtester
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run backtest
                status_text.text("Initializing backtest...")
                
                # Define progress callback function
                def update_progress(percent, message):
                    progress_bar.progress(percent / 100)
                    status_text.text(message)
                
                # Use a try-except block to handle potential errors
                try:
                    # Run the backtest with progress updates
                    results = backtester.run(progress_callback=update_progress)
                    
                    # Update final progress
                    progress_bar.progress(100)
                    status_text.text("Backtest completed successfully!")
                    
                    # Store results in session state
                    st.session_state.backtest_results = results
                
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"An error occurred during backtesting: {str(e)}")
                    st.code(error_details, language="python")
                    progress_bar.progress(100)
                    status_text.text("Backtest failed.")
            
            # Display backtest results if available
            if st.session_state.backtest_results is not None:
                results = st.session_state.backtest_results
                
                # Results tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Equity & Drawdown", "Trades", "Detailed Analysis"])
                
                # Tab 1: Summary
                with tab1:
                    st.markdown("### Backtest Results Summary")
                    
                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Return",
                            f"{results['return_pct']:.2f}%",
                            f"{results.get('outperformance', 0):.2f}% vs B&H"
                        )
                    
                    with col2:
                        st.metric(
                            "Max Drawdown",
                            f"{results['max_drawdown_pct']:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Final Equity",
                            f"${results['final_equity']:.2f}",
                            f"${results['final_equity'] - results['initial_capital']:.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Total Trades",
                            f"{results['trades_count']}"
                        )
                    
                    # Strategy parameters summary
                    st.markdown("### Strategy Parameters")
                    
                    # Display parameters in a clean table format based on strategy type
                    if strategy_type == "Liquidity-Managed DCA":
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Invest Increase',
                                'Invest Flat',
                                'Invest Drop Significant',
                                'Invest Drop Non-Significant',
                                'Drop Threshold',
                                'Exit Profit Margin',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"${invest_increase:.2f}",
                                f"${invest_flat:.2f}",
                                f"${invest_drop_significant:.2f}",
                                f"${invest_drop_non_significant:.2f}",
                                f"{drop_threshold:.1f}%",
                                f"{exit_profit_margin:.1f}%",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    elif strategy_type == "Interval DCA":
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Fixed DCA Amount',
                                'Investment Interval',
                                'Profit Target',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"${dca_amount:.2f}",
                                f"{interval_minutes} minutes",
                                f"{profit_target:.1f}%",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    elif strategy_type == "Hybrid DCA":
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Fixed DCA Amount',
                                'Investment Interval',
                                'Reserve Capital',
                                'Drop Threshold',
                                'Extra Buy Amount',
                                'Profit Target',
                                'Stop Loss',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"${dca_amount:.2f}",
                                f"{interval_minutes} minutes",
                                f"{reserve_pct:.1f}%",
                                f"{drop_threshold:.1f}%",
                                f"{extra_buy_pct:.1f}%",
                                f"{profit_target:.1f}%",
                                f"{stop_loss_pct:.1f}%",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    elif strategy_type == "Alternative DCA v3.0":
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Investment Amount',
                                'Price Drop Threshold',
                                'Profit Threshold',
                                'Fixed Stop-Loss',
                                'Trailing Stop',
                                'Max Position Duration',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{stop_loss_fixed_pct:.1f}%",
                                f"{trailing_stop_pct:.1f}%",
                                f"{max_position_duration} days",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    elif strategy_type == "Optimized Alternative DCA v4":
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Investment Amount',
                                'Price Drop Threshold',
                                'Profit Threshold',
                                'Fixed Stop-Loss',
                                'Trailing Stop',
                                'Max Position Duration',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{stop_loss_fixed_pct:.1f}%",
                                f"{trailing_stop_pct:.1f}%",
                                f"{max_position_duration} days",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    elif strategy_type == "Optimized Alternative DCA v5":
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
                                'Partial Profit Taking',
                                'Profit Taking Percentage',
                                'Max Position Duration',
                                'Number of Bots',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{stop_loss_fixed_pct:.1f}%",
                                f"{trailing_stop_pct:.1f}%",
                                f"{trailing_stop_activation_pct:.1f}%",
                                f"{'Yes' if use_trailing_stop_main else 'No'}",
                                f"{'Enabled' if partial_profit_taking else 'Disabled'}",
                                f"{profit_taking_percentage:.1f}%" if partial_profit_taking else "N/A",
                                f"{max_position_duration} days",
                                f"{number_of_bots} bots (umbrella strategy)",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    elif strategy_type == "Optimized Alternative DCA v6":
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
                                'Partial Profit Taking',
                                'Profit Taking Percentage',
                                'Max Position Duration',
                                'Number of Bots',
                                'Bot Action Interval',
                                # Cooldown Parameters
                                'Profit Cooldown Threshold',
                                'Min Time Between Trades',
                                'Min Time Between Buy/Sell',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{stop_loss_fixed_pct:.1f}%",
                                f"{trailing_stop_pct:.1f}%",
                                f"{trailing_stop_activation_pct:.1f}%",
                                f"{'Yes' if use_trailing_stop_main else 'No'}",
                                f"{'Enabled' if partial_profit_taking else 'Disabled'}",
                                f"{profit_taking_percentage:.1f}%" if partial_profit_taking else "N/A",
                                f"{max_position_duration} days",
                                f"{number_of_bots} bots (umbrella strategy)",
                                f"{bot_action_interval_choice}",
                                # Cooldown Parameter Values
                                f"{profit_cooldown_threshold_pct:.1f}%",
                                f"{min_time_between_trades_minutes} minutes",
                                f"{min_time_between_buy_sell_minutes} minutes",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    # Optimized Alternative DCA v5.1 has been removed
                    elif strategy_type == "Alternative DCA v2.0":
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Investment Amount',
                                'Price Drop Threshold',
                                'Profit Threshold',
                                'Fixed Stop-Loss',
                                'Trailing Stop',
                                'ATR Stop Multiplier',
                                'Max Position Duration',
                                'Trend Filter MA Period',
                                'ATR Volatility Window',
                                'Min Order Size',
                                'Trailing Stop Activation Threshold',
                                'Order Aggregation Threshold',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{stop_loss_fixed_pct:.1f}%",
                                f"{trailing_stop_pct:.1f}%",
                                f"{atr_stop_multiplier:.1f}",
                                f"{max_position_duration} days",
                                f"{trend_filter_ma_period} days",
                                f"{volatility_atr_window} days",
                                f"${min_order_size:.2f}",
                                f"{trailing_stop_activation_pct:.1f}%",
                                f"{order_aggregation_threshold:.1f}%",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    else:
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Investment Amount',
                                'Price Drop Threshold',
                                'Profit Threshold',
                                'Trading Fee',
                                'Start Date',
                                'End Date',
                                'Backtest Duration'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{trading_fee_pct:.2f}%",
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'),
                                f"{(end_date - start_date).days} days"
                            ]
                        })
                    
                    st.table(params_df)
                    
                    # Performance metrics
                    st.markdown("### Performance Metrics")
                    
                    # Create a metrics dataframe from available results
                    metrics_df = PerformancePlots.create_performance_metrics_table(results)
                    
                    # Group by category and display
                    for category in metrics_df['Category'].unique():
                        st.markdown(f"#### {category} Metrics")
                        category_metrics = metrics_df[metrics_df['Category'] == category]
                        st.table(category_metrics[['Metric', 'Value']])
                
                # Tab 2: Equity & Drawdown
                with tab2:
                    st.markdown("### Equity Curve")
                    
                    # For multi-bot strategies that include buy_hold_data, use that directly
                    if 'buy_hold_data' in results and results['buy_hold_data'] is not None:
                        buy_hold_equity = results['buy_hold_data']
                    else:
                        # For other strategies, calculate buy and hold equity for comparison
                        initial_btc = initial_capital / filtered_df.iloc[0]['close']
                        buy_hold_equity = pd.DataFrame(
                            {
                                'value': filtered_df['close'] * initial_btc,
                                'date': filtered_df.index
                            }, 
                            index=filtered_df.index
                        )
                    
                    # Create equity curve plot with normalized view
                    # Add a toggle for normalized view
                    normalize_equity = st.checkbox("Normalize equity curves (show percentage change)", value=True,
                                                 help="When checked, curves start at 100% for easier comparison. For multi-bot strategies, this shows accurate combined performance.")
                    
                    equity_fig = PerformancePlots.plot_equity_curve(
                        results['equity_curve'], 
                        buy_hold_equity,
                        normalized=normalize_equity
                    )
                    
                    st.plotly_chart(equity_fig, use_container_width=True, key="backtest_equity_curve")
                    
                    # Individual bot performance chart for V5 and V6 strategies
                    if strategy_type in ["Optimized Alternative DCA v5", "Optimized Alternative DCA v6"] and 'bot_results' in results:
                        st.markdown("### Individual Bot Performance")
                        
                        # Add info about the chart
                        st.info("This chart shows the performance of each individual bot. When normalized, all bots start at 100% of their individual capital regardless of when they were deployed, allowing direct comparison of relative performance between bots.")
                        
                        # Create the normalized plot of individual bot performance curves
                        bot_comparison_fig = PerformancePlots.plot_bot_comparison(
                            results['bot_results'],
                            normalized=normalize_equity
                        )
                        
                        st.plotly_chart(bot_comparison_fig, use_container_width=True, key="bot_comparison_chart")
                    
                    # Drawdown plot
                    st.markdown("### Drawdown Chart")
                    
                    drawdown_fig = PerformancePlots.plot_drawdown_chart(results['equity_curve'])
                    st.plotly_chart(drawdown_fig, use_container_width=True, key="backtest_drawdown_chart")
                    
                    # Monthly returns heatmap
                    st.markdown("### Monthly Returns")
                    
                    monthly_fig = PerformancePlots.plot_monthly_returns(results['equity_curve'])
                    st.plotly_chart(monthly_fig, use_container_width=True, key="backtest_monthly_returns")
                
                # Tab 3: Trades
                with tab3:
                    st.markdown("### Trade Analysis")
                    
                    # Get trade statistics from the backtester in session state
                    if 'backtester' in st.session_state:
                        trade_analysis = st.session_state.backtester.get_trade_analysis()
                    else:
                        trade_analysis = {"win_rate": 0, "avg_profit": 0, "best_trade": 0, "worst_trade": 0}
                    
                    # Display trade metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Win Rate", f"{trade_analysis.get('win_rate', 0):.1f}%")
                    
                    with col2:
                        st.metric("Avg. Profit", f"{trade_analysis.get('avg_profit', 0):.2f}%")
                    
                    with col3:
                        st.metric("Best Trade", f"{trade_analysis.get('best_trade', 0):.2f}%")
                    
                    with col4:
                        st.metric("Worst Trade", f"{trade_analysis.get('worst_trade', 0):.2f}%")
                    
                    # Create trades chart if there are trades available
                    if 'trades' in results and len(results['trades']) > 0:
                        trades_fig = PerformancePlots.plot_trades_chart(
                            filtered_df,
                            results['trades']
                        )
                        
                        st.plotly_chart(trades_fig, use_container_width=True, key="backtest_trades_chart")
                    else:
                        st.info("No trades data available to plot.")
                    
                    # Display trades table
                    st.markdown("### Trade List")
                    
                    if 'trades' not in results or len(results['trades']) == 0:
                        st.info("No trades were executed during the backtest period.")
                    else:
                        # Format trades for display
                        trades_df = results['trades'].copy()
                        
                        # First, select only the columns we want to display
                        selected_columns = []
                        
                        # Show bot number first if available (for multi-bot strategy)
                        if 'bot_number' in trades_df.columns:
                            selected_columns.append('bot_number')
                        elif 'bot_id' in trades_df.columns:
                            selected_columns.append('bot_id')
                        
                        # For buy trades, show basic info
                        if 'type' in trades_df.columns:
                            selected_columns.append('type')
                        
                        # Always include BTC amount and price 
                        if 'btc_amount' in trades_df.columns:
                            selected_columns.append('btc_amount')
                        if 'price' in trades_df.columns:
                            selected_columns.append('price')
                            
                        # For sell trades, add profit information
                        if 'profit_pct' in trades_df.columns:
                            selected_columns.append('profit_pct')
                        if 'profit_usdt' in trades_df.columns:
                            selected_columns.append('profit_usdt')
                            
                        # Add trade value and fees
                        if 'value' in trades_df.columns:
                            selected_columns.append('value')
                        if 'fee' in trades_df.columns:
                            selected_columns.append('fee')
                        
                        # Add holding period if available
                        if 'holding_period' in trades_df.columns:
                            selected_columns.append('holding_period')
                            
                        # Add cash information
                        if 'remaining_cash' in trades_df.columns:
                            selected_columns.append('remaining_cash')
                        elif 'cash_after' in trades_df.columns:
                            selected_columns.append('cash_after')
                            
                        # Filter columns that exist in the DataFrame
                        display_columns = [col for col in selected_columns if col in trades_df.columns]
                        trades_df = trades_df[display_columns].copy()
                        
                        # Format columns
                        if 'price' in trades_df.columns:
                            trades_df['price'] = trades_df['price'].map('${:.2f}'.format)
                        
                        if 'value' in trades_df.columns:
                            trades_df['value'] = trades_df['value'].map('${:.2f}'.format)
                            
                        if 'fee' in trades_df.columns:
                            trades_df['fee'] = trades_df['fee'].map('${:.2f}'.format)
                            
                        # Format bot number with more visible presentation
                        if 'bot_number' in trades_df.columns:
                            # Convert to string with leading zeroes if numeric
                            if pd.api.types.is_numeric_dtype(trades_df['bot_number']):
                                trades_df['bot_number'] = trades_df['bot_number'].astype(int).map(lambda x: f"Bot #{x}")
                            else:
                                trades_df['bot_number'] = trades_df['bot_number'].astype(str).map(lambda x: f"Bot #{x}" if x.isdigit() else x)
                        
                        # Handle different column names for remaining cash
                        if 'remaining_cash' in trades_df.columns:
                            trades_df['remaining_cash'] = trades_df['remaining_cash'].map('${:.2f}'.format)
                        elif 'cash_after' in trades_df.columns:
                            trades_df['cash_after'] = trades_df['cash_after'].map('${:.2f}'.format)
                        
                        if 'profit_pct' in trades_df.columns:
                            trades_df['profit_pct'] = trades_df['profit_pct'].map('{:.2f}%'.format)
                            
                        if 'profit_usdt' in trades_df.columns:
                            trades_df['profit_usdt'] = trades_df['profit_usdt'].map('${:.2f}'.format)
                            
                        if 'holding_period' in trades_df.columns:
                            trades_df['holding_period'] = trades_df['holding_period'].map('{:.1f}'.format)
                        
                        # Reset index to show date as a column
                        trades_df = trades_df.reset_index()
                        trades_df = trades_df.rename(columns={'index': 'date'})
                        
                        # Format the date column to be more readable
                        if 'date' in trades_df.columns:
                            # Make sure date is in datetime format
                            if pd.api.types.is_datetime64_any_dtype(trades_df['date']):
                                # Format date as readable string
                                trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(trades_df['date'].iloc[0], (int, float)):
                                # Handle numeric indices by showing actual count
                                trades_df['date'] = ['Trade #' + str(i+1) for i in range(len(trades_df))]
                        
                        # Improve column names for display
                        # Define column renames with more descriptive labels
                        column_renames = {
                            'date': 'Date/Time',
                            'type': 'Trade Type',
                            'price': 'Price (USD)',
                            'btc_amount': 'BTC Amount',
                            'value': 'Trade Value (USD)',
                            'fee': 'Fee (USD)',
                            'profit_pct': 'Profit %',
                            'profit_usdt': 'Profit (USD)',
                            'holding_period': 'Holding Period (days)',
                            'entry_price': 'Entry Price (USD)',
                            'exit_price': 'Exit Price (USD)',
                            'entry_time': 'Entry Time',
                            'exit_time': 'Exit Time',
                            'positions_count': 'Positions',
                            'bot_number': 'Bot #',  # Add bot number for tracking
                            'bot_id': 'Bot ID',     # Alternative bot identifier if present
                        }
                        
                        # Add appropriate cash column rename based on what's available
                        if 'remaining_cash' in trades_df.columns:
                            column_renames['remaining_cash'] = 'Cash After (USD)'
                        elif 'cash_after' in trades_df.columns:
                            column_renames['cash_after'] = 'Cash After (USD)'
                            
                        trades_df = trades_df.rename(columns=column_renames)
                        
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # Option to download trades as CSV
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="Download Trades as CSV",
                            data=csv,
                            file_name="backtest_trades.csv",
                            mime="text/csv"
                        )
                
                # Tab 4: Detailed Analysis
                with tab4:
                    st.markdown("### Detailed Performance Analysis")
                    
                    # Proceed with detailed performance analysis
                    if 'equity_curve' not in results:
                        st.info("No equity curve data available for analysis.")
                    else:
                        # Equity and BTC price overlay
                        st.markdown("#### Equity vs BTC Price")
                        
                        try:
                            # Create a figure with dual y-axes
                            equity_fig = go.Figure()
                            
                            # Add equity curve
                            equity_fig.add_trace(
                                go.Scatter(
                                    x=results['equity_curve'].index,
                                    y=results['equity_curve']['equity'],
                                    name='Strategy Equity',
                                    line=dict(color='#1f77b4', width=2)
                                )
                            )
                            
                            # Add BTC price as secondary y-axis
                            equity_fig.add_trace(
                                go.Scatter(
                                    x=filtered_df.index,
                                    y=filtered_df['close'],
                                    name='BTC Price',
                                    line=dict(color='#ff7f0e', width=1, dash='dash'),
                                    yaxis='y2'
                                )
                            )
                            
                            # Update layout with secondary y-axis
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
                            
                            st.plotly_chart(equity_fig, use_container_width=True, key="detailed_equity_vs_btc")
                        except Exception as fig_error:
                            st.warning(f"Could not display equity chart: {str(fig_error)}")
                        
                        # BTC Holdings over time
                        if 'btc_holdings' in results['equity_curve'].columns:
                            st.markdown("#### BTC Holdings Over Time")
                            
                            # Create BTC holdings chart
                            fig = go.Figure()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=results['equity_curve'].index,
                                    y=results['equity_curve']['btc_holdings'],
                                    name='BTC Holdings',
                                    line=dict(color='#2ca02c', width=2),
                                    fill='tozeroy'
                                )
                            )
                            
                            fig.update_layout(
                                title='BTC Holdings Over Time',
                                xaxis_title='Date',
                                yaxis_title='BTC Amount',
                                template='plotly_white',
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="btc_holdings_over_time")
                        
                        # Cash allocation over time
                        st.markdown("#### Cash vs. Position Allocation")
                        
                        try:
                            # Check if required columns exist in equity_curve
                            if 'cash' in results['equity_curve'].columns and 'position_value' in results['equity_curve'].columns:
                                # Create cash allocation chart
                                fig = go.Figure()
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=results['equity_curve'].index,
                                        y=results['equity_curve']['cash'],
                                        name='Cash',
                                        stackgroup='one',
                                        line=dict(color='#1f77b4')
                                    )
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=results['equity_curve'].index,
                                        y=results['equity_curve']['position_value'],
                                        name='BTC Position Value',
                                        stackgroup='one',
                                        line=dict(color='#ff7f0e')
                                    )
                                )
                                
                                fig.update_layout(
                                    title='Cash vs. BTC Position Allocation',
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
                                
                                st.plotly_chart(fig, use_container_width=True, key="cash_vs_position")
                            else:
                                st.info("Required columns (cash and position_value) not found in equity curve data.")
                        except Exception as e:
                            st.warning(f"Error creating cash allocation chart: {str(e)}")
                        
                        # Equity curve distribution
                        st.markdown("#### Equity Return Distribution")
                        
                        # Calculate daily returns
                        if 'daily_return' not in results['equity_curve'].columns:
                            results['equity_curve']['daily_return'] = results['equity_curve']['equity'].pct_change() * 100
                        
                        # Plot return distribution
                        daily_returns = results['equity_curve']['daily_return'].dropna().tolist()
                        
                        if daily_returns:
                            returns_fig = PerformancePlots.plot_return_distribution(
                                daily_returns,
                                "Daily Return Distribution (%)"
                            )
                            
                            st.plotly_chart(returns_fig, use_container_width=True, key="return_distribution")

                # Option to save backtest results for comparison
                st.markdown("### Save Backtest Results")
                
                save_name = st.text_input(
                    "Name for this backtest",
                    value=f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    max_chars=50
                )
                
                # Create two columns for different save operations
                col1, col2 = st.columns(2)
                
                with col1:
                    save_to_session = st.button("Save to Session", help="Save results to current session for comparison. Results will be available until you refresh or close the page.")
                
                with col2:
                    save_to_disk = st.button("💾 Save to Disk", help="Save results permanently to disk. Results will be available even after restarting the application.")
                
                if save_to_session:
                    # Create a dictionary with essential results and parameters
                    saved_result = {
                        'name': save_name,
                        'timestamp': datetime.now(),
                        'parameters': {
                            'strategy_type': strategy_type,
                            'initial_capital': initial_capital,
                            'trading_fee_pct': trading_fee_pct
                        },
                        'data_file': "Selected BTC data",
                        'date_range': f"{filtered_df.index.min().date()} to {filtered_df.index.max().date()}",
                        'results': {
                            'return_pct': results['return_pct'],
                            'buy_hold_return_pct': results['buy_hold_return_pct'],
                            'max_drawdown_pct': results['max_drawdown_pct'],
                            'trades_count': results['trades_count'],
                            'final_equity': results['final_equity'],
                            'sharpe_ratio': results.get('sharpe_ratio', 0),
                            'win_rate': results.get('win_rate', 0),
                            'equity_curve': results['equity_curve'].to_dict()
                        }
                    }
                    
                    # Initialize the saved_backtests list if it doesn't exist
                    if 'saved_backtests' not in st.session_state:
                        st.session_state.saved_backtests = []
                    
                    # Add the result to the session state
                    st.session_state.saved_backtests.append(saved_result)
                    
                    st.success(f"Backtest '{save_name}' saved successfully! View it in the Results page.")
                
                if save_to_disk:
                    try:
                        # Import needed modules
                        from backtesting.backtester import Backtester
                        
                        # Create a new backtester instance
                        backtester = Backtester()
                        
                        # Set the results
                        backtester.results = results
                        
                        # Extract strategy name from strategy_type
                        strategy_name = strategy_type
                        
                        # Create parameters dictionary
                        params = {
                            'strategy_type': strategy_type,
                            'initial_capital': initial_capital,
                            'investment_pct': investment_pct if 'investment_pct' in locals() else 0.0,
                            'price_drop_threshold': price_drop_threshold if 'price_drop_threshold' in locals() else 0.0,
                            'profit_threshold': profit_threshold if 'profit_threshold' in locals() else 0.0,
                            'trading_fee_pct': trading_fee_pct
                        }
                        
                        # Add strategy-specific parameters if they exist
                        if strategy_type == "Optimized Alternative DCA v4" or strategy_type == "Optimized Alternative DCA v5":
                            if 'stop_loss_fixed_pct' in locals():
                                params['stop_loss_fixed_pct'] = stop_loss_fixed_pct
                            if 'trailing_stop_pct' in locals():
                                params['trailing_stop_pct'] = trailing_stop_pct
                            if 'max_position_duration' in locals():
                                params['max_position_duration'] = max_position_duration
                            
                        # Save results to disk
                        filepath = backtester.save_results(strategy_name, params)
                        
                        if filepath:
                            st.success(f"✅ Backtest '{save_name}' saved successfully to disk! View it in the Results page.")
                        else:
                            st.error("❌ Failed to save results to disk. Check logs for details.")
                    except Exception as e:
                        st.error(f"❌ Error saving results to disk: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
