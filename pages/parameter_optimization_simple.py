"""
Simplified Parameter Optimization Page

This page provides a streamlined interface for optimizing trading strategies
using the simplified parameter optimizer.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import sys
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.data_validator import DataValidator
from utils.results_manager import ResultsManager
from backtesting.parameter_optimizer_simple import ParameterOptimizer
from visualization.performance_plots import PerformancePlots

st.title("📊 Simplified Parameter Optimization")

st.markdown("""
This tool allows you to optimize strategy parameters more efficiently using 
our simplified optimizer. Select your parameter ranges and find the best combination.
""")

# Sidebar for data and optimization settings
with st.sidebar:
    st.header("Data Settings")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Use saved data file", "Fetch new data"],
        index=0
    )
    
    if data_source == "Use saved data file":
        # List available data files
        data_fetcher = DataFetcher()
        saved_files = data_fetcher.list_saved_data()
        
        if not saved_files:
            st.warning("No saved data files found.")
            st.stop()
        
        # Create a list of file options with descriptive labels
        file_options = [f"{f['exchange']} - {f['symbol']} - {f['timeframe']} ({os.path.basename(f['filepath'])})" 
                      for f in saved_files]
        
        selected_file_idx = st.selectbox(
            "Select saved data file",
            range(len(file_options)),
            format_func=lambda i: file_options[i]
        )
        
        # Load the selected data file
        selected_file = saved_files[selected_file_idx]
        st.info(f"Using data file: {os.path.basename(selected_file['filepath'])}")
        
        # Load data
        data = data_fetcher.load_data(selected_file['filepath'])
        
        if data is not None:
            st.success(f"Loaded {len(data)} data points from {selected_file['exchange']} for {selected_file['symbol']}")
        else:
            st.error("Failed to load data file.")
            st.stop()
    else:
        st.info("Fetching new data functionality will be implemented in a future update.")
        st.stop()
    
    # Date range selection
    st.subheader("Date Range")
    date_range_option = st.radio(
        "Select date range",
        ["Use full dataset", "Specify date range"],
        index=0
    )
    
    if date_range_option == "Specify date range":
        min_date = data.index.min().date()
        max_date = data.index.max().date()
        
        start_date = st.date_input(
            "Start date",
            value=min_date + timedelta(days=30),
            min_value=min_date,
            max_value=max_date - timedelta(days=7)
        )
        
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=start_date + timedelta(days=7),
            max_value=max_date
        )
        
        # Filter data based on selected date range
        filtered_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)].copy()
        st.info(f"Using {len(filtered_data)} data points from {start_date} to {end_date}")
    else:
        filtered_data = data.copy()
        st.info(f"Using full dataset with {len(filtered_data)} data points")
    
    # Strategy selection
    st.header("Strategy Settings")
    
    strategy_type = st.selectbox(
        "Strategy Type",
        ["Optimized Alternative DCA v5", "Optimized Alternative DCA v6"]
    )
    
    # Initial capital
    initial_capital = st.number_input(
        "Initial Capital (USDT)",
        min_value=100.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0
    )
    
    # Trading fee percentage
    trading_fee_pct = st.number_input(
        "Trading Fee (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        format="%.2f"
    )
    
    # Optimization metric
    optimize_metric = st.selectbox(
        "Optimize For",
        ["return_pct", "sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor"]
    )
    
    # Limit on parameter combinations to test
    limit_combinations = st.checkbox("Limit parameter combinations", value=True)
    max_combinations = st.number_input(
        "Maximum combinations to test",
        min_value=10,
        max_value=10000,
        value=500,
        step=100,
        disabled=not limit_combinations
    )

# Main panel - Parameter configuration
st.header("Parameter Optimization Settings")

tab1, tab2 = st.tabs(["Basic Parameters", "Advanced Parameters"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Investment percentage range
        min_investment = st.number_input("Min Investment (%)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
        max_investment = st.number_input("Max Investment (%)", min_value=min_investment, max_value=100.0, value=30.0, step=5.0)
        investment_step = st.number_input("Investment Step", min_value=1.0, max_value=10.0, value=5.0, step=1.0)
    
    with col2:
        # Price drop threshold range
        min_drop = st.number_input("Min Price Drop (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        max_drop = st.number_input("Max Price Drop (%)", min_value=min_drop, max_value=10.0, value=3.0, step=0.5)
        drop_step = st.number_input("Drop Step", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    # Profit threshold range
    col1, col2 = st.columns(2)
    
    with col1:
        min_profit = st.number_input("Min Profit Threshold (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
        max_profit = st.number_input("Max Profit Threshold (%)", min_value=min_profit, max_value=20.0, value=5.0, step=0.5)
        profit_step = st.number_input("Profit Step", min_value=0.5, max_value=1.0, value=1.0, step=0.5)
    
    with col2:
        # Risk management parameters
        optimize_risk = st.checkbox("Optimize Risk Management", value=True)
        
        if optimize_risk:
            min_stop_loss = st.number_input("Min Stop Loss (%)", min_value=5.0, max_value=20.0, value=10.0, step=1.0)
            max_stop_loss = st.number_input("Max Stop Loss (%)", min_value=min_stop_loss, max_value=50.0, value=20.0, step=2.0)
            
            min_trailing_stop = st.number_input("Min Trailing Stop (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
            max_trailing_stop = st.number_input("Max Trailing Stop (%)", min_value=min_trailing_stop, max_value=5.0, value=2.0, step=0.5)

with tab2:
    # Multi-bot settings for v5 and v6
    optimize_num_bots = st.checkbox("Optimize Number of Bots", value=True)
    
    if optimize_num_bots:
        min_bots = st.number_input("Min Bots", min_value=1, max_value=5, value=1)
        max_bots = st.number_input("Max Bots", min_value=min_bots, max_value=10, value=3)
    
    # For v6 only - Action intervals
    if strategy_type == "Optimized Alternative DCA v6":
        optimize_interval = st.checkbox("Optimize Action Intervals", value=True)
        
        if optimize_interval:
            st.info("Select trading intervals to test")
            bot_action_intervals = st.multiselect(
                "Bot Action Intervals",
                ["1min", "5min", "15min", "30min", "1h", "4h", "1d"],
                default=["15min", "1h", "1d"]
            )

# Create parameter grid
if st.button("Run Optimization", key="run_optimization_button"):
    if filtered_data is None or len(filtered_data) == 0:
        st.error("No data available for optimization.")
        st.stop()
    
    # Create parameter grid
    param_grid = {
        'strategy_type': [strategy_type],
        'initial_capital': [initial_capital],
        'investment_pct': np.arange(min_investment, max_investment + investment_step, investment_step).tolist(),
        'price_drop_threshold': np.arange(min_drop, max_drop + drop_step, drop_step).tolist(),
        'profit_threshold': np.arange(min_profit, max_profit + profit_step, profit_step).tolist(),
        'trading_fee_pct': [trading_fee_pct]
    }
    
    # Add risk management parameters if selected
    if optimize_risk:
        param_grid['stop_loss_fixed_pct'] = np.arange(min_stop_loss, max_stop_loss + 2.0, 2.0).tolist()
        param_grid['trailing_stop_pct'] = np.arange(min_trailing_stop, max_trailing_stop + 0.5, 0.5).tolist()
        param_grid['max_position_duration'] = [90, 180]  # 3 months, 6 months
    
    # Add multi-bot parameters if selected
    if optimize_num_bots:
        param_grid['number_of_bots'] = list(range(min_bots, max_bots + 1))
    
    # Add action intervals for v6 if selected
    if strategy_type == "Optimized Alternative DCA v6":
        # Always set default bot_action_interval if not explicitly optimizing
        if optimize_interval:
            param_grid['bot_action_interval'] = bot_action_intervals
        else:
            param_grid['bot_action_interval'] = ["15min"]  # Default value
    
    # Show parameter grid summary
    total_combinations = 1
    for key, values in param_grid.items():
        total_combinations *= len(values)
    
    st.info(f"Testing {total_combinations} parameter combinations")
    
    # Limit combinations if selected
    if limit_combinations and total_combinations > max_combinations:
        st.warning(f"Limiting to {max_combinations} combinations (from {total_combinations})")
        # Reduce parameter ranges proportionally
        reduction_factor = max_combinations / total_combinations
        param_names = ['investment_pct', 'price_drop_threshold', 'profit_threshold']
        for param in param_names:
            if param in param_grid and len(param_grid[param]) > 3:
                orig_len = len(param_grid[param])
                new_len = max(3, int(orig_len * reduction_factor))
                param_grid[param] = param_grid[param][::(orig_len // new_len)]
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(data=filtered_data)
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(percent, message):
        progress_bar.progress(percent / 100)
        status_text.text(message)
    
    # Run grid search
    start_time = time.time()
    optimization_results = optimizer.grid_search(
        param_grid=param_grid,
        optimize_metric=optimize_metric,
        progress_callback=update_progress
    )
    end_time = time.time()
    
    # Update final progress
    progress_bar.progress(100)
    status_text.text(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Display results
    if optimization_results:
        st.header("Optimization Results")
        
        # Show top 10 results in a table
        st.subheader("Top Parameter Combinations")
        
        # Convert results to DataFrame for display
        results_df = pd.DataFrame([
            {
                'Investment (%)': r['params']['investment_pct'],
                'Price Drop (%)': r['params']['price_drop_threshold'],
                'Profit Threshold (%)': r['params']['profit_threshold'],
                'Return (%)': r['return_pct'],
                'Max Drawdown (%)': r['max_drawdown_pct'],
                'Sharpe': r.get('sharpe_ratio', 0),
                'Win Rate (%)': r.get('win_rate', 0) * 100,
                'Trades': r['trades_count']
            }
            for r in optimization_results[:10]
        ])
        
        st.dataframe(results_df)
        
        # Show best parameters in detail
        st.subheader("Best Parameters")
        best_params = optimization_results[0]['params']
        best_result = optimization_results[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strategy Parameters:**")
            for param, value in best_params.items():
                if param != 'strategy_type' and param != 'trading_fee_pct':
                    st.write(f"- {param}: {value}")
        
        with col2:
            st.write("**Performance Metrics:**")
            st.write(f"- Return: {best_result['return_pct']:.2f}%")
            st.write(f"- Max Drawdown: {best_result['max_drawdown_pct']:.2f}%")
            st.write(f"- Sharpe Ratio: {best_result.get('sharpe_ratio', 0):.2f}")
            st.write(f"- Win Rate: {best_result.get('win_rate', 0) * 100:.2f}%")
            st.write(f"- Number of Trades: {best_result['trades_count']}")
        
        # Run backtest with best parameters button
        if st.button("Run Backtest with Best Parameters"):
            st.session_state.best_params = best_params
            st.session_state.optimize_results = optimization_results
            st.info("Redirecting to backtest page with optimal parameters...")
            # In a real implementation, you would redirect or initialize the backtest here
        
        # Save results button
        if st.button("Save Optimization Results"):
            save_path = optimizer.save_results(strategy_type)
            if save_path:
                st.success(f"Results saved to {save_path}")
            else:
                st.error("Failed to save results")
    else:
        st.error("Optimization failed or no valid results returned.")