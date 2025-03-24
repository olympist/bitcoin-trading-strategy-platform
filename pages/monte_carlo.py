import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, date
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_fetcher import DataFetcher
from strategy.variable_timeframe_dca import VariableTimeframeDCA
from strategy.alternative_dca import AlternativeDCA
from strategy.optimized_alternative_dca import OptimizedAlternativeDCA
from strategy.optimized_alternative_dca_v5 import OptimizedAlternativeDCAv5
from strategy.fixed_investment_dca import FixedInvestmentDCA
from strategy.liquidity_managed_dca import LiquidityManagedDCA
from backtesting.monte_carlo import MonteCarloSimulator
from visualization.performance_plots import PerformancePlots

st.set_page_config(
    page_title="Monte Carlo Simulation - Bitcoin Backtesting Platform",
    page_icon="🎲",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data_files' not in st.session_state:
    st.session_state.data_files = []
if 'selected_data' not in st.session_state:
    st.session_state.selected_data = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'monte_carlo_results' not in st.session_state:
    st.session_state.monte_carlo_results = None

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
st.markdown('<p class="section-header">Monte Carlo Simulation</p>', unsafe_allow_html=True)
st.markdown("Test the robustness of your trading strategy through Monte Carlo simulations.")

# Sidebar for data selection and simulation settings
with st.sidebar:
    st.header("Simulation Configuration")
    
    # Data selection section
    st.subheader("1. Select Data")
    
    # If data files list is empty, load it
    if not st.session_state.data_files:
        st.session_state.data_files = data_fetcher.list_saved_data()
    
    if not st.session_state.data_files:
        st.warning("No data files found. Please go to the Data Manager to fetch data first.")
    else:
        # Create a dataframe for selection
        df_files = pd.DataFrame(st.session_state.data_files)
        
        # File selection
        selected_file = st.selectbox(
            "Data File",
            options=df_files['filename'].tolist(),
            format_func=lambda x: f"{x} ({df_files.loc[df_files['filename'] == x, 'exchange'].iloc[0]}, {df_files.loc[df_files['filename'] == x, 'timeframe'].iloc[0]})"
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
            st.error(f"Failed to load data from {selected_file}.")
        else:
            # Store in session state
            st.session_state.selected_data = df
            
            # Show data info
            st.info(f"Loaded {len(df)} data points from {df.index.min().date()} to {df.index.max().date()}.")
    
    # Strategy parameters section
    st.subheader("2. Strategy Parameters")
    
    with st.form("strategy_params_form"):
        # Strategy selection
        strategy_type = st.selectbox(
            "Strategy Type",
            options=["Original DCA", "Alternative DCA", "Optimized Alternative DCA", "Optimized Alternative DCA v4", "Optimized Alternative DCA v5", "Fixed Investment DCA", "Liquidity-Managed DCA"],
            index=4,  # Select "Optimized Alternative DCA v5" by default
            help="Select which DCA strategy variant to use for simulations"
        )
        
        # Common parameter - Initial Capital
        initial_capital = st.number_input(
            "Initial Capital (USDT)",
            min_value=100.0,
            max_value=1000000.0,
            value=1800.0 if 'best_params' not in st.session_state else st.session_state.best_params['initial_capital'],
            step=100.0
        )
        
        # Show different parameter inputs based on strategy type
        if strategy_type == "Optimized Alternative DCA":
            # Optimized Alternative DCA specific parameters
            st.markdown("#### Core Strategy Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                investment_pct = st.slider(
                    "Investment Amount (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0
                )
                
                price_drop_threshold = st.slider(
                    "Price Drop Threshold (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5
                )
            
            with col2:
                profit_threshold = st.slider(
                    "Profit Threshold (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
            
            # Advanced parameters with expandable section
            with st.expander("Advanced Risk Management Parameters"):
                st.markdown("##### Stop-Loss Settings")
                col1, col2 = st.columns(2)
                
                with col1:
                    stop_loss_fixed_pct = st.slider(
                        "Fixed Stop-Loss (%)",
                        min_value=5.0,
                        max_value=25.0,
                        value=15.0,
                        step=1.0,
                        help="Fixed percentage stop-loss from average entry price"
                    )
                    
                    atr_stop_multiplier = st.slider(
                        "ATR Stop Multiplier",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.5,
                        help="Multiplier for ATR-based stop-loss"
                    )
                
                with col2:
                    trailing_stop_pct = st.slider(
                        "Trailing Stop (%)",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.5,
                        step=0.5,
                        help="Trailing stop percentage that follows price up"
                    )
                    
                    max_position_duration = st.slider(
                        "Max Position Duration (days)",
                        min_value=30,
                        max_value=365,
                        value=180,
                        step=30,
                        help="Maximum days to hold a position before forced exit"
                    )
                
                st.markdown("##### Market & Volatility Filters")
                col1, col2 = st.columns(2)
                
                with col1:
                    trend_filter_ma_period = st.slider(
                        "Trend Filter MA Period",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=5,
                        help="Moving average period for trend filter"
                    )
                    
                    volatility_atr_window = st.slider(
                        "Volatility ATR Window",
                        min_value=5,
                        max_value=30,
                        value=14,
                        step=1,
                        help="ATR window for volatility measurement"
                    )
                
                with col2:
                    min_order_size = st.slider(
                        "Minimum Order Size (USDT)",
                        min_value=10.0,
                        max_value=100.0,
                        value=50.0,
                        step=10.0,
                        help="Minimum investment size per order"
                    )
                    
                    trailing_stop_activation_pct = st.slider(
                        "Trailing Stop Activation (%)",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.5,
                        step=0.5,
                        help="Profit percentage at which trailing stop becomes active"
                    )
                    
                    order_aggregation_threshold = st.slider(
                        "Order Aggregation Threshold (%)",
                        min_value=0.5,
                        max_value=5.0,
                        value=2.0,
                        step=0.5,
                        help="Threshold for aggregating orders close in price"
                    )
        
        elif strategy_type == "Liquidity-Managed DCA":
            # Liquidity-Managed DCA specific parameters
            st.markdown("#### Strategy Parameters")
            st.markdown("Configure the investment amounts for different market conditions:")
            
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
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
            col1, col2 = st.columns(2)
            
            with col1:
                drop_threshold = st.slider(
                    "U: Drop Threshold (%)",
                    min_value=0.1, 
                    value=2.0, 
                    step=0.1,
                    help="Threshold percentage (U%) for defining a significant drop"
                )
            
            with col2:
                exit_profit_margin = st.slider(
                    "K: Exit Profit Target (%)",
                    min_value=0.1, 
                    value=2.0,  # Updated from 1.0% to 2.0% to match optimized value
                    step=0.1,
                    help="Target profit percentage (K%) that triggers selling"
                )
        else:
            # Standard parameters for other strategies
            # Use best parameters from optimization if available
            if 'best_params' in st.session_state:
                st.success("Using optimized parameters!")
                
                investment_pct = st.slider(
                    "Investment Amount (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=st.session_state.best_params['investment_pct'],
                    step=1.0
                )
                
                price_drop_threshold = st.slider(
                    "Price Drop Threshold (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=st.session_state.best_params['price_drop_threshold'],
                    step=0.5
                )
                
                profit_threshold = st.slider(
                    "Profit Threshold (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=st.session_state.best_params['profit_threshold'],
                    step=0.5
                )
            else:
                investment_pct = st.slider(
                    "Investment Amount (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0
                )
                
                price_drop_threshold = st.slider(
                    "Price Drop Threshold (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5
                )
                
                profit_threshold = st.slider(
                    "Profit Threshold (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
        
        # Common parameters across all strategies
        trading_fee_pct = st.slider(
            "Trading Fee (%)",
            min_value=0.01,
            max_value=1.0,
            value=0.1 if 'best_params' not in st.session_state else st.session_state.best_params['trading_fee_pct'],
            step=0.01
        )
        
        # Monte Carlo specific settings
        st.markdown("#### Monte Carlo Settings")
        
        simulation_method = st.selectbox(
            "Simulation Method",
            options=["bootstrap", "block_bootstrap", "permutation"],
            format_func=lambda x: {
                "bootstrap": "Bootstrap Resampling",
                "block_bootstrap": "Block Bootstrap",
                "permutation": "Price Permutation"
            }.get(x, x),
            index=0,
            help="Method used to generate simulated price data"
        )
        
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of Monte Carlo simulations to run"
        )
        
        use_multiprocessing = st.checkbox(
            "Use Multiprocessing", 
            value=True,
            help="Enables parallel processing for faster simulations"
        )
        
        run_simulation_button = st.form_submit_button("Run Monte Carlo Simulation")
    
    # Date range filter
    if st.session_state.selected_data is not None:
        st.subheader("3. Date Range (Optional)")
        
        df = st.session_state.selected_data
        date_min = df.index.min().date()
        date_max = df.index.max().date()
        
        start_date = st.date_input(
            "Start Date",
            value=date_min,
            min_value=date_min,
            max_value=date_max
        )
        
        end_date = st.date_input(
            "End Date",
            value=date_max,
            min_value=date_min,
            max_value=date_max
        )

# Main content area
if st.session_state.selected_data is None:
    st.warning("Please select a data file from the sidebar to begin.")
else:
    # Apply date filter if specified
    df = st.session_state.selected_data
    
    try:
        # Convert dates to timestamps for filtering
        if 'start_date' in locals() and 'end_date' in locals():
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_df = df.loc[mask]
        else:
            filtered_df = df
            
        if filtered_df.empty:
            st.error("No data in selected date range.")
        else:
            # Import dependencies needed for Monte Carlo simulation
            import time
            import logging
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            # Get logger for status updates
            logger = logging.getLogger(__name__)
            
            # Define custom run_with_progress function
            def run_with_progress(simulator, method, num_simulations, strategy_params, use_multiprocessing, progress_bar, progress_text):
                """Run simulations with progress updates"""
                
                # Setup
                start_time = time.time()
                results = []
                progress_step = 100 / num_simulations
                current_progress = 0
                
                # Choose the appropriate data generator based on method
                if method == "bootstrap":
                    generate_data = simulator._generate_bootstrap_sample
                elif method == "block_bootstrap":
                    generate_data = lambda df, seed: simulator._generate_block_bootstrap_sample(df, block_size=10)
                elif method == "permutation":
                    generate_data = simulator._generate_permutation_sample
                else:
                    generate_data = simulator._generate_bootstrap_sample  # Default to bootstrap
                
                logger.info(f"Running {num_simulations} Monte Carlo simulations using {method} method")
                
                # Run simulations
                if use_multiprocessing:
                    with ProcessPoolExecutor() as executor:
                        futures = []
                        for i in range(num_simulations):
                            seed = i + 1  # Use a different seed for each simulation
                            sim_data = generate_data(simulator.data, seed)
                            futures.append(executor.submit(simulator._run_simulation, sim_data, strategy_params, seed))
                        
                        completed = 0
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                results.append(result)
                            completed += 1
                            
                            # Update progress
                            current_progress = (completed / num_simulations) * 100
                            progress_bar.progress(int(min(current_progress, 100)))
                            progress_text.text(f"Running simulations: {completed}/{num_simulations} complete ({int(current_progress)}%)")
                else:
                    for i in range(num_simulations):
                        seed = i + 1  # Use a different seed for each simulation
                        sim_data = generate_data(simulator.data, seed)
                        result = simulator._run_simulation(sim_data, strategy_params, seed)
                        if result:
                            results.append(result)
                        
                        # Update progress
                        current_progress += progress_step
                        progress_bar.progress(int(min(current_progress, 100)))
                        progress_text.text(f"Running simulations: {i+1}/{num_simulations} complete ({int(current_progress)}%)")
                
                end_time = time.time()
                logger.info(f"Monte Carlo simulations completed in {end_time - start_time:.2f} seconds")
                
                if results:
                    simulator.simulation_results = results
                    simulator._calculate_summary_statistics()
                
                return results
            
            
            # Run Monte Carlo simulation if button clicked
            if run_simulation_button:
                # Create a progress container
                progress_container = st.empty()
                progress_container.info("Initializing Monte Carlo simulations...")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                progress_text = st.empty()
                progress_text.text("Running simulations: 0% complete")
                
                # Create strategy parameters dictionary based on strategy type
                if strategy_type == "Liquidity-Managed DCA":
                    strategy_params = {
                        'strategy_type': strategy_type,
                        'initial_capital': initial_capital,
                        'exit_profit_margin': exit_profit_margin,
                        'invest_increase': invest_increase,
                        'invest_flat': invest_flat,
                        'invest_drop_significant': invest_drop_significant,
                        'invest_drop_non_significant': invest_drop_non_significant,
                        'drop_threshold': drop_threshold,
                        'trading_fee_pct': trading_fee_pct
                    }
                elif strategy_type == "Optimized Alternative DCA":
                    strategy_params = {
                        'strategy_type': strategy_type,
                        'initial_capital': initial_capital,
                        'investment_pct': investment_pct,
                        'price_drop_threshold': price_drop_threshold,
                        'profit_threshold': profit_threshold,
                        'trading_fee_pct': trading_fee_pct,
                        # Advanced risk management parameters
                        'stop_loss_fixed_pct': stop_loss_fixed_pct,
                        'atr_stop_multiplier': atr_stop_multiplier,
                        'trailing_stop_pct': trailing_stop_pct,
                        'max_position_duration': max_position_duration,
                        'trend_filter_ma_period': trend_filter_ma_period,
                        'volatility_atr_window': volatility_atr_window,
                        'min_order_size': min_order_size,
                        'order_aggregation_threshold': order_aggregation_threshold,
                        'trailing_stop_activation_pct': trailing_stop_activation_pct
                    }
                else:
                    strategy_params = {
                        'strategy_type': strategy_type,
                        'initial_capital': initial_capital,
                        'investment_pct': investment_pct,
                        'price_drop_threshold': price_drop_threshold,
                        'profit_threshold': profit_threshold,
                        'trading_fee_pct': trading_fee_pct
                    }
                
                # Create Monte Carlo simulator
                simulator = MonteCarloSimulator(data=filtered_df)
                
                # Run simulation
                progress_text.text(f"Running {num_simulations} Monte Carlo simulations...")
                
                # Use a try-except block to handle potential errors
                try:
                    # Run simulations with progress updates
                    simulation_results = run_with_progress(
                        simulator, 
                        simulation_method, 
                        num_simulations, 
                        strategy_params, 
                        use_multiprocessing,
                        progress_bar,
                        progress_text
                    )
                    
                    # Update final progress
                    progress_bar.progress(100)
                    progress_text.text("Monte Carlo simulation completed successfully!")
                    
                    # Store results in session state
                    st.session_state.monte_carlo_results = simulation_results
                    st.session_state.monte_carlo_summary = simulator.get_summary()
                    st.session_state.monte_carlo_report = simulator.create_simulation_report()
                
                except Exception as e:
                    st.error(f"An error occurred during simulation: {str(e)}")
                    progress_bar.progress(100)
                    progress_text.text("Simulation failed.")
            
            # Display simulation results if available
            if st.session_state.monte_carlo_results is not None:
                simulation_results = st.session_state.monte_carlo_results
                summary = st.session_state.monte_carlo_summary
                report = st.session_state.monte_carlo_report
                
                # Results tabs
                tab1, tab2, tab3 = st.tabs(["Simulation Summary", "Return Distribution", "Risk Analysis"])
                
                # Tab 1: Simulation Summary
                with tab1:
                    st.markdown("### Monte Carlo Simulation Results")
                    
                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Mean Return",
                            f"{summary['return_pct']['mean']:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Median Return",
                            f"{summary['return_pct']['median']:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Probability of Profit",
                            f"{summary['probability_of_profit']:.1f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "Mean Final Equity",
                            f"${summary['final_equity']['mean']:.2f}"
                        )
                    
                    # Display equity curves
                    st.markdown("### Equity Curves")
                    
                    equity_fig = PerformancePlots.plot_monte_carlo_equity_curves(
                        simulation_results,
                        num_curves=20
                    )
                    
                    st.plotly_chart(equity_fig, use_container_width=True, key="monte_carlo_equity_curves")
                    
                    # Strategy parameters summary
                    st.markdown("### Strategy Parameters Used")
                    
                    # Display parameters in a clean table format based on strategy type
                    if strategy_type == "Liquidity-Managed DCA":
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'X: Invest When Price Increases',
                                'Y: Invest When Price is Flat',
                                'Z: Invest On Significant Drops',
                                'W: Invest On Non-Significant Drops',
                                'U: Drop Threshold',
                                'K: Exit Profit Target',
                                'Trading Fee',
                                'Number of Simulations',
                                'Simulation Method'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{invest_increase:.2f}%",
                                f"{invest_flat:.2f}%",
                                f"{invest_drop_significant:.2f}%",
                                f"{invest_drop_non_significant:.2f}%",
                                f"{drop_threshold:.1f}%",
                                f"{exit_profit_margin:.1f}%",
                                f"{trading_fee_pct:.2f}%",
                                f"{num_simulations}",
                                f"{simulation_method}"
                            ]
                        })
                    elif strategy_type == "Optimized Alternative DCA":
                        # Create two DataFrames - one for core params, one for advanced params
                        core_params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Investment Amount',
                                'Price Drop Threshold',
                                'Profit Threshold',
                                'Trading Fee'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{trading_fee_pct:.2f}%"
                            ]
                        })
                        
                        advanced_params_df = pd.DataFrame({
                            'Parameter': [
                                'Fixed Stop-Loss',
                                'ATR Stop Multiplier',
                                'Trailing Stop',
                                'Max Position Duration',
                                'Trend Filter MA Period',
                                'Volatility ATR Window',
                                'Minimum Order Size',
                                'Trailing Stop Activation',
                                'Order Aggregation Threshold'
                            ],
                            'Value': [
                                f"{stop_loss_fixed_pct:.1f}%",
                                f"{atr_stop_multiplier:.1f}",
                                f"{trailing_stop_pct:.1f}%",
                                f"{max_position_duration} days",
                                f"{trend_filter_ma_period} periods",
                                f"{volatility_atr_window} periods",
                                f"${min_order_size:.2f}",
                                f"{trailing_stop_activation_pct:.1f}%",
                                f"{order_aggregation_threshold:.1f}%"
                            ]
                        })
                        
                        sim_params_df = pd.DataFrame({
                            'Parameter': [
                                'Number of Simulations',
                                'Simulation Method'
                            ],
                            'Value': [
                                f"{num_simulations}",
                                f"{simulation_method}"
                            ]
                        })
                        
                        st.subheader("Core Parameters")
                        st.table(core_params_df)
                        
                        st.subheader("Advanced Risk Management Parameters")
                        st.table(advanced_params_df)
                        
                        st.subheader("Simulation Settings")
                        st.table(sim_params_df)
                        
                        # Create empty params_df for consistency with rest of code
                        params_df = core_params_df
                        
                        # Skip showing table again since we already displayed tables
                        show_table = False
                    else:
                        params_df = pd.DataFrame({
                            'Parameter': [
                                'Strategy Type',
                                'Initial Capital',
                                'Investment Amount',
                                'Price Drop Threshold',
                                'Profit Threshold',
                                'Trading Fee',
                                'Number of Simulations',
                                'Simulation Method'
                            ],
                            'Value': [
                                strategy_type,
                                f"${initial_capital:.2f}",
                                f"{investment_pct:.1f}%",
                                f"{price_drop_threshold:.1f}%",
                                f"{profit_threshold:.1f}%",
                                f"{trading_fee_pct:.2f}%",
                                f"{num_simulations}",
                                f"{simulation_method}"
                            ]
                        })
                        
                        # Show the table for other strategies
                        show_table = True
                    
                    # Only show table if not already shown (for OptimizedAlternativeDCA)
                    if show_table:
                        st.table(params_df)
                    
                    # Confidence intervals
                    st.markdown("### Return Confidence Intervals")
                    
                    ci_df = pd.DataFrame({
                        'Confidence Level': ['50%', '80%', '90%', '95%', '99%'],
                        'Lower Bound': [
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 25):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 10):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 5):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 2.5):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 0.5):.2f}%"
                        ],
                        'Upper Bound': [
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 75):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 90):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 95):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 97.5):.2f}%",
                            f"{np.percentile([r['return_pct'] for r in simulation_results], 99.5):.2f}%"
                        ]
                    })
                    
                    st.table(ci_df)
                
                # Tab 2: Return Distribution
                with tab2:
                    st.markdown("### Return Distribution Analysis")
                    
                    # Return histogram
                    returns = [r['return_pct'] for r in simulation_results]
                    return_fig = PerformancePlots.plot_return_distribution(
                        returns,
                        "Monte Carlo Return Distribution"
                    )
                    
                    st.plotly_chart(return_fig, use_container_width=True, key="monte_carlo_return_distribution")
                    
                    # Return statistics
                    st.markdown("### Return Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Central Tendency")
                        
                        central_df = pd.DataFrame({
                            'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Skewness', 'Kurtosis'],
                            'Value': [
                                f"{summary['return_pct']['mean']:.2f}%",
                                f"{summary['return_pct']['median']:.2f}%",
                                f"{summary['return_pct']['std']:.2f}%",
                                f"{report['return_distribution']['skewness']:.2f}",
                                f"{report['return_distribution']['kurtosis']:.2f}"
                            ]
                        })
                        
                        st.table(central_df)
                    
                    with col2:
                        st.markdown("#### Percentiles")
                        
                        percentiles_df = pd.DataFrame({
                            'Percentile': ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%'],
                            'Value': [
                                f"{report['return_distribution']['percentiles']['1%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['5%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['10%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['25%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['50%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['75%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['90%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['95%']:.2f}%",
                                f"{report['return_distribution']['percentiles']['99%']:.2f}%"
                            ]
                        })
                        
                        st.table(percentiles_df)
                    
                    # Best and worst case scenarios
                    st.markdown("### Best and Worst Case Scenarios")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Best Case")
                        
                        best_case = report['best_case']
                        best_case_index = best_case['simulation_index']
                        best_case_result = simulation_results[best_case_index]
                        
                        st.metric("Return", f"{best_case['return_pct']:.2f}%")
                        st.metric("Max Drawdown", f"{best_case_result['max_drawdown_pct']:.2f}%")
                        st.metric("Trades", f"{best_case_result['trades_count']}")
                        
                        # Add best case equity curve
                        if 'equity_curve' in best_case_result:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=best_case_result['equity_curve'],
                                mode='lines',
                                name='Equity',
                                line=dict(color='green', width=2)
                            ))
                            
                            fig.update_layout(
                                title='Best Case Equity Curve',
                                yaxis_title='Equity (USDT)',
                                template='plotly_white',
                                height=300,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="best_case_equity_curve")
                    
                    with col2:
                        st.markdown("#### Worst Case")
                        
                        worst_case = report['worst_case']
                        worst_case_index = worst_case['simulation_index']
                        worst_case_result = simulation_results[worst_case_index]
                        
                        st.metric("Return", f"{worst_case['return_pct']:.2f}%")
                        st.metric("Max Drawdown", f"{worst_case_result['max_drawdown_pct']:.2f}%")
                        st.metric("Trades", f"{worst_case_result['trades_count']}")
                        
                        # Add worst case equity curve
                        if 'equity_curve' in worst_case_result:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=worst_case_result['equity_curve'],
                                mode='lines',
                                name='Equity',
                                line=dict(color='red', width=2)
                            ))
                            
                            fig.update_layout(
                                title='Worst Case Equity Curve',
                                yaxis_title='Equity (USDT)',
                                template='plotly_white',
                                height=300,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="worst_case_equity_curve")
                
                # Tab 3: Risk Analysis
                with tab3:
                    st.markdown("### Risk Metrics")
                    
                    # Drawdown distribution
                    if simulation_results and len(simulation_results) > 0:
                        drawdowns = [r['max_drawdown_pct'] for r in simulation_results if 'max_drawdown_pct' in r]
                        if drawdowns and len(drawdowns) > 0:
                            drawdown_fig = PerformancePlots.plot_return_distribution(
                                drawdowns,
                                "Monte Carlo Drawdown Distribution"
                            )
                            
                            if drawdown_fig is not None:
                                st.plotly_chart(drawdown_fig, use_container_width=True, key="monte_carlo_drawdown_distribution")
                        else:
                            st.warning("No drawdown data available to display")
                    else:
                        st.warning("No simulation results available")
                    
                    # Value at Risk (VaR) and Conditional Value at Risk (CVaR)
                    st.markdown("### Value at Risk (VaR)")
                    
                    col1, col2 = st.columns(2)
                    
                    # Check if report and summary data exist and have the required structure
                    if (report and 'risk_metrics' in report and 
                        'value_at_risk' in report['risk_metrics'] and 
                        'conditional_var' in report['risk_metrics']):
                        
                        with col1:
                            st.markdown("#### Return Value at Risk")
                            
                            var_df = pd.DataFrame({
                                'Confidence Level': ['95%', '99%'],
                                'VaR': [
                                    f"{-report['risk_metrics']['value_at_risk'].get('95%_confidence', 0):.2f}%",
                                    f"{-report['risk_metrics']['value_at_risk'].get('99%_confidence', 0):.2f}%"
                                ],
                                'CVaR (Expected Shortfall)': [
                                    f"{-report['risk_metrics']['conditional_var'].get('95%_confidence', 0):.2f}%",
                                    f"{-report['risk_metrics']['conditional_var'].get('99%_confidence', 0):.2f}%"
                                ]
                            })
                            
                            st.table(var_df)
                    else:
                        with col1:
                            st.warning("Risk metrics data is not available")
                    
                    if (summary and 'max_drawdown_pct' in summary and 
                        isinstance(summary['max_drawdown_pct'], dict) and 
                        'percentiles' in summary['max_drawdown_pct']):
                        
                        with col2:
                            st.markdown("#### Drawdown Risk")
                            
                            dd_df = pd.DataFrame({
                                'Statistic': ['Mean Drawdown', 'Median Drawdown', 'Worst Drawdown', '95% VaR Drawdown'],
                                'Value': [
                                    f"{summary['max_drawdown_pct'].get('mean', 0):.2f}%",
                                    f"{summary['max_drawdown_pct'].get('median', 0):.2f}%",
                                    f"{summary['max_drawdown_pct'].get('min', 0):.2f}%",
                                    f"{summary['max_drawdown_pct']['percentiles'].get('95%', 0):.2f}%"
                                ]
                            })
                            
                            st.table(dd_df)
                    else:
                        with col2:
                            st.warning("Drawdown risk data is not available")
                    
                    # Risk metrics
                    st.markdown("### Risk-Adjusted Return Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if summary and 'sharpe_ratio' in summary and isinstance(summary['sharpe_ratio'], dict) and 'mean' in summary['sharpe_ratio']:
                        with col1:
                            st.metric(
                                "Mean Sharpe Ratio",
                                f"{summary['sharpe_ratio']['mean']:.2f}"
                            )
                    else:
                        with col1:
                            st.metric("Mean Sharpe Ratio", "N/A")
                    
                    if summary and 'trades_count' in summary and isinstance(summary['trades_count'], dict) and 'mean' in summary['trades_count'] and 'std' in summary['trades_count']:
                        with col2:
                            st.metric(
                                "Trade Consistency",
                                f"{summary['trades_count']['mean']:.1f} ± {summary['trades_count']['std']:.1f}"
                            )
                    else:
                        with col2:
                            st.metric("Trade Consistency", "N/A")
                    
                    if summary and 'win_rate' in summary and isinstance(summary['win_rate'], dict) and 'mean' in summary['win_rate'] and 'std' in summary['win_rate']:
                        with col3:
                            st.metric(
                                "Win Rate",
                                f"{summary['win_rate']['mean']:.1f}% ± {summary['win_rate']['std']:.1f}%"
                            )
                    else:
                        with col3:
                            st.metric("Win Rate", "N/A")
                    
                    # Advanced risk analysis
                    st.markdown("### Probability Analysis")
                    
                    # Calculate probabilities
                    if simulation_results and len(simulation_results) > 0:
                        returns = np.array([r['return_pct'] for r in simulation_results if 'return_pct' in r])
                        
                        if len(returns) > 0:
                            prob_profit = (returns > 0).mean() * 100
                            prob_beat_5pct = (returns > 5).mean() * 100
                            prob_beat_10pct = (returns > 10).mean() * 100
                            prob_beat_20pct = (returns > 20).mean() * 100
                            prob_loss_5pct = (returns < -5).mean() * 100
                            prob_loss_10pct = (returns < -10).mean() * 100
                        else:
                            # Default values if no valid returns found
                            prob_profit = prob_beat_5pct = prob_beat_10pct = prob_beat_20pct = prob_loss_5pct = prob_loss_10pct = 0.0
                    else:
                        # Default values if no simulation results
                        prob_profit = prob_beat_5pct = prob_beat_10pct = prob_beat_20pct = prob_loss_5pct = prob_loss_10pct = 0.0
                    
                    # Display probabilities
                    prob_df = pd.DataFrame({
                        'Scenario': [
                            'Probability of Profit',
                            'Probability of > 5% Return',
                            'Probability of > 10% Return',
                            'Probability of > 20% Return',
                            'Probability of > 5% Loss',
                            'Probability of > 10% Loss'
                        ],
                        'Probability': [
                            f"{prob_profit:.1f}%",
                            f"{prob_beat_5pct:.1f}%",
                            f"{prob_beat_10pct:.1f}%",
                            f"{prob_beat_20pct:.1f}%",
                            f"{prob_loss_5pct:.1f}%",
                            f"{prob_loss_10pct:.1f}%"
                        ]
                    })
                    
                    st.table(prob_df, key="probability_analysis_table")
                
                # Option to save simulation results
                st.markdown("### Save Simulation Results")
                
                save_name = st.text_input(
                    "Name for this simulation",
                    value=f"MonteCarlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    max_chars=50
                )
                
                if st.button("Save Results for Comparison"):
                    # Create a dictionary with essential results and parameters
                    if strategy_type == "Optimized Alternative DCA":
                        saved_result = {
                            'name': save_name,
                            'type': 'monte_carlo',
                            'timestamp': datetime.now(),
                            'parameters': {
                                'strategy_type': strategy_type,
                                'initial_capital': initial_capital,
                                'investment_pct': investment_pct,
                                'price_drop_threshold': price_drop_threshold,
                                'profit_threshold': profit_threshold,
                                'trading_fee_pct': trading_fee_pct,
                                # Advanced risk management parameters
                                'stop_loss_fixed_pct': stop_loss_fixed_pct,
                                'atr_stop_multiplier': atr_stop_multiplier,
                                'trailing_stop_pct': trailing_stop_pct,
                                'max_position_duration': max_position_duration, 
                                'trend_filter_ma_period': trend_filter_ma_period,
                                'volatility_atr_window': volatility_atr_window,
                                'min_order_size': min_order_size,
                                'order_aggregation_threshold': order_aggregation_threshold,
                                'trailing_stop_activation_pct': trailing_stop_activation_pct,
                                # Simulation parameters
                                'num_simulations': num_simulations,
                                'simulation_method': simulation_method
                            },
                            'data_file': selected_file,
                            'date_range': f"{filtered_df.index.min().date()} to {filtered_df.index.max().date()}",
                            'summary': summary,
                            'report': report
                        }
                    else:
                        saved_result = {
                            'name': save_name,
                            'type': 'monte_carlo',
                            'timestamp': datetime.now(),
                            'parameters': {
                                'strategy_type': strategy_type,
                                'initial_capital': initial_capital,
                                'investment_pct': investment_pct,
                                'price_drop_threshold': price_drop_threshold,
                                'profit_threshold': profit_threshold,
                                'trading_fee_pct': trading_fee_pct,
                                'num_simulations': num_simulations,
                                'simulation_method': simulation_method
                            },
                            'data_file': selected_file,
                            'date_range': f"{filtered_df.index.min().date()} to {filtered_df.index.max().date()}",
                            'summary': summary,
                            'report': report
                        }
                    
                    # Initialize the saved_results list if it doesn't exist
                    if 'saved_results' not in st.session_state:
                        st.session_state.saved_results = []
                    
                    # Add the result to the session state
                    st.session_state.saved_results.append(saved_result)
                    
                    st.success(f"Monte Carlo simulation '{save_name}' saved successfully! View it in the Results page.")
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.error(f"Error details: {error_details}")
