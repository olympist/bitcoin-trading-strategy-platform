import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import itertools
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import strategy modules
from strategy.optimized_alternative_dca_v5 import OptimizedAlternativeDCAv5
from backtesting.backtester import Backtester
from utils.data_fetcher import DataFetcher
from utils.results_manager import ResultsManager
from visualization.performance_plots import PerformancePlots

# Initialize components
data_fetcher = DataFetcher()
results_manager = ResultsManager()

def analyze_previous_results(flat_results, param_name):
    """
    Analyze previous optimization results to find optimal parameter ranges
    
    Args:
        flat_results: List of result dictionaries from previous batches
        param_name: Parameter name to analyze
        
    Returns:
        min_val, max_val, optimal_values - a tuple containing min/max and list of top values
    """
    try:
        # Sort results by return percentage
        sorted_results = sorted(flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        
        # Extract values for this parameter
        param_values = [res['parameters'][param_name] for res in sorted_results if param_name in res['parameters']]
        
        if not param_values:
            return None, None, []
        
        # Find min and max values
        min_val = min(param_values)
        max_val = max(param_values)
        
        # Get top 3 values (or fewer if not enough results)
        top_results = sorted_results[:min(3, len(sorted_results))]
        optimal_values = [res['parameters'][param_name] for res in top_results 
                          if param_name in res['parameters']]
        
        # Handle potential edge cases (all values are the same)
        if len(set(optimal_values)) == 1 and len(param_values) > 1:
            # Add some variation by including the second best set of parameters
            more_results = sorted_results[3:min(6, len(sorted_results))]
            more_values = [res['parameters'][param_name] for res in more_results 
                           if param_name in res['parameters']]
            optimal_values.extend(more_values)
        
        # Remove duplicates while preserving order
        seen = set()
        optimal_values = [x for x in optimal_values if not (x in seen or seen.add(x))]
        
        return min_val, max_val, optimal_values
    except Exception as e:
        st.warning(f"Error analyzing parameter {param_name}: {str(e)}")
        return None, None, []

# Page title and layout
st.set_page_config(
    page_title="Optimization V2.0",
    page_icon="📊",
    layout="wide"
)

def run_backtest(params, price_data, initial_capital=1000.0):
    """Run a single backtest with the specified parameters"""
    try:
        # Convert parameter values to appropriate types
        # This ensures proper type handling for the strategy initialization
        processed_params = {}
        for key, value in params.items():
            if key in ['shared_capital', 'multi_bot_mode', 'advanced_exits', 'partial_profit_taking', 'use_trailing_stop_main']:
                # Boolean parameters
                processed_params[key] = bool(value)
            elif key in ['number_of_bots', 'max_position_duration']:
                # Integer parameters
                processed_params[key] = int(value)
            else:
                # Float parameters
                processed_params[key] = float(value)
        
        # Use the backtester's run_backtest method which has proper error handling
        backtester = Backtester()
        backtester.set_data(price_data)
        
        # Create parameters dictionary with corrected types for the backtester
        strategy_params = {
            'initial_capital': initial_capital,
            'investment_pct': processed_params.get('investment_pct', 10.0),
            'price_drop_threshold': processed_params.get('price_drop_threshold', 2.0),
            'profit_threshold': processed_params.get('profit_threshold', 3.0),
            'trading_fee_pct': processed_params.get('trading_fee_pct', 0.1),
            'stop_loss_fixed_pct': processed_params.get('stop_loss_fixed_pct', 15.0),
            'trailing_stop_pct': processed_params.get('trailing_stop_pct', 1.5),
            'max_position_duration': processed_params.get('max_position_duration', 180),
            'number_of_bots': processed_params.get('number_of_bots', 1),
            'min_trade_size': processed_params.get('min_trade_size', 10.0),
            'partial_profit_taking': processed_params.get('partial_profit_taking', False),
            'profit_taking_percentage': processed_params.get('profit_taking_percentage', 50.0),
            'use_trailing_stop_main': processed_params.get('use_trailing_stop_main', True),
            'trailing_stop_activation_pct': processed_params.get('trailing_stop_activation_pct', 1.5)
        }
        
        # Run the backtest using the backtester's run_backtest method
        results = backtester.run_backtest("Optimized Alternative DCA v5", strategy_params)
        
        # Calculate additional metrics
        total_trades = len(results.get('trades', [])) if 'trades' in results else 0
        win_rate = results.get('win_rate', 0)
        max_drawdown = results.get('max_drawdown_pct', 0)
        profit_factor = results.get('profit_factor', 0)
        return_pct = results.get('return_pct', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        
        # Return the key metrics and full results
        return {
            'parameters': params,  # Keep original parameters for display
            'metrics': {
                'return_pct': return_pct,
                'max_drawdown_pct': max_drawdown,
                'win_rate': win_rate,
                'trades_count': total_trades,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio
            },
            'full_results': results
        }
    except Exception as e:
        st.error(f"Error during backtest: {str(e)}")
        return {
            'parameters': params,
            'metrics': {
                'return_pct': 0,
                'max_drawdown_pct': 0,
                'win_rate': 0,
                'trades_count': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0
            },
            'error': str(e)
        }

def optimize_strategy(parameter_ranges, fixed_parameters, price_data, initial_capital=1000.0, 
                     max_combinations=10000, batch_id=None):
    """
    Run an optimization over the parameter ranges provided.
    
    Args:
        parameter_ranges: Dict of parameter names to lists of values
        fixed_parameters: Dict of parameters that stay constant
        price_data: DataFrame with price data
        initial_capital: Starting capital for backtest
        max_combinations: Maximum number of parameter combinations to test
        (Parameter removed: verbose)
        batch_id: Optional batch identifier
        
    Returns:
        List of results dictionaries
    """
    # Combine parameter ranges into a list of parameter combinations
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    # Warn if there are too many combinations
    if total_combinations > max_combinations:
        st.warning(f"Total combinations ({total_combinations}) exceeds maximum ({max_combinations}). Only the first {max_combinations} will be tested.")
        total_combinations = max_combinations
    
    # Generate parameter combinations
    combinations = list(itertools.product(*param_values))
    if len(combinations) > max_combinations:
        combinations = combinations[:max_combinations]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run backtest for each combination
    results = []
    for i, combo in enumerate(combinations):
        # Create parameter dictionary for this combination
        params = {**fixed_parameters}  # Start with fixed parameters
        for j, param_name in enumerate(param_names):
            params[param_name] = combo[j]
        
        # Update progress
        progress = i / len(combinations)
        progress_bar.progress(progress)
        
        # Format status message
        param_text = ", ".join([f"{p}={v}" for p, v in zip(param_names, combo)])
        status_text.text(f"Testing combination {i+1}/{len(combinations)}: {param_text}")
        
        # Run backtest with these parameters (remove verbose parameter)
        result = run_backtest(params, price_data, initial_capital)
        results.append(result)
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text(f"Optimization completed! Tested {len(combinations)} parameter combinations.")
    
    # Sort results by return
    sorted_results = sorted(results, key=lambda x: x['metrics']['return_pct'], reverse=True)
    
    # Store batch information
    if batch_id:
        batch_info = {
            'batch_id': batch_id,
            'parameter_ranges': parameter_ranges,
            'fixed_parameters': fixed_parameters,
            'total_combinations': len(combinations),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        results_manager.save_optimization_results(
            sorted_results,
            'OptimizedAlternativeDCAv5',
            parameter_ranges,
            batch_info
        )
    
    return sorted_results

# This function is now defined at the top of the file, so we remove it here
# to avoid duplication.

def run_batch_optimization(batches, price_data, initial_capital=1000.0, max_combinations_per_batch=10000, 
                          adaptive_optimization=True):
    """
    Run multiple optimization batches sequentially, with adaptive parameter refinement.
    
    Args:
        batches: List of batch dictionaries with parameter_ranges and fixed_parameters
        price_data: DataFrame with price data
        initial_capital: Starting capital for backtest
        max_combinations_per_batch: Maximum parameter combinations per batch
        adaptive_optimization: Whether to adaptively refine parameters based on previous results
        
    Returns:
        Dictionary of batch results
    """
    all_results = {}
    all_batch_results = []  # Collect all individual test results
    
    progress_bar = st.progress(0)
    batch_status = st.empty()
    adaptive_info = st.empty()
    
    for i, batch in enumerate(batches):
        batch_id = batch.get('name', f"Batch {i+1}")
        batch_status.text(f"Running {batch_id} ({i+1}/{len(batches)})")
        
        # Adaptively refine parameters if enabled and we have previous results
        if adaptive_optimization and i > 0 and all_batch_results:
            original_optimize = batch['optimize'].copy()
            adaptive_updates = {}
            
            # Display adaptive optimization info
            adaptive_info.markdown("### Adaptive Parameter Refinement")
            adaptive_text = []
            
            # For each parameter in this batch, check if we have insights from previous batches
            for param_name in original_optimize.keys():
                # Find if this parameter was tested in previous batches
                for prev_batch_id, prev_results in all_results.items():
                    flat_results = []
                    for result_batch in prev_results:
                        if param_name in result_batch['parameters']:
                            flat_results.append(result_batch)
                    
                    if flat_results:
                        min_val, max_val, optimal_values = analyze_previous_results(flat_results, param_name)
                        
                        if min_val is not None and max_val is not None:
                            # Update parameter range based on previous findings
                            range_def = {
                                'min': min_val,
                                'max': max_val,
                                'steps': len(original_optimize[param_name])  # Keep same number of steps
                            }
                            
                            # Create new values focusing on the optimal range
                            if isinstance(original_optimize[param_name][0], int):
                                # Integer parameter
                                new_values = np.linspace(
                                    int(range_def['min']), 
                                    int(range_def['max']), 
                                    int(range_def['steps'])
                                ).astype(int).tolist()
                                
                                # Ensure we include the optimal values
                                for opt_val in optimal_values:
                                    if opt_val not in new_values and opt_val >= min_val and opt_val <= max_val:
                                        new_values.append(int(opt_val))
                                
                                new_values = sorted(list(set(new_values)))
                            else:
                                # Float parameter
                                new_values = np.linspace(
                                    range_def['min'], 
                                    range_def['max'], 
                                    int(range_def['steps'])
                                ).tolist()
                                
                                # Ensure we include the optimal values
                                for opt_val in optimal_values:
                                    if opt_val not in new_values and opt_val >= min_val and opt_val <= max_val:
                                        new_values.append(float(opt_val))
                                
                                # Round values appropriately
                                if param_name in ['investment_pct', 'price_drop_threshold', 'profit_threshold', 
                                                'stop_loss_fixed_pct', 'trailing_stop_pct', 'trailing_stop_activation_pct']:
                                    new_values = [round(v, 2) for v in sorted(list(set(new_values)))]
                                else:
                                    new_values = [round(v, 4) for v in sorted(list(set(new_values)))]
                            
                            # Update the batch with refined parameter values
                            batch['optimize'][param_name] = new_values
                            
                            # Record the adaptation for display
                            adaptive_updates[param_name] = {
                                'original': original_optimize[param_name],
                                'refined': new_values,
                                'optimal_values': optimal_values
                            }
                            
                            # Add to adaptive text
                            adaptive_text.append(f"**{param_name}**: Refined based on previous tests")
                            adaptive_text.append(f"  - Top performing values: {optimal_values}")
                            adaptive_text.append(f"  - Original range: {original_optimize[param_name]}")
                            adaptive_text.append(f"  - Refined range: {new_values}")
                            
                            # Only use the first found previous results for this parameter
                            break
            
            # Display adaptive updates
            if adaptive_updates:
                adaptive_info.markdown("\n".join(adaptive_text))
            else:
                adaptive_info.markdown("No parameter refinements needed for this batch.")
        
        # Run this batch
        batch_results = optimize_strategy(
            batch['optimize'],
            batch['fixed'],
            price_data,
            initial_capital,
            max_combinations_per_batch,
            batch_id=batch_id
        )
        
        all_results[batch_id] = batch_results
        # Add these results to the collective results for future refinement
        all_batch_results.extend(batch_results)
        
        progress_bar.progress((i + 1) / len(batches))
    
    batch_status.text(f"Completed all {len(batches)} batches!")
    
    # Final analysis
    if adaptive_optimization and all_batch_results:
        # Display overall best parameters found
        st.markdown("### Overall Best Parameters Found")
        
        # Sort all results by return percentage
        sorted_results = sorted(all_batch_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        
        # Display top 3 parameter combinations
        for i, result in enumerate(sorted_results[:3]):
            st.markdown(f"**#{i+1} Return: {result['metrics']['return_pct']:.2f}%**")
            params_text = []
            for param, value in result['parameters'].items():
                params_text.append(f"{param}: {value}")
            st.markdown(", ".join(params_text))
    
    return all_results

def main():
    st.title("V5 Strategy Optimization V2.0")
    st.markdown("""
    This tool allows you to optimize strategy parameters using grid search with batch processing.
    Define parameter combinations and run multiple optimization batches sequentially.
    """)
    
    # Sidebar: Data selection and general settings
    with st.sidebar:
        st.header("Data Source")
        # Data source selection (file list or date range)
        data_source = st.radio("Select data source", ["Use saved data file", "Fetch data for date range"])
        
        if data_source == "Use saved data file":
            # List available data files
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            if data_files:
                selected_file = st.selectbox("Select data file", data_files)
                filepath = os.path.join(data_dir, selected_file)
                
                if os.path.exists(filepath):
                    df = data_fetcher.load_data(filepath)
                    if df is not None:
                        st.success(f"Loaded {len(df)} rows of data")
                        # Show data summary
                        if not df.empty:
                            st.write(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
                            
                            # Calculate days of data
                            days = (df.index[-1] - df.index[0]).days
                            st.write(f"Total days: {days}")
            else:
                st.warning("No data files found. Upload CSV files to the 'data' directory.")
                df = None
        
        else:  # Date range option
            st.write("Fetch data for a specific date range")
            # Date selection
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            start_date = st.date_input("Start date", value=start_date)
            end_date = st.date_input("End date", value=end_date)
            
            symbol = st.text_input("Trading pair", value="BTC/USDT")
            exchange = st.selectbox("Exchange", ["binance", "bitstamp", "coinbase", "kraken"], index=0)
            timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
            
            if st.button("Fetch Data"):
                with st.spinner("Fetching data..."):
                    # Convert dates to datetime objects
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.min.time())
                    
                    # Fetch data
                    df = data_fetcher.fetch_historical_data(
                        exchange_id=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_datetime,
                        end_date=end_datetime
                    )
                    
                    if df is not None and not df.empty:
                        st.success(f"Fetched {len(df)} rows of data")
                        # Show data summary
                        st.write(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
                        
                        # Calculate days of data
                        days = (df.index[-1] - df.index[0]).days
                        st.write(f"Total days: {days}")
                        
                        # Option to save data
                        save_filename = f"{exchange}_{symbol.replace('/', '')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                        if st.button(f"Save as {save_filename}"):
                            save_path = os.path.join("data", save_filename)
                            df.to_csv(save_path)
                            st.success(f"Data saved to {save_path}")
                    else:
                        st.error("Failed to fetch data. Please check your inputs and try again.")
                        df = None
        
        # Optimization settings
        st.header("Optimization Settings")
        initial_capital = st.number_input("Initial Capital (USDT)", min_value=100.0, max_value=100000.0, value=1000.0, step=500.0)
        max_combinations = st.number_input("Max Combinations Per Batch", min_value=10, max_value=20000, value=10000)
        
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Standard Optimization", "Batch Optimization", "Results Comparison"])
    
    with tab1:
        st.header("Standard Parameter Optimization")
        st.markdown("""
        Define parameter ranges to test in a single optimization run.
        Each parameter can have multiple values to test. The system will test all combinations.
        """)
        
        # Default parameters for V5 strategy
        default_params = {
            'investment_pct': 10.0,
            'price_drop_threshold': 2.0,
            'profit_threshold': 3.0,
            'trading_fee_pct': 0.1,
            'stop_loss_fixed_pct': 15.0,
            'trailing_stop_pct': 1.5,
            'max_position_duration': 180,
            'min_trade_size': 10.0,
            'partial_profit_taking': False,
            'profit_taking_percentage': 50.0,
            'use_trailing_stop_main': True,
            'trailing_stop_activation_pct': 1.5,
            'number_of_bots': 1
        }
        
        # Parameter selection
        st.subheader("Select Parameters to Optimize")
        params_to_optimize = st.multiselect(
            "Parameters",
            list(default_params.keys()),
            default=['investment_pct', 'price_drop_threshold', 'profit_threshold']
        )
        
        # Parameter ranges
        st.subheader("Parameter Ranges")
        param_ranges = {}
        
        # Columns for parameter inputs
        col1, col2, col3 = st.columns(3)
        
        # Display parameters in columns
        for i, param in enumerate(params_to_optimize):
            col = [col1, col2, col3][i % 3]
            
            with col:
                st.markdown(f"**{param}**")
                
                if param in ['partial_profit_taking', 'use_trailing_stop_main']:
                    # Boolean parameters
                    values = st.multiselect(
                        f"{param} values",
                        [True, False],
                        default=[True, False]
                    )
                    
                elif param == 'number_of_bots':
                    # Integer parameter with slider
                    min_val = st.number_input(f"Min {param}", min_value=1, max_value=10, value=1)
                    max_val = st.number_input(f"Max {param}", min_value=1, max_value=10, value=5)
                    steps = st.number_input(f"Steps for {param}", min_value=1, max_value=10, value=5)
                    
                    if min_val <= max_val:
                        values = list(range(min_val, max_val + 1, max(1, (max_val - min_val) // min(steps, max_val - min_val + 1))))
                    else:
                        values = [min_val]
                    
                    # Ensure integer values
                    values = [int(v) for v in values]
                    
                    st.write(f"Values: {values}")
                    
                elif param == 'timeframes':
                    # List parameter
                    st.info("Timeframes configuration not available in this UI")
                    continue
                    
                else:
                    # Numeric parameters with min/max/step
                    min_val = st.number_input(f"Min {param}", min_value=0.0, max_value=1000.0, value=float(default_params[param]) / 2)
                    max_val = st.number_input(f"Max {param}", min_value=0.0, max_value=1000.0, value=float(default_params[param]) * 2)
                    steps = st.number_input(f"Steps for {param}", min_value=2, max_value=10, value=5)
                    
                    if min_val < max_val:
                        # Create float values with proper rounding
                        values = [float(round(v, 4)) for v in np.linspace(min_val, max_val, int(steps))]
                    else:
                        values = [float(min_val)]
                    
                    st.write(f"Values: {values}")
                
                param_ranges[param] = values
        
        # Fixed parameters
        st.subheader("Fixed Parameters")
        fixed_params = {}
        
        for param, default_value in default_params.items():
            if param not in params_to_optimize:
                if param in ['shared_capital', 'multi_bot_mode', 'advanced_exits']:
                    # Boolean parameters
                    fixed_params[param] = st.checkbox(f"{param}", value=default_value)
                elif param == 'timeframes':
                    # Skip timeframes config
                    continue
                else:
                    # Numeric parameters
                    fixed_params[param] = st.number_input(f"{param}", value=float(default_value))
        
        # Total combinations calculation
        total_combinations = 1
        for param, values in param_ranges.items():
            total_combinations *= len(values)
        
        st.info(f"Total parameter combinations: {total_combinations}")
        if total_combinations > max_combinations:
            st.warning(f"Exceeds maximum combinations ({max_combinations}). Only the first {max_combinations} will be tested.")
        
        # Run optimization button
        if st.button("Run Standard Optimization"):
            if 'df' in locals() and df is not None and not df.empty:
                with st.spinner("Running optimization..."):
                    # Run the optimization
                    optimization_results = optimize_strategy(
                        param_ranges,
                        fixed_params,
                        df,
                        initial_capital,
                        max_combinations
                    )
                    
                    # Store in session state
                    st.session_state.optimization_results = optimization_results
                    st.session_state.optimization_params = {
                        'parameter_ranges': param_ranges,
                        'fixed_parameters': fixed_params
                    }
                    
                    # Save to results manager
                    results_manager.save_optimization_results(
                        optimization_results, 
                        'OptimizedAlternativeDCAv5', 
                        param_ranges,
                        fixed_params
                    )
                    
                    st.success("Optimization completed!")
                    
                    # Display top results
                    st.subheader("Top 10 Parameter Combinations")
                    top_results = optimization_results[:10]
                    
                    # Create DataFrame for display
                    result_rows = []
                    for i, res in enumerate(top_results):
                        row = {
                            'Rank': i+1,
                            'Return (%)': f"{res['metrics']['return_pct']:.2f}",
                            'Max DD (%)': f"{res['metrics']['max_drawdown_pct']:.2f}",
                            'Win Rate (%)': f"{res['metrics']['win_rate']:.2f}",
                            'Trades': res['metrics']['trades_count'],
                            'Sharpe': f"{res['metrics']['sharpe_ratio']:.2f}"
                        }
                        # Add parameters
                        for param, value in res['parameters'].items():
                            if param in param_ranges:
                                row[param] = value
                        
                        result_rows.append(row)
                    
                    results_df = pd.DataFrame(result_rows)
                    st.dataframe(results_df)
                    
                    # Create parameter heatmaps if we have at least 2 optimized parameters
                    if len(param_ranges) >= 2:
                        st.subheader("Parameter Heatmaps")
                        st.markdown("Select two parameters to visualize their impact on performance:")
                        
                        # Parameter selection for heatmap
                        heatmap_cols = st.columns(2)
                        with heatmap_cols[0]:
                            x_param = st.selectbox("X-axis parameter", list(param_ranges.keys()))
                        with heatmap_cols[1]:
                            y_param = st.selectbox("Y-axis parameter", [p for p in param_ranges.keys() if p != x_param])
                        
                        # Metric selection for heatmap
                        metric = st.selectbox(
                            "Metric to display", 
                            ["return_pct", "max_drawdown_pct", "win_rate", "trades_count", "sharpe_ratio"],
                            format_func=lambda x: {
                                "return_pct": "Return (%)",
                                "max_drawdown_pct": "Max Drawdown (%)",
                                "win_rate": "Win Rate (%)",
                                "trades_count": "Number of Trades",
                                "sharpe_ratio": "Sharpe Ratio"
                            }[x]
                        )
                        
                        # Create the heatmap data
                        heatmap_data = []
                        for result in optimization_results:
                            x_val = result['parameters'][x_param]
                            y_val = result['parameters'][y_param]
                            metric_val = result['metrics'][metric]
                            
                            heatmap_data.append({
                                x_param: x_val,
                                y_param: y_val,
                                'metric': metric_val
                            })
                        
                        heatmap_df = pd.DataFrame(heatmap_data)
                        
                        # Create heatmap
                        pivot_df = heatmap_df.pivot_table(
                            values='metric',
                            index=y_param,
                            columns=x_param,
                            aggfunc='mean'
                        )
                        
                        # Determine color scale based on metric
                        if metric in ["max_drawdown_pct"]:
                            color_scale = "RdYlGn_r"  # Reversed for drawdown (lower is better)
                        else:
                            color_scale = "RdYlGn"  # Higher is better for other metrics
                        
                        fig = px.imshow(
                            pivot_df,
                            labels=dict(x=x_param, y=y_param, color=metric),
                            x=pivot_df.columns,
                            y=pivot_df.index,
                            color_continuous_scale=color_scale,
                            title=f"Impact of {x_param} and {y_param} on {metric}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please load or fetch data first before running optimization.")
    
    with tab2:
        st.header("Batch Optimization")
        st.markdown("""
        Run multiple optimization batches sequentially. Each batch will focus on different parameters.
        This allows testing more combinations by breaking them into manageable chunks.
        
        The Auto Mode automatically determines optimal batch configurations.
        """)
        
        # Option to use automatic or manual mode
        optimization_mode = st.radio("Optimization Mode", ["Auto Mode", "Manual Mode"], index=0)
        
        if optimization_mode == "Auto Mode":
            # Create batch configurations automatically
            auto_max_batch_size = st.number_input("Maximum Batch Size", min_value=10, max_value=1000, value=125)
            
            # Option for comprehensive initial scan or focused optimization
            auto_optimization_strategy = st.radio(
                "Optimization Strategy",
                ["Comprehensive Initial Scan", "Focused Parameter Optimization"],
                index=0,
                help="Comprehensive scan tests all parameters with wide ranges first. Focused optimization targets specific parameter combinations based on priority."
            )
            
            auto_batch_priority = st.selectbox("Optimization Priority", 
                ["Balanced", "Performance Focused", "Risk Management Focused", "Capital Efficiency Focused"], 
                index=0)
            
            # Define parameter importance based on selected priority
            parameter_priority = {
                "Balanced": {
                    "investment_pct": 90,
                    "price_drop_threshold": 85,
                    "profit_threshold": 80,
                    "stop_loss_fixed_pct": 75,
                    "trailing_stop_pct": 70,
                    "max_position_duration": 60,
                    "number_of_bots": 55,
                    "min_trade_size": 40,
                    "trailing_stop_activation_pct": 35,
                    "profit_taking_percentage": 30
                },
                "Performance Focused": {
                    "investment_pct": 95,
                    "profit_threshold": 90,
                    "number_of_bots": 85,
                    "price_drop_threshold": 80,
                    "trailing_stop_activation_pct": 75,
                    "trailing_stop_pct": 70,
                    "max_position_duration": 50,
                    "stop_loss_fixed_pct": 45,
                    "min_trade_size": 35,
                    "profit_taking_percentage": 30
                },
                "Risk Management Focused": {
                    "stop_loss_fixed_pct": 95,
                    "trailing_stop_pct": 90,
                    "max_position_duration": 85,
                    "trailing_stop_activation_pct": 80,
                    "profit_threshold": 75,
                    "price_drop_threshold": 70, 
                    "investment_pct": 65,
                    "min_trade_size": 50,
                    "number_of_bots": 45,
                    "profit_taking_percentage": 30
                },
                "Capital Efficiency Focused": {
                    "investment_pct": 95,
                    "min_trade_size": 90,
                    "number_of_bots": 85,
                    "price_drop_threshold": 80,
                    "profit_threshold": 75,
                    "max_position_duration": 70,
                    "trailing_stop_pct": 65,
                    "trailing_stop_activation_pct": 60,
                    "stop_loss_fixed_pct": 55,
                    "profit_taking_percentage": 50
                }
            }[auto_batch_priority]
            
            # Sort parameters by priority
            sorted_params = sorted(parameter_priority.items(), key=lambda x: x[1], reverse=True)
            
            # Define optimal parameter batches - pair highly important parameters
            auto_batches = []
            
            # Define batch setup based on selected optimization strategy
            batch_pairs = []
            
            if auto_optimization_strategy == "Comprehensive Initial Scan":
                # For comprehensive initial scan, we want to test ALL parameters
                # including edge values to identify boundaries
                
                # Get all parameters to test (excluding only boolean parameters)
                all_testable_params = list(default_params.keys())
                bool_params = ['partial_profit_taking', 'use_trailing_stop_main']
                numeric_params = [p for p in all_testable_params if p not in bool_params]
                
                st.markdown("### Comprehensive Edge-Value Scan Configuration")
                st.markdown("Testing all parameters with wide ranges and edge values for complete boundary assessment")
                
                # Create batches for all parameters - ensuring we test every parameter
                # and ensure edge values are tested first
                
                # First round: Test extreme parameter ranges in pairs
                # We'll create batches that combine pairs of parameters, testing min/max combinations
                for i in range(0, len(numeric_params), 2):
                    if i + 1 < len(numeric_params):
                        batch_pairs.append((numeric_params[i], numeric_params[i+1]))
                
                # Make sure we don't miss any parameters if we have an odd number
                if len(numeric_params) % 2 == 1:
                    # Pair the last parameter with the first one to ensure coverage
                    batch_pairs.append((numeric_params[-1], numeric_params[0]))
                
                # Now add cross-category combinations to test parameter interactions
                # (ensuring we haven't missed any parameter)
                
                # Group parameters into logical categories for cross-testing
                parameter_groups = {
                    "Investment Strategy": ['investment_pct', 'price_drop_threshold', 'profit_threshold'],
                    "Risk Management": ['stop_loss_fixed_pct', 'trailing_stop_pct', 'trailing_stop_activation_pct'],
                    "Position Management": ['max_position_duration', 'profit_taking_percentage'],
                    "Capital Allocation": ['number_of_bots', 'min_trade_size']
                }
                
                # Create cross-group pairs to test interactions between different parameter types
                for g1_name, g1_params in parameter_groups.items():
                    for g2_name, g2_params in parameter_groups.items():
                        if g1_name != g2_name:  # Only cross-group
                            # Take one parameter from each group for cross-testing
                            batch_pairs.append((g1_params[0], g2_params[0]))
                
                # Remove any duplicates that might have been created
                unique_batch_pairs = []
                for pair in batch_pairs:
                    if pair not in unique_batch_pairs and (pair[1], pair[0]) not in unique_batch_pairs:
                        unique_batch_pairs.append(pair)
                
                batch_pairs = unique_batch_pairs
                
            else:  # Focused Parameter Optimization
                # For focused optimization, prioritize parameters by importance
                top_params = [param for param, _ in sorted_params[:8]]
                
                if len(top_params) >= 2:
                    # Primary pairs (most important parameters paired together)
                    batch_pairs.append((top_params[0], top_params[1]))
                    
                    # Secondary pairs (cross-importance pairs for better coverage)
                    if len(top_params) >= 4:
                        batch_pairs.append((top_params[0], top_params[2]))
                        batch_pairs.append((top_params[1], top_params[3]))
                    
                    # Additional important parameters
                    if len(top_params) >= 6:
                        batch_pairs.append((top_params[2], top_params[4]))
                        batch_pairs.append((top_params[3], top_params[5]))
                    
                    # If we have more parameters, add one more batch
                    if len(top_params) >= 8:
                        batch_pairs.append((top_params[4], top_params[6]))
            
            # Parameter range definitions with FULL extreme ranges (0-100% where applicable)
            param_range_defs = {
                'investment_pct': {
                    'min': 0.0, 'max': 100.0, 'steps': 6,  # Full range from 0% to 100%
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]  # Include absolute boundaries
                },
                'price_drop_threshold': {
                    'min': 0.0, 'max': 20.0, 'steps': 6,  # Extended range with zero threshold
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 0.1, 0.5, 2.0, 5.0, 10.0, 20.0]
                },
                'profit_threshold': {
                    'min': 0.0, 'max': 20.0, 'steps': 6,  # Including zero profit target
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 0.5, 1.5, 3.0, 8.0, 15.0, 20.0]
                },
                'stop_loss_fixed_pct': {
                    'min': 0.0, 'max': 100.0, 'steps': 6,  # From no stop-loss to 100%
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 1.0, 5.0, 15.0, 30.0, 50.0, 100.0]
                },
                'trailing_stop_pct': {
                    'min': 0.0, 'max': 10.0, 'steps': 6,  # From no trailing stop to very loose
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 0.1, 0.5, 1.5, 3.0, 5.0, 10.0]
                },
                'max_position_duration': {
                    'min': 0, 'max': 1000, 'steps': 6,  # From no time limit to very long
                    'log_scale': True, 'precision': 0,
                    'edge_values': [0, 1, 30, 180, 360, 500, 1000]
                },
                'number_of_bots': {
                    'min': 1, 'max': 20, 'steps': 6,  # Testing even more bots
                    'log_scale': False, 'precision': 0,
                    'edge_values': [1, 2, 5, 8, 10, 15, 20]
                },
                'min_trade_size': {
                    'min': 0.0, 'max': 500.0, 'steps': 6,  # Full spectrum of order sizes
                    'log_scale': True, 'precision': 2,
                    'edge_values': [0.0, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
                },
                'trailing_stop_activation_pct': {
                    'min': 0.0, 'max': 20.0, 'steps': 6,  # Including no activation threshold
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 0.1, 0.5, 1.5, 3.0, 10.0, 20.0]
                },
                'profit_taking_percentage': {
                    'min': 0.0, 'max': 100.0, 'steps': 6,  # Full spectrum from 0% to 100%
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]
                },
                'trading_fee_pct': {
                    'min': 0.0, 'max': 1.0, 'steps': 5,  # Zero fees to extreme high fees
                    'log_scale': False, 'precision': 2,
                    'edge_values': [0.0, 0.1, 0.25, 0.5, 1.0]
                }
            }
            
            # Generate the batches based on parameter pairs
            batches = []
            
            # Create a dataset hash for deterministic batch names
            import hashlib
            dataset_name = st.session_state.get('dataset_name', 'default')
            dataset_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
            
            for i, (param1, param2) in enumerate(batch_pairs):
                batch_name = f"Batch_{dataset_hash}_iter{i+1}"
                
                # Generate parameter ranges using the definitions
                batch_ranges = {}
                
                for param in [param1, param2]:
                    range_def = param_range_defs[param]
                    
                    # Use edge values directly when available - testing extremes first
                    if 'edge_values' in range_def:
                        # Use the pre-defined edge values which already include min/max/important values
                        values = range_def['edge_values']
                    elif range_def['log_scale']:
                        if range_def['precision'] == 0:
                            # Integer logarithmic scale
                            values = np.logspace(
                                np.log10(range_def['min']), 
                                np.log10(range_def['max']), 
                                range_def['steps']
                            ).astype(int).tolist()
                        else:
                            # Float logarithmic scale
                            values = [round(float(v), range_def['precision']) for v in np.logspace(
                                np.log10(range_def['min']), 
                                np.log10(range_def['max']), 
                                range_def['steps']
                            )]
                    else:
                        if range_def['precision'] == 0:
                            # Integer linear scale
                            values = np.linspace(
                                range_def['min'], 
                                range_def['max'], 
                                range_def['steps']
                            ).astype(int).tolist()
                        else:
                            # Float linear scale
                            values = [round(float(v), range_def['precision']) for v in np.linspace(
                                range_def['min'], 
                                range_def['max'], 
                                range_def['steps']
                            )]
                    
                    # Ensure we always include the absolute min and max values for boundary testing
                    if values[0] != range_def['min']:
                        values.insert(0, range_def['min'])
                    if values[-1] != range_def['max']:
                        values.append(range_def['max'])
                    
                    # Sort and remove duplicates while preserving order
                    values = sorted(list(dict.fromkeys(values)))
                    
                    batch_ranges[param] = values
                
                # Fixed parameters (all parameters not in batch_ranges)
                batch_fixed = {}
                for param, default_value in default_params.items():
                    if param not in batch_ranges:
                        batch_fixed[param] = default_value
                
                # Calculate combinations
                batch_combinations = 1
                for values in batch_ranges.values():
                    batch_combinations *= len(values)
                
                # Add batch to list
                batches.append({
                    'name': batch_name,
                    'optimize': batch_ranges,
                    'fixed': batch_fixed,
                    'combinations': min(batch_combinations, auto_max_batch_size)
                })
            
            # Display batch information
            for i, batch in enumerate(batches):
                st.markdown(f"### Auto Batch {i+1}: {batch['name']}")
                st.markdown("**Parameters to Optimize:**")
                
                # Create columns for parameter display
                param_cols = st.columns(2)
                
                for j, (param, values) in enumerate(batch['optimize'].items()):
                    col = param_cols[j % 2]
                    with col:
                        st.markdown(f"**{param}**")
                        st.write(f"Values: {values}")
                
                st.info(f"Batch {i+1} combinations: {batch['combinations']}")
            
            # Total combinations across all batches
            total_batch_combinations = sum(batch['combinations'] for batch in batches)
            st.info(f"Total combinations across all auto batches: {total_batch_combinations}")
        
        else:  # Manual Mode
            # Number of batches
            num_batches = st.number_input("Number of Optimization Batches", min_value=1, max_value=10, value=2)
            
            # Create batch configurations
            batches = []
            
            for i in range(num_batches):
                st.markdown(f"### Batch {i+1} Configuration")
                
                # Create a dataset hash for deterministic batch names
                import hashlib
                dataset_name = st.session_state.get('dataset_name', 'default')
                dataset_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
                
                # Batch name
                batch_name = st.text_input(f"Batch {i+1} Name", value=f"Batch_{dataset_hash}_iter{i+1}")
                
                # Parameter selection for this batch
                batch_params = st.multiselect(
                    f"Parameters to optimize in Batch {i+1}",
                    list(default_params.keys()),
                    default=['investment_pct', 'price_drop_threshold'] if i == 0 else ['profit_threshold', 'stop_loss_fixed_pct']
                )
                
                # Parameter ranges
                st.markdown(f"**Parameter Ranges for Batch {i+1}**")
                batch_ranges = {}
                
                # Create grid for parameters
                param_cols = st.columns(3)
                
                for j, param in enumerate(batch_params):
                    col = param_cols[j % 3]
                    
                    with col:
                        st.markdown(f"**{param}**")
                        
                        if param in ['shared_capital', 'multi_bot_mode', 'advanced_exits']:
                            # Boolean parameters
                            values = st.multiselect(
                                f"{param} values",
                                [True, False],
                                default=[True, False]
                            )
                        
                        elif param == 'number_of_bots':
                            # Integer parameter with slider
                            min_val = st.number_input(f"Min {param} (Batch {i+1})", min_value=1, max_value=10, value=1, key=f"b{i}_{param}_min")
                            max_val = st.number_input(f"Max {param} (Batch {i+1})", min_value=1, max_value=10, value=5, key=f"b{i}_{param}_max")
                            steps = st.number_input(f"Steps for {param} (Batch {i+1})", min_value=1, max_value=10, value=5, key=f"b{i}_{param}_steps")
                            
                            if min_val <= max_val:
                                values = list(range(min_val, max_val + 1, max(1, (max_val - min_val) // min(steps, max_val - min_val + 1))))
                            else:
                                values = [min_val]
                            
                            # Ensure integer values
                            values = [int(v) for v in values]
                            
                            st.write(f"Values: {values}")
                        
                        elif param == 'timeframes':
                            # List parameter
                            st.info("Timeframes configuration not available in this UI")
                            continue
                        
                        else:
                            # Numeric parameters with min/max/step
                            min_val = st.number_input(f"Min {param} (Batch {i+1})", min_value=0.0, max_value=1000.0, value=float(default_params[param]) / 2, key=f"b{i}_{param}_min")
                            max_val = st.number_input(f"Max {param} (Batch {i+1})", min_value=0.0, max_value=1000.0, value=float(default_params[param]) * 2, key=f"b{i}_{param}_max")
                            steps = st.number_input(f"Steps for {param} (Batch {i+1})", min_value=2, max_value=10, value=5, key=f"b{i}_{param}_steps")
                            
                            if min_val < max_val:
                                # Create float values with proper rounding
                                values = [float(round(v, 4)) for v in np.linspace(min_val, max_val, int(steps))]
                            else:
                                values = [float(min_val)]
                            
                            st.write(f"Values: {values}")
                        
                        batch_ranges[param] = values
                
                # Fixed parameters for this batch
                st.markdown(f"**Fixed Parameters for Batch {i+1}**")
                batch_fixed = {}
                
                fixed_cols = st.columns(3)
                col_index = 0
                
                for param, default_value in default_params.items():
                    if param not in batch_params:
                        with fixed_cols[col_index % 3]:
                            if param in ['shared_capital', 'multi_bot_mode', 'advanced_exits']:
                                # Boolean parameters
                                batch_fixed[param] = st.checkbox(f"{param} (Batch {i+1})", value=default_value, key=f"b{i}_{param}_fixed")
                            elif param == 'timeframes':
                                # Skip timeframes config
                                continue
                            else:
                                # Numeric parameters
                                batch_fixed[param] = st.number_input(f"{param} (Batch {i+1})", value=float(default_value), key=f"b{i}_{param}_fixed")
                        col_index += 1
                
                # Calculate combinations for this batch
                batch_combinations = 1
                for values in batch_ranges.values():
                    batch_combinations *= len(values)
                
                st.info(f"Batch {i+1} combinations: {batch_combinations}")
                if batch_combinations > max_combinations:
                    st.warning(f"Exceeds maximum combinations per batch ({max_combinations}). Only the first {max_combinations} will be tested.")
                
                # Save batch configuration
                batches.append({
                    'name': batch_name,
                    'optimize': batch_ranges,
                    'fixed': batch_fixed,
                    'combinations': min(batch_combinations, max_combinations)
                })
            
            # Total combinations across all batches
            total_batch_combinations = sum(batch['combinations'] for batch in batches)
            st.info(f"Total combinations across all batches: {total_batch_combinations}")
        
        # Run batch optimization button
        # Add option for adaptive optimization based on previous batch results
        enable_adaptive_optimization = st.checkbox("Enable Adaptive Optimization (Refine Parameters Automatically)", value=True, 
                                                 help="Automatically analyzes results from earlier batches to refine parameters for later batches")
        
        # Check if we're continuing optimization from a previous run
        continuing_optimization = st.session_state.get('continuing_optimization', False)
        optimization_iteration = st.session_state.get('optimization_iteration', 0)
        
        if continuing_optimization and 'batch_results' in st.session_state:
            st.info(f"Continuing optimization - Iteration {optimization_iteration}")
            
            # Generate refined batches based on previous results
            refined_batches = []
            
            # Create a dataset hash for deterministic batch names
            import hashlib
            dataset_name = st.session_state.get('dataset_name', 'default')
            iteration = st.session_state.get('optimization_iteration', 0)
            dataset_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
            
            # Create new batches with refined parameter ranges based on previous top results
            for i, batch in enumerate(st.session_state.batch_configs):
                batch_id = f"RefinedBatch_{dataset_hash}_iter{iteration}_{i+1}"
                refined_optimize = {}
                
                for param_name, param_range in batch['optimize'].items():
                    # Find optimal values for this parameter from previous batches
                    flat_results = []
                    for prev_batch_id, prev_results in st.session_state.batch_results.items():
                        for result in prev_results:
                            if param_name in result['parameters']:
                                flat_results.append(result)
                    
                    if flat_results:
                        min_val, max_val, optimal_values = analyze_previous_results(flat_results, param_name)
                        
                        if min_val is not None and max_val is not None:
                            # Generate refined range around top values
                            if isinstance(optimal_values[0], int):
                                # Integer parameter - create focused range
                                new_values = []
                                for val in optimal_values:
                                    # Add values around the optimal ones
                                    new_values.extend([max(1, val-1), val, val+1])
                                # Add edge case if not already present
                                if min_val not in new_values:
                                    new_values.append(min_val)
                                if max_val not in new_values:
                                    new_values.append(max_val)
                                # Remove duplicates and sort
                                new_values = sorted(list(set(new_values)))
                                refined_optimize[param_name] = new_values
                            else:
                                # Float parameter - create focused range
                                new_values = []
                                for val in optimal_values:
                                    # Add values around the optimal ones with 5% variations
                                    variation = max(0.1, val * 0.05)  # At least 0.1 variation
                                    new_values.extend([max(0, val-variation), val, val+variation])
                                # Add edge cases if not already present
                                if min_val not in new_values:
                                    new_values.append(min_val)
                                if max_val not in new_values:
                                    new_values.append(max_val)
                                # Remove duplicates, round to 2 decimals and sort
                                new_values = sorted(list(set([round(val, 2) for val in new_values])))
                                refined_optimize[param_name] = new_values
                        else:
                            # Keep original range if analysis failed
                            refined_optimize[param_name] = param_range
                    else:
                        # Keep original range if no data
                        refined_optimize[param_name] = param_range
                
                # Calculate combinations
                refined_combinations = 1
                for param_values in refined_optimize.values():
                    refined_combinations *= len(param_values)
                
                refined_batches.append({
                    'name': batch_id,
                    'optimize': refined_optimize,
                    'fixed': batch['fixed'],
                    'combinations': min(refined_combinations, max_combinations)
                })
            
            # Use the refined batches
            batches = refined_batches
            
            # Reset continuing_optimization flag
            st.session_state.continuing_optimization = False
            
            # Display the refined batches
            for i, batch in enumerate(batches):
                st.subheader(f"Refined Batch {i+1}: {batch['name']}")
                st.write("Parameters to Optimize:")
                
                for param_name, param_values in batch['optimize'].items():
                    st.write(f"  - {param_name}")
                    st.write(f"    Values: {param_values}")
                
                st.write(f"Batch {i+1} combinations: {batch['combinations']}")
            
            # Total combinations across all refined batches
            total_refined_combinations = sum(batch['combinations'] for batch in batches)
            st.info(f"Total combinations across all refined batches: {total_refined_combinations}")
            
            # Automatically run the optimization with the refined batches
            if 'df' in locals() and df is not None and not df.empty:
                with st.spinner("Running refined batch optimization..."):
                    # Run batch optimization with the refined parameters
                    st.success("Automatically continuing with refined parameters...")
        
        if st.button("Run Batch Optimization") or continuing_optimization:
            if 'df' in locals() and df is not None and not df.empty:
                with st.spinner("Running batch optimization..."):
                    # Run batch optimization with adaptive optimization if enabled
                    batch_results = run_batch_optimization(
                        batches,
                        df,
                        initial_capital,
                        max_combinations,
                        adaptive_optimization=enable_adaptive_optimization
                    )
                    
                    # Store in session state
                    st.session_state.batch_results = batch_results
                    st.session_state.batch_configs = batches
                    
                    st.success(f"Completed {len(batches)} optimization batches!")
                    
                    # Add a "Next" button to continue with refined parameters
                    if enable_adaptive_optimization:
                        next_col1, next_col2 = st.columns([1, 3])
                        with next_col1:
                            if st.button("➡️ Continue Optimization", key="continue_optimization"):
                                # Generate new batches with refined parameters based on the results
                                st.session_state.continuing_optimization = True
                                st.session_state.optimization_iteration = st.session_state.get('optimization_iteration', 0) + 1
                                st.experimental_rerun()
                        with next_col2:
                            st.info("Click to run another round of optimization with automatically refined parameters based on previous results.")
                    
                    # Collect all results for improvement tracking
                    all_batch_top_results = []
                    all_batch_names = []
                    best_performance_metrics = {
                        'return_pct': 0,
                        'max_drawdown_pct': float('inf'),
                        'win_rate': 0,
                        'sharpe_ratio': 0,
                        'profit_factor': 0
                    }
                    
                    # First, display top results from each batch and collect data for improvement visualization
                    for batch_id, results in batch_results.items():
                        st.subheader(f"Top Results from {batch_id}")
                        
                        # Get top 5 from this batch
                        top_results = results[:5]
                        
                        # Find the best result from this batch
                        if results:
                            best_batch_result = results[0]  # Results already sorted
                            all_batch_top_results.append(best_batch_result)
                            all_batch_names.append(batch_id)
                            
                            # Update best performance metrics
                            for metric in best_performance_metrics:
                                if metric in best_batch_result['metrics']:
                                    if metric == 'max_drawdown_pct':
                                        best_performance_metrics[metric] = min(
                                            best_performance_metrics[metric], 
                                            best_batch_result['metrics'][metric]
                                        )
                                    else:
                                        best_performance_metrics[metric] = max(
                                            best_performance_metrics[metric], 
                                            best_batch_result['metrics'][metric]
                                        )
                        
                        # Create DataFrame for display
                        result_rows = []
                        for i, res in enumerate(top_results):
                            row = {
                                'Rank': i+1,
                                'Return (%)': f"{res['metrics']['return_pct']:.2f}",
                                'Max DD (%)': f"{res['metrics']['max_drawdown_pct']:.2f}",
                                'Win Rate (%)': f"{res['metrics']['win_rate']:.2f}",
                                'Trades': res['metrics']['trades_count'],
                                'Sharpe': f"{res['metrics']['sharpe_ratio']:.2f}"
                            }
                            
                            # Add optimized parameters
                            batch_config = next((b for b in batches if b['name'] == batch_id), None)
                            if batch_config:
                                for param in batch_config['optimize'].keys():
                                    row[param] = res['parameters'].get(param, 'N/A')
                            
                            result_rows.append(row)
                        
                        batch_df = pd.DataFrame(result_rows)
                        st.dataframe(batch_df)
                    
                    # Now create improvement charts to visualize the refinement process
                    st.subheader("Optimization Progress and Improvement")
                    
                    # Create improvement visualization for Return, Drawdown, Win Rate
                    if len(all_batch_top_results) > 1:
                        progress_metrics = {
                            'return_pct': 'Return (%)',
                            'max_drawdown_pct': 'Max Drawdown (%)',
                            'win_rate': 'Win Rate (%)',
                            'sharpe_ratio': 'Sharpe Ratio',
                            'trades_count': 'Number of Trades'
                        }
                        
                        # Create tabs for different metric visualizations
                        progress_tabs = st.tabs([v for v in progress_metrics.values()])
                        
                        for i, (metric_key, metric_name) in enumerate(progress_metrics.items()):
                            with progress_tabs[i]:
                                # Get data for this metric
                                batch_numbers = list(range(1, len(all_batch_top_results) + 1))
                                metric_values = [res['metrics'][metric_key] for res in all_batch_top_results]
                                
                                # Create progress chart
                                fig = go.Figure()
                                
                                # Special handling for drawdown (lower is better)
                                if metric_key == 'max_drawdown_pct':
                                    # Invert for visualization
                                    fig.add_trace(go.Scatter(
                                        x=batch_numbers,
                                        y=metric_values,
                                        mode='lines+markers',
                                        name=metric_name,
                                        line=dict(color='red', width=3),
                                        marker=dict(size=10)
                                    ))
                                    
                                    # Add annotation for improvement direction
                                    fig.add_annotation(
                                        x=batch_numbers[-1],
                                        y=metric_values[-1],
                                        text="Lower is better",
                                        showarrow=True,
                                        arrowhead=1
                                    )
                                else:
                                    fig.add_trace(go.Scatter(
                                        x=batch_numbers,
                                        y=metric_values,
                                        mode='lines+markers',
                                        name=metric_name,
                                        line=dict(color='green', width=3),
                                        marker=dict(size=10)
                                    ))
                                    
                                    # Add annotation for improvement direction
                                    fig.add_annotation(
                                        x=batch_numbers[-1],
                                        y=metric_values[-1],
                                        text="Higher is better",
                                        showarrow=True,
                                        arrowhead=1
                                    )
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"Improvement in {metric_name} Across Batches",
                                    xaxis_title="Batch Number",
                                    yaxis_title=metric_name,
                                    xaxis=dict(tickmode='linear'),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add a metrics summary for this metric
                                if metric_key != 'trades_count':
                                    best_value = max(metric_values) if metric_key != 'max_drawdown_pct' else min(metric_values)
                                    initial_value = metric_values[0]
                                    
                                    if metric_key == 'max_drawdown_pct':
                                        improvement = initial_value - best_value  # Lower is better
                                        improvement_pct = (improvement / initial_value) * 100 if initial_value != 0 else 0
                                        st.metric(
                                            label=f"Drawdown Improvement",
                                            value=f"{best_value:.2f}%",
                                            delta=f"-{improvement:.2f}% ({improvement_pct:.1f}% reduction)"
                                        )
                                    else:
                                        improvement = best_value - initial_value  # Higher is better
                                        improvement_pct = (improvement / initial_value) * 100 if initial_value != 0 else 0
                                        st.metric(
                                            label=f"{metric_name} Improvement",
                                            value=f"{best_value:.2f}" + ("%" if "%" in metric_name else ""),
                                            delta=f"+{improvement:.2f} ({improvement_pct:.1f}% increase)"
                                        )
                        
                        # Create parameter evolution visualization
                        st.subheader("Parameter Value Evolution")
                        
                        # Find all optimized parameters
                        all_optimized_params = set()
                        for batch in batches:
                            all_optimized_params.update(batch['optimize'].keys())
                        
                        # Select a parameter to visualize
                        param_to_visualize = st.selectbox(
                            "Select Parameter to Visualize Evolution",
                            list(all_optimized_params)
                        )
                        
                        # Create parameter evolution chart
                        param_values = []
                        for i, res in enumerate(all_batch_top_results):
                            batch_id = all_batch_names[i]
                            # Find if this parameter was optimized in this batch
                            batch_config = next((b for b in batches if b['name'] == batch_id), None)
                            if batch_config and param_to_visualize in batch_config['optimize']:
                                # Parameter was explicitly optimized
                                param_values.append({
                                    'batch': i+1,
                                    'value': res['parameters'].get(param_to_visualize, None),
                                    'optimized': True
                                })
                            elif param_to_visualize in res['parameters']:
                                # Parameter was fixed but included in results
                                param_values.append({
                                    'batch': i+1,
                                    'value': res['parameters'].get(param_to_visualize, None),
                                    'optimized': False
                                })
                        
                        if param_values:
                            param_fig = go.Figure()
                            
                            # Split into optimized and fixed points
                            optimized_points = [p for p in param_values if p['optimized']]
                            fixed_points = [p for p in param_values if not p['optimized']]
                            
                            # Add optimized points
                            if optimized_points:
                                param_fig.add_trace(go.Scatter(
                                    x=[p['batch'] for p in optimized_points],
                                    y=[p['value'] for p in optimized_points],
                                    mode='lines+markers',
                                    name=f"Optimized {param_to_visualize}",
                                    line=dict(color='blue', width=3),
                                    marker=dict(size=12, symbol='circle')
                                ))
                            
                            # Add fixed points
                            if fixed_points:
                                param_fig.add_trace(go.Scatter(
                                    x=[p['batch'] for p in fixed_points],
                                    y=[p['value'] for p in fixed_points],
                                    mode='markers',
                                    name=f"Fixed {param_to_visualize}",
                                    marker=dict(size=8, symbol='square', color='gray')
                                ))
                            
                            # Update layout
                            param_fig.update_layout(
                                title=f"Evolution of {param_to_visualize} Value Across Batches",
                                xaxis_title="Batch Number",
                                yaxis_title=param_to_visualize,
                                xaxis=dict(tickmode='linear'),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(param_fig, use_container_width=True)
                    
                    # Create an Improvement Summary card
                    st.subheader("Optimization Results Summary")
                    
                    # Create columns for metrics
                    metric_cols = st.columns(4)
                    
                    if len(all_batch_top_results) > 0:
                        # Calculate improvements from first to best
                        first_result = all_batch_top_results[0]
                        best_return = max(res['metrics']['return_pct'] for res in all_batch_top_results)
                        best_win_rate = max(res['metrics']['win_rate'] for res in all_batch_top_results)
                        best_drawdown = min(res['metrics']['max_drawdown_pct'] for res in all_batch_top_results)
                        best_sharpe = max(res['metrics']['sharpe_ratio'] for res in all_batch_top_results)
                        
                        # Display metrics with improvements
                        with metric_cols[0]:
                            return_improvement = best_return - first_result['metrics']['return_pct']
                            return_pct_change = (return_improvement / first_result['metrics']['return_pct']) * 100 if first_result['metrics']['return_pct'] != 0 else 0
                            st.metric(
                                label="Best Return (%)",
                                value=f"{best_return:.2f}%",
                                delta=f"+{return_improvement:.2f}% ({return_pct_change:.1f}%)" if return_improvement > 0 else "No improvement"
                            )
                        
                        with metric_cols[1]:
                            dd_improvement = first_result['metrics']['max_drawdown_pct'] - best_drawdown
                            dd_pct_change = (dd_improvement / first_result['metrics']['max_drawdown_pct']) * 100 if first_result['metrics']['max_drawdown_pct'] != 0 else 0
                            st.metric(
                                label="Best Max Drawdown (%)",
                                value=f"{best_drawdown:.2f}%",
                                delta=f"-{dd_improvement:.2f}% ({dd_pct_change:.1f}%)" if dd_improvement > 0 else "No improvement",
                                delta_color="inverse"  # Lower is better for drawdown
                            )
                        
                        with metric_cols[2]:
                            win_rate_improvement = best_win_rate - first_result['metrics']['win_rate']
                            win_rate_pct_change = (win_rate_improvement / first_result['metrics']['win_rate']) * 100 if first_result['metrics']['win_rate'] != 0 else 0
                            st.metric(
                                label="Best Win Rate (%)",
                                value=f"{best_win_rate:.2f}%",
                                delta=f"+{win_rate_improvement:.2f}% ({win_rate_pct_change:.1f}%)" if win_rate_improvement > 0 else "No improvement"
                            )
                        
                        with metric_cols[3]:
                            sharpe_improvement = best_sharpe - first_result['metrics']['sharpe_ratio']
                            sharpe_pct_change = (sharpe_improvement / first_result['metrics']['sharpe_ratio']) * 100 if first_result['metrics']['sharpe_ratio'] != 0 else 0
                            st.metric(
                                label="Best Sharpe Ratio",
                                value=f"{best_sharpe:.2f}",
                                delta=f"+{sharpe_improvement:.2f} ({sharpe_pct_change:.1f}%)" if sharpe_improvement > 0 else "No improvement"
                            )
                    
                    # Parameter Refinement Visualization
                    st.subheader("Parameter Refinement Path")
                    st.markdown("This visualization shows how parameter values were refined during the optimization process:")
                    
                    # Select parameters to display
                    if len(all_optimized_params) > 0:
                        selected_params = st.multiselect(
                            "Select parameters to visualize refinement path",
                            list(all_optimized_params),
                            default=list(all_optimized_params)[:3] if len(all_optimized_params) >= 3 else list(all_optimized_params)
                        )
                        
                        if selected_params:
                            # Create parallel coordinates plot for parameter refinement
                            param_data = []
                            
                            for i, res in enumerate(all_batch_top_results):
                                row = {'Batch': i+1, 'Return (%)': res['metrics']['return_pct']}
                                for param in selected_params:
                                    if param in res['parameters']:
                                        row[param] = res['parameters'][param]
                                param_data.append(row)
                            
                            if param_data:
                                param_df = pd.DataFrame(param_data)
                                
                                # Create parallel coordinates plot
                                dimensions = [dict(label='Batch', values=param_df['Batch'], range=[0.5, len(param_df) + 0.5])]
                                
                                # Add selected parameters
                                for param in selected_params:
                                    if param in param_df.columns:
                                        dimensions.append(dict(label=param, values=param_df[param]))
                                
                                # Add return as the final dimension
                                dimensions.append(dict(label='Return (%)', values=param_df['Return (%)'], range=[param_df['Return (%)'].min() * 0.9, param_df['Return (%)'].max() * 1.1]))
                                
                                # Create the figure
                                parallel_fig = go.Figure(data=
                                    go.Parcoords(
                                        line=dict(color=param_df['Return (%)'], colorscale='Viridis', showscale=True),
                                        dimensions=dimensions
                                    )
                                )
                                
                                # Update layout
                                parallel_fig.update_layout(
                                    title="Parameter Refinement Path with Performance",
                                    height=600
                                )
                                
                                st.plotly_chart(parallel_fig, use_container_width=True)
            else:
                st.error("Please load or fetch data first before running optimization.")
    
    with tab3:
        st.header("Optimization Results Comparison")
        st.markdown("""
        Compare results from different optimization runs to identify the best parameters.
        This view can combine results from standard optimization and batch optimization.
        """)
        
        # Check for available results
        has_standard_results = 'optimization_results' in st.session_state
        has_batch_results = 'batch_results' in st.session_state
        
        if has_standard_results or has_batch_results:
            # Results available for comparison
            st.success("Optimization results available for comparison")
            
            # Options for visualization
            st.subheader("Visualization Options")
            
            metric_options = {
                'return_pct': 'Return (%)',
                'max_drawdown_pct': 'Max Drawdown (%)',
                'win_rate': 'Win Rate (%)',
                'trades_count': 'Number of Trades',
                'sharpe_ratio': 'Sharpe Ratio',
                'profit_factor': 'Profit Factor'
            }
            
            primary_metric = st.selectbox(
                "Primary Metric", 
                list(metric_options.keys()),
                format_func=lambda x: metric_options[x]
            )
            
            secondary_metric = st.selectbox(
                "Secondary Metric",
                [m for m in metric_options.keys() if m != primary_metric],
                format_func=lambda x: metric_options[x]
            )
            
            # Combine results from different sources
            all_results = []
            result_sources = []
            
            if has_standard_results and st.session_state.optimization_results is not None:
                all_results.extend(st.session_state.optimization_results)
                result_sources.extend(['Standard'] * len(st.session_state.optimization_results))
            
            if has_batch_results and st.session_state.batch_results is not None:
                for batch_id, results in st.session_state.batch_results.items():
                    if results:  # Make sure results is not empty
                        all_results.extend(results)
                        result_sources.extend([batch_id] * len(results))
            
            # Create scatter plot of results
            fig = px.scatter(
                x=[r['metrics'][primary_metric] for r in all_results],
                y=[r['metrics'][secondary_metric] for r in all_results],
                color=result_sources,
                labels={'x': metric_options[primary_metric], 'y': metric_options[secondary_metric], 'color': 'Source'},
                title=f"{metric_options[primary_metric]} vs {metric_options[secondary_metric]}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find global best parameters
            st.subheader("Global Best Parameters")
            
            # Sort by primary metric
            sorted_results = sorted(
                all_results, 
                key=lambda x: x['metrics'][primary_metric],
                reverse=primary_metric not in ['max_drawdown_pct']  # Reverse sort for metrics where higher is better
            )
            
            # Display top 10 global results
            top_global = sorted_results[:10]
            
            # Create DataFrame for display
            global_rows = []
            for i, res in enumerate(top_global):
                # Find source - use safer approach than list.index() to avoid DataFrame issues
                source = "Unknown"
                for j, result in enumerate(all_results):
                    if result is res or (isinstance(result, dict) and isinstance(res, dict) and 
                                        result.get('parameters') == res.get('parameters')):
                        source = result_sources[j]
                        break
                
                row = {
                    'Rank': i+1,
                    'Source': source,
                    'Return (%)': f"{res['metrics']['return_pct']:.2f}",
                    'Max DD (%)': f"{res['metrics']['max_drawdown_pct']:.2f}",
                    'Win Rate (%)': f"{res['metrics']['win_rate']:.2f}",
                    'Trades': res['metrics']['trades_count'],
                    'Sharpe': f"{res['metrics']['sharpe_ratio']:.2f}"
                }
                
                # Add key parameters
                key_params = ['investment_pct', 'price_drop_threshold', 'profit_threshold', 'stop_loss_fixed_pct']
                for param in key_params:
                    if param in res['parameters']:
                        row[param] = res['parameters'][param]
                
                global_rows.append(row)
            
            global_df = pd.DataFrame(global_rows)
            st.dataframe(global_df)
            
            # Parameter importance analysis
            st.subheader("Parameter Importance Analysis")
            
            # Get all parameters that were varied
            all_params = set()
            for res in all_results:
                all_params.update(res['parameters'].keys())
            
            # Select parameter to analyze
            param_to_analyze = st.selectbox("Select parameter to analyze", list(all_params))
            
            # Group results by parameter value
            param_values = {}
            for res in all_results:
                if param_to_analyze in res['parameters']:
                    value = res['parameters'][param_to_analyze]
                    if value not in param_values:
                        param_values[value] = []
                    param_values[value].append(res)
            
            # Calculate average performance for each parameter value
            param_performance = []
            for value, results in param_values.items():
                avg_metrics = {}
                for metric in ['return_pct', 'max_drawdown_pct', 'win_rate', 'sharpe_ratio']:
                    vals = [r['metrics'][metric] for r in results]
                    avg_metrics[metric] = sum(vals) / len(vals)
                
                param_performance.append({
                    'value': value,
                    'count': len(results),
                    **avg_metrics
                })
            
            # Sort by parameter value for the chart
            param_performance.sort(key=lambda x: x['value'])
            
            # Create chart
            param_fig = go.Figure()
            
            # Add bars for return
            param_fig.add_trace(go.Bar(
                x=[str(p['value']) for p in param_performance],
                y=[p['return_pct'] for p in param_performance],
                name='Return (%)',
                marker_color='green'
            ))
            
            # Add line for sharpe ratio on secondary axis
            param_fig.add_trace(go.Scatter(
                x=[str(p['value']) for p in param_performance],
                y=[p['sharpe_ratio'] for p in param_performance],
                name='Sharpe Ratio',
                yaxis='y2',
                mode='lines+markers',
                marker_color='blue'
            ))
            
            # Update layout with secondary y-axis
            param_fig.update_layout(
                title=f"Impact of {param_to_analyze} on Performance",
                xaxis_title=param_to_analyze,
                yaxis_title='Return (%)',
                yaxis2=dict(
                    title='Sharpe Ratio',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(param_fig, use_container_width=True)
            
            # Create table of performance by parameter value
            param_df = pd.DataFrame(param_performance)
            param_df['return_pct'] = param_df['return_pct'].round(2)
            param_df['max_drawdown_pct'] = param_df['max_drawdown_pct'].round(2)
            param_df['win_rate'] = param_df['win_rate'].round(2)
            param_df['sharpe_ratio'] = param_df['sharpe_ratio'].round(2)
            
            st.dataframe(param_df)
            
            # Export optimal parameters
            st.subheader("Export Optimal Parameters")
            
            if st.button("Export Best Parameters"):
                # Get best result based on primary metric
                best_result = sorted_results[0]
                
                # Create config file content
                config = {
                    'strategy': 'OptimizedAlternativeDCAv5',
                    'parameters': best_result['parameters'],
                    'performance': best_result['metrics'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'notes': f"Best parameters based on {metric_options[primary_metric]}"
                }
                
                # Save to file
                config_dir = "configs"
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                
                filename = f"v5_optimal_params_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                filepath = os.path.join(config_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                
                st.success(f"Best parameters exported to {filepath}")
                
                # Display the parameters
                st.json(config)
        else:
            st.warning("No optimization results available. Run optimization first.")

if __name__ == "__main__":
    main()