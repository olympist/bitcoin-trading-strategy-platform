import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

from backtesting.backtester import Backtester
from strategy.variable_timeframe_dca import VariableTimeframeDCA
from strategy.alternative_dca import AlternativeDCA
from strategy.fixed_investment_dca import FixedInvestmentDCA
from strategy.liquidity_managed_dca import LiquidityManagedDCA
from strategy.liquidity_managed_dca_new import LiquidityManagedDCA as LiquidityManagedDCANew
from utils.data_fetcher import DataFetcher
from visualization import PerformancePlots

def update_progress(percent, message):
    """Update progress bar with message."""
    if 'progress_bar' in st.session_state:
        st.session_state.progress_bar.progress(percent / 100)
        st.session_state.status_text.text(f"Status: {message} ({percent}%)")

def run_strategy_comparison():
    st.title("Strategy Comparison: DCA Strategies")
    
    st.write("""
    This page allows you to compare five variations of the Dollar Cost Averaging (DCA) strategy:
    
    **Original DCA Strategy:**
    - Tracks the highest observed price as reference
    - Buys when price drops by X% from reference high
    - Does NOT reset reference high after buying
    - Resets reference high only after selling
    - Can make multiple buys during extended downtrends
    - Invests a percentage of remaining capital
    
    **Alternative DCA Strategy:**
    - Tracks the highest observed price as reference
    - Buys when price drops by X% from reference high
    - DOES reset reference high to the purchase price after each buy
    - Requires a new X% drop from the new reference high for subsequent buys
    - Resets reference high after selling
    - Invests a percentage of remaining capital
    
    **Fixed Investment DCA Strategy:**
    - Tracks the highest observed price as reference
    - Buys when price drops by X% from reference high
    - Resets reference price to purchase price after each buy
    - Invests a fixed percentage of initial capital each time
    - May execute more consistent investment amounts
    
    **Liquidity-Managed DCA Strategy:**
    - Dynamically adjusts investment amount based on price movement patterns
    - Buys more aggressively during significant price drops
    - Preserves capital during uptrends with smaller investments
    - Investment amounts calculated as percentages of remaining capital
    - Takes profit at specified margin
    - Balances liquidity and market participation
    
    **Enhanced Liquidity-Managed DCA Strategy:**
    - Features a two-tier capital system with initial and reserve funds
    - Dynamically adjusts investment amount based on price movement
    - Investment amounts calculated as percentages of remaining capital
    - Accesses reserve capital during significant market opportunities
    - Implements intelligent capital management for extended market cycles
    - Takes profit at specified margin and replenishes reserves
    - Optimized for longer investment horizons
    """)
    
    # Sidebar for data selection and strategy parameters
    st.sidebar.header("Data Selection")
    
    # Data source options
    data_source = st.sidebar.radio(
        "Data Source",
        ["Standard Data Files", "Attached Test Data"]
    )
    
    data_fetcher = DataFetcher()
    
    if data_source == "Standard Data Files":
        # List available data files
        available_files = data_fetcher.list_saved_data()
        
        if not available_files:
            st.sidebar.warning("No data files found. Please go to the Data Manager page to download data.")
            return
        
        # Sort files by created date (newest first) and timeframe
        available_files.sort(key=lambda x: (x['created'], x['timeframe']), reverse=True)
        
        # Format file options for selectbox
        file_options = [f"{f['exchange']} | {f['symbol']} | {f['timeframe']} | {f['created']}" for f in available_files]
        selected_file_idx = st.sidebar.selectbox("Select Data File", range(len(file_options)), 
                                              format_func=lambda x: file_options[x])
        selected_file = available_files[selected_file_idx]
        selected_filepath = selected_file['filepath']
        
        # Load the selected data
        data = data_fetcher.load_data(selected_filepath)
        
    else:  # Attached Test Data
        # Use the provided test data file
        test_data_path = "./attached_assets/bitstamp_2025-03-08 15-31-50_BTC-USDT_15m.csv"
        
        if not os.path.exists(test_data_path):
            st.sidebar.error(f"Test data file not found at {test_data_path}")
            return
            
        st.sidebar.success("Using the provided test data file for backtesting")
        # Load the test data directly
        try:
            data = pd.read_csv(test_data_path)
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                
            # Create a mock selected_file for consistency with the UI flow
            selected_file = {
                'exchange': 'bitstamp',
                'symbol': 'BTC/USDT',
                'timeframe': '15m',
                'created': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'filepath': test_data_path
            }
        except Exception as e:
            st.sidebar.error(f"Error loading test data: {str(e)}")
            return
    
    if data is None:
        st.error(f"Failed to load data from {selected_file['filepath']}")
        return
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    
    # Convert index to datetime if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Get min and max dates from data
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    
    # Default to last 90 days if data range permits
    default_start = max(min_date, max_date - timedelta(days=90))
    
    start_date = st.sidebar.date_input("Start Date", default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=start_date, max_value=max_date)
    
    # Filter data by date
    mask = (data.index.date >= start_date) & (data.index.date <= end_date)
    filtered_data = data.loc[mask].copy()
    
    if len(filtered_data) == 0:
        st.error("No data available for the selected date range. Please select different dates.")
        return
    
    st.sidebar.subheader("Strategy Selection")
    
    # Strategy selection
    strategies_to_compare = st.sidebar.multiselect(
        "Select Strategies to Compare",
        ["Original DCA", "Alternative DCA", "Fixed Investment DCA", "Liquidity-Managed DCA", "Enhanced Liquidity-Managed DCA"],
        default=["Original DCA", "Alternative DCA"]
    )
    
    if not strategies_to_compare:
        st.sidebar.warning("Please select at least one strategy to run")
        return
        
    st.sidebar.subheader("Strategy Parameters")
    
    # Strategy parameters
    initial_capital = st.sidebar.number_input("Initial Capital (USDT)", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
    
    # Check if Liquidity-Managed DCA or Enhanced Liquidity-Managed DCA is selected
    if "Liquidity-Managed DCA" in strategies_to_compare or "Enhanced Liquidity-Managed DCA" in strategies_to_compare:
        st.sidebar.subheader("Liquidity-Managed DCA Parameters")
        st.sidebar.markdown("Configure investment amounts for different market conditions:")
        
        # Liquidity-Managed DCA specific parameters - using percentages of remaining capital
        invest_increase = st.sidebar.slider(
            "X: Invest When Price Increases (%)",
            min_value=0.1, 
            max_value=5.0,
            value=0.5, 
            step=0.1,
            help="Percentage of remaining capital to invest when price increases"
        )
        
        invest_flat = st.sidebar.slider(
            "Y: Invest When Price is Flat (%)",
            min_value=0.1, 
            max_value=5.0,
            value=1.0, 
            step=0.1,
            help="Percentage of remaining capital to invest when price stays flat"
        )
        
        invest_drop_significant = st.sidebar.slider(
            "Z: Invest On Significant Drops (%)",
            min_value=0.1, 
            max_value=10.0,
            value=3.0, 
            step=0.1,
            help="Percentage of remaining capital to invest on significant price drops (>= drop threshold)"
        )
        
        invest_drop_non_significant = st.sidebar.slider(
            "W: Invest On Non-Significant Drops (%)",
            min_value=0.1, 
            max_value=5.0,
            value=1.5, 
            step=0.1,
            help="Percentage of remaining capital to invest on smaller price drops (< drop threshold)"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("Threshold settings:")
        
        drop_threshold = st.sidebar.slider(
            "U: Drop Threshold (%)",
            min_value=0.1, 
            max_value=5.0,
            value=1.5, 
            step=0.1,
            help="Threshold percentage (U%) for defining a significant drop"
        )
        
        exit_profit_margin = st.sidebar.slider(
            "K: Exit Profit Target (%)",
            min_value=0.1, 
            max_value=10.0,
            value=2.0, 
            step=0.1,
            help="Target profit percentage (K%) that triggers selling"
        )
        
        # Parameters for Enhanced Liquidity-Managed DCA
        if "Enhanced Liquidity-Managed DCA" in strategies_to_compare:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Enhanced DCA Reserve Parameters")
            
            reserve_cash = st.sidebar.number_input(
                "Reserve Capital (USDT)",
                min_value=0.0,
                max_value=100000.0,
                value=3000.0,
                step=100.0,
                help="Additional reserve capital that can be used when needed for larger market opportunities (investment amounts calculated as % of remaining capital)"
            )
    
    # Parameters for other strategies
    if any(strat in strategies_to_compare for strat in ["Original DCA", "Alternative DCA", "Fixed Investment DCA"]):
        st.sidebar.subheader("Standard DCA Parameters")
        investment_pct = st.sidebar.slider("Investment Percentage per Buy", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
        price_drop_threshold = st.sidebar.slider("Price Drop Threshold (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        profit_threshold = st.sidebar.slider("Profit Target (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    
    trading_fee_pct = st.sidebar.slider("Trading Fee (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    # Button to run comparison
    if st.sidebar.button("Run Comparison"):
        # Initialize progress
        st.session_state.progress_bar = st.progress(0)
        st.session_state.status_text = st.empty()
        
        # Dictionary to store results for all selected strategies
        results = {}
        
        # Calculate progress step based on number of strategies
        progress_step = 90 / len(strategies_to_compare)
        
        # Run backtest for each selected strategy
        for i, strategy_name in enumerate(strategies_to_compare):
            progress_start = 5 + i * progress_step
            
            st.subheader(f"{strategy_name}")
            update_progress(progress_start, f"Initializing {strategy_name} backtest")
            
            # Initialize and run the appropriate strategy
            if strategy_name == "Original DCA":
                strategy = VariableTimeframeDCA(
                    initial_capital=initial_capital,
                    investment_pct=investment_pct,
                    price_drop_threshold=price_drop_threshold,
                    profit_threshold=profit_threshold,
                    trading_fee_pct=trading_fee_pct
                )
            elif strategy_name == "Alternative DCA":
                strategy = AlternativeDCA(
                    initial_capital=initial_capital,
                    investment_pct=investment_pct,
                    price_drop_threshold=price_drop_threshold,
                    profit_threshold=profit_threshold,
                    trading_fee_pct=trading_fee_pct
                )
            elif strategy_name == "Fixed Investment DCA":
                strategy = FixedInvestmentDCA(
                    initial_capital=initial_capital,
                    investment_pct=investment_pct,
                    price_drop_threshold=price_drop_threshold,
                    profit_threshold=profit_threshold,
                    trading_fee_pct=trading_fee_pct
                )
            elif strategy_name == "Liquidity-Managed DCA":
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
            elif strategy_name == "Enhanced Liquidity-Managed DCA":
                strategy = LiquidityManagedDCANew(
                    initial_capital=initial_capital,
                    reserve_cash=reserve_cash,
                    exit_profit_margin=exit_profit_margin,
                    invest_increase=invest_increase,
                    invest_flat=invest_flat,
                    invest_drop_significant=invest_drop_significant,
                    invest_drop_non_significant=invest_drop_non_significant,
                    drop_threshold=drop_threshold,
                    trading_fee_pct=trading_fee_pct
                )
                
            backtester = Backtester(filtered_data, strategy)
            results[strategy_name] = backtester.run(
                progress_callback=lambda p, m, s=strategy_name, ps=progress_start: 
                    update_progress(ps + (p * progress_step / 100), f"{s}: {m}")
            )
            
            # Debug: Check results structure for each strategy
            st.write(f"{strategy_name} results keys:", list(results[strategy_name].keys()) if results[strategy_name] else "None")
        
        # Comparison completed
        update_progress(100, "Comparison completed")
        
        # Display results
        st.subheader("Performance Summary")
        
        # Create comparison table
        comparison_metrics = [
            "return_pct", "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
            "trades_count", "win_rate", "profit_factor", "avg_profit_pct"
        ]
        
        comparison_labels = {
            "return_pct": "Total Return (%)",
            "max_drawdown_pct": "Maximum Drawdown (%)",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "trades_count": "Number of Trades",
            "win_rate": "Win Rate (%)",
            "profit_factor": "Profit Factor",
            "avg_profit_pct": "Average Profit per Trade (%)"
        }
        
        # Create comparison DataFrame
        comparison_data = {"Metric": [comparison_labels.get(m, m) for m in comparison_metrics]}
        
        # Add a column for each strategy
        for strategy_name in strategies_to_compare:
            comparison_data[strategy_name] = [f"{results[strategy_name].get(m, 0):.2f}" for m in comparison_metrics]
            
        comparison_df = pd.DataFrame(comparison_data)
        
        st.table(comparison_df)
        
        # Performance Visualization
        st.subheader("Equity Curves")
        
        try:
            # Check if equity_curve is present in each strategy's results
            for strategy_name in strategies_to_compare:
                if 'equity_curve' not in results[strategy_name]:
                    st.error(f"Error: Missing equity curve data in {strategy_name} backtest results. Cannot generate equity curve comparison.")
                    st.write(f"{strategy_name} results keys:", list(results[strategy_name].keys()) if results[strategy_name] else "None")
                    return
            
            # Combine equity curves for all strategies
            equity_dfs = {}
            common_index = pd.DatetimeIndex([])
            
            for strategy_name in strategies_to_compare:
                equity_df = results[strategy_name]['equity_curve'][['equity']].copy()
                equity_df.columns = [strategy_name]
                equity_dfs[strategy_name] = equity_df
                common_index = common_index.union(equity_df.index)
            
            # Create a DataFrame with all equity curves
            all_equity = pd.DataFrame(index=common_index)
            for strategy_name in strategies_to_compare:
                all_equity[strategy_name] = equity_dfs[strategy_name][strategy_name]
            
        except Exception as e:
            st.error(f"Error generating equity curves: {str(e)}")
            for strategy_name in strategies_to_compare:
                st.write(f"{strategy_name} results keys:", list(results[strategy_name].keys()) if results[strategy_name] else "None")
            return
        
        # Buy & Hold equity curve
        buy_hold_data = None
        # Use the first strategy that has buy_hold_equity
        for strategy_name in strategies_to_compare:
            if 'buy_hold_equity' in results[strategy_name]:
                buy_hold_equity = pd.DataFrame(results[strategy_name]['buy_hold_equity'], columns=['Buy & Hold'])
                all_equity['Buy & Hold'] = buy_hold_equity['Buy & Hold']
                break
        
        # Plot combined equity curve
        fig = go.Figure()
        
        for col in all_equity.columns:
            fig.add_trace(go.Scatter(
                x=all_equity.index,
                y=all_equity[col],
                mode='lines',
                name=col
            ))
        
        fig.update_layout(
            title='Equity Curves Comparison',
            xaxis_title='Date',
            yaxis_title='Equity (USDT)',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown Comparison
        st.subheader("Drawdown Comparison")
        
        try:
            # Check if equity_curve is present in all results
            for strategy_name in strategies_to_compare:
                if 'equity_curve' not in results[strategy_name]:
                    st.error(f"Error: Missing equity curve data in {strategy_name} backtest results. Cannot generate drawdown comparison.")
                    return
            
            # Extract drawdown data for all strategies
            drawdown_dfs = {}
            
            for strategy_name in strategies_to_compare:
                # Get drawdown column, or create zeros if missing
                if 'drawdown' in results[strategy_name]['equity_curve'].columns:
                    drawdown_df = results[strategy_name]['equity_curve'][['drawdown']].copy()
                else:
                    drawdown_df = pd.DataFrame(
                        index=results[strategy_name]['equity_curve'].index, 
                        data={'drawdown': 0}
                    )
                drawdown_df.columns = [strategy_name]
                drawdown_dfs[strategy_name] = drawdown_df
            
            # Create combined drawdown DataFrame with common index
            all_drawdowns = pd.DataFrame(index=common_index)
            for strategy_name in strategies_to_compare:
                all_drawdowns[strategy_name] = drawdown_dfs[strategy_name][strategy_name]
                
        except Exception as e:
            st.error(f"Error generating drawdown comparison: {str(e)}")
            return
        
        # Plot combined drawdown chart
        fig = go.Figure()
        
        for col in all_drawdowns.columns:
            fig.add_trace(go.Scatter(
                x=all_drawdowns.index,
                y=all_drawdowns[col],
                mode='lines',
                name=col,
                fill='tozeroy'
            ))
        
        # Calculate minimum drawdown value (most negative)
        min_drawdown = all_drawdowns.values.min()
        
        fig.update_layout(
            title='Drawdown Comparison',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[min_drawdown * 1.1, 0])  # Set range explicitly to invert y-axis for drawdowns (removed rangemode)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade Analysis
        st.subheader("Trade Analysis")
        
        # Create a flexible layout based on the number of strategies
        num_strategies = len(strategies_to_compare)
        num_cols = min(2, num_strategies)  # At most 2 columns
        cols = st.columns(num_cols)
        
        # Display trades for each strategy
        for i, strategy_name in enumerate(strategies_to_compare):
            col_idx = i % num_cols  # Determine which column to use
            
            with cols[col_idx]:
                st.write(f"**{strategy_name} Trades**")
                strategy_trades = results[strategy_name].get('trades', [])
                
                # Check if trades is a list or DataFrame
                if isinstance(strategy_trades, list) and len(strategy_trades) > 0:
                    trades_df = pd.DataFrame(strategy_trades)
                    trades_df['date'] = pd.to_datetime(trades_df['date'])
                    trades_df.set_index('date', inplace=True)
                    st.dataframe(trades_df.sort_index(ascending=False).head(10))
                    st.write(f"Total trades: {len(strategy_trades)}")
                elif isinstance(strategy_trades, pd.DataFrame) and not strategy_trades.empty:
                    # If it's already a DataFrame
                    trades_df = strategy_trades.copy()
                    if 'date' in trades_df.columns and not trades_df.index.name == 'date':
                        trades_df['date'] = pd.to_datetime(trades_df['date'])
                        trades_df.set_index('date', inplace=True)
                    st.dataframe(trades_df.sort_index(ascending=False).head(10))
                    st.write(f"Total trades: {len(trades_df)}")
                else:
                    st.write("No trades executed")
        
        # Monthly Returns Comparison
        st.subheader("Monthly Returns Comparison")
        
        try:
            # Check if equity_curve is present in each strategy's results
            for strategy_name in strategies_to_compare:
                if 'equity_curve' not in results[strategy_name]:
                    st.error(f"Error: Missing equity curve data in {strategy_name} backtest results. Cannot generate monthly returns.")
                    return
            
            # Create a flexible layout based on the number of strategies
            num_strategies = len(strategies_to_compare)
            num_cols = min(2, num_strategies)  # At most 2 columns
            cols = st.columns(num_cols)
            
            # Display monthly returns for each strategy
            for i, strategy_name in enumerate(strategies_to_compare):
                col_idx = i % num_cols  # Determine which column to use
                
                with cols[col_idx]:
                    st.write(f"**{strategy_name} Monthly Returns**")
                    monthly_fig = PerformancePlots.plot_monthly_returns(
                        results[strategy_name]['equity_curve'][['equity']]
                    )
                    st.plotly_chart(monthly_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating monthly returns: {str(e)}")
            return
            
        # Trading activity comparison
        st.subheader("Trading Activity")
        
        # Create price chart with trade markers
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['close'],
            mode='lines',
            name='BTC Price',
            line=dict(color='black', width=1)
        ))
        
        # Define marker styles for different strategies
        marker_styles = {
            0: {
                'buy': dict(symbol='triangle-up', size=10, color='green'),
                'sell': dict(symbol='triangle-down', size=10, color='red')
            },
            1: {
                'buy': dict(symbol='circle', size=8, color='blue'),
                'sell': dict(symbol='x', size=8, color='purple')
            },
            2: {
                'buy': dict(symbol='diamond', size=9, color='orange'),
                'sell': dict(symbol='cross', size=9, color='brown')
            }
        }
        
        # Process trade data for each strategy
        try:
            for i, strategy_name in enumerate(strategies_to_compare):
                # Get marker style for this strategy
                style_idx = i % len(marker_styles)
                current_markers = marker_styles[style_idx]
                
                # Get trades for this strategy
                strategy_trades = results[strategy_name].get('trades', [])
                
                # Handle both list and DataFrame formats
                if isinstance(strategy_trades, list) and len(strategy_trades) > 0:
                    # List format
                    buys = []
                    sells = []
                    for t in strategy_trades:
                        if isinstance(t, dict) and 'type' in t:
                            if t['type'] == 'buy':
                                buys.append(t)
                            elif t['type'] == 'sell':
                                sells.append(t)
                    
                    if buys:
                        buy_dates = [t['date'] for t in buys]
                        buy_prices = [t['price'] for t in buys]
                        fig.add_trace(go.Scatter(
                            x=buy_dates,
                            y=buy_prices,
                            mode='markers',
                            name=f'{strategy_name} Buys',
                            marker=current_markers['buy']
                        ))
                    
                    if sells:
                        sell_dates = [t['date'] for t in sells]
                        sell_prices = [t['price'] for t in sells]
                        fig.add_trace(go.Scatter(
                            x=sell_dates,
                            y=sell_prices,
                            mode='markers',
                            name=f'{strategy_name} Sells',
                            marker=current_markers['sell']
                        ))
                elif isinstance(strategy_trades, pd.DataFrame) and not strategy_trades.empty:
                    # DataFrame format
                    buys = strategy_trades[strategy_trades['type'] == 'buy']
                    sells = strategy_trades[strategy_trades['type'] == 'sell']
                    
                    if not buys.empty:
                        fig.add_trace(go.Scatter(
                            x=buys.index if buys.index.name == 'date' else buys['date'],
                            y=buys['price'],
                            mode='markers',
                            name=f'{strategy_name} Buys',
                            marker=current_markers['buy']
                        ))
                    
                    if not sells.empty:
                        fig.add_trace(go.Scatter(
                            x=sells.index if sells.index.name == 'date' else sells['date'],
                            y=sells['price'],
                            mode='markers',
                            name=f'{strategy_name} Sells',
                            marker=current_markers['sell']
                        ))
        except Exception as e:
            st.warning(f"Error plotting trades: {str(e)}")
        
        fig.update_layout(
            title='Trading Activity Comparison',
            xaxis_title='Date',
            yaxis_title='BTC Price (USDT)',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Conclusion
        st.subheader("Strategy Comparison Conclusion")
        
        # Calculate summary metrics for comparison
        risk_reward_ratios = {}
        
        for strategy_name in strategies_to_compare:
            return_pct = results[strategy_name].get('return_pct', 0)
            drawdown_pct = results[strategy_name].get('max_drawdown_pct', 0)
            
            # Calculate risk-reward ratio (return divided by max drawdown)
            risk_reward = abs(return_pct / drawdown_pct) if drawdown_pct != 0 else 0
            risk_reward_ratios[strategy_name] = risk_reward
        
        # Find the best strategy based on risk-reward ratio
        if risk_reward_ratios:
            best_strategy = max(risk_reward_ratios.items(), key=lambda x: x[1])[0]
            
            # Create risk-reward ratio string
            risk_reward_text = "**Risk-Return Ratio:**\n"
            for strategy_name, ratio in risk_reward_ratios.items():
                risk_reward_text += f"- {strategy_name}: {ratio:.2f}\n"
            
            st.write(f"""
            {risk_reward_text}
            
            Based on this specific backtest period and parameter set, the **{best_strategy}** delivered better risk-adjusted returns.
            
            Remember that different market conditions may favor different strategy variations. Consider running multiple tests across various market cycles for a more comprehensive comparison.
            """)
        else:
            st.write("""
            No comparison results available. Please run the backtest with at least one strategy selected.
            """)

# Run the strategy comparison UI
run_strategy_comparison()