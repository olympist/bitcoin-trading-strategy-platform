import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import json
from datetime import datetime, date
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.results_manager import ResultsManager
from visualization import PerformancePlots

st.set_page_config(
    page_title="Results Analysis - Bitcoin Backtesting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #f7931a;
    }
    .highlight {
        color: #f7931a;
        font-weight: bold;
    }
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stButton button {
        background-color: #f7931a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title("Results Analysis")
st.markdown("View, compare, and analyze saved backtests, optimization results, and Monte Carlo simulations.")

# Initialize the results manager
results_manager = ResultsManager()

# Define function to view detailed result
def view_detailed_result(result):
    """
    Display a detailed view of a backtest result, similar to the backtest page
    """
    st.markdown(f"## Detailed Result: {result['name']}")
    
    # Add a back button
    if st.button("← Back to Results"):
        # Clear the viewing state and rerun
        st.session_state.viewing_result_details = None
        st.rerun()
    
    # Check what type of result we're viewing
    result_type = result.get('type', 'backtest')
    
    # Handle V7 strategy backtest results specifically
    if 'v7' in result.get('strategy', '').lower() or 'v7' in result.get('name', '').lower():
        # Create tabs similar to the backtest page
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
            "Performance Summary", "Equity & Drawdown", "Trade Analysis", "Advanced Analytics"
        ])
        
        # Tab 1: Performance Summary
        with result_tab1:
            st.markdown("### Strategy Performance")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{result.get('return_pct', 0):.2f}%",
                    delta=f"{result.get('return_pct', 0) - 0:.2f}%" if result.get('return_pct', 0) != 0 else None
                )
            
            with col2:
                st.metric(
                    "Max Drawdown",
                    f"{result.get('max_drawdown_pct', 0):.2f}%"
                )
            
            with col3:
                st.metric(
                    "Win Rate",
                    f"{result.get('win_rate', 0):.2f}%"
                )
            
            with col4:
                st.metric(
                    "Trades",
                    f"{result.get('trades_count', 0)}"
                )
            
            # Strategy parameters
            st.markdown("### Strategy Parameters")
            params = result.get('parameters', {})
            
            params_df = pd.DataFrame({
                'Parameter': list(params.keys()),
                'Value': [str(v) for v in params.values()]
            })
            
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        # Tab 2: Equity Curve and Drawdown
        with result_tab2:
            st.markdown("### Equity Curve and Drawdown Analysis")
            
            # Check if we have equity curve data
            if 'equity_curve' in result and result['equity_curve']:
                # Convert data to DataFrame if it's in records format
                if isinstance(result['equity_curve'], list):
                    equity_df = pd.DataFrame(result['equity_curve'])
                    # Convert date strings to datetime
                    if 'date' in equity_df.columns:
                        equity_df['date'] = pd.to_datetime(equity_df['date'])
                        equity_df.set_index('date', inplace=True)
                else:
                    equity_df = result['equity_curve']
                
                # Create equity figure
                equity_fig = go.Figure()
                
                # Handle different column names that might be present
                equity_col = next((col for col in ['equity', 'portfolio_value', 'value'] 
                                  if col in equity_df.columns), None)
                
                buyhold_col = next((col for col in ['buy_hold_equity', 'buy_hold_value', 'buy_hold'] 
                                    if col in equity_df.columns), None)
                
                if equity_col:
                    # Add equity curve line
                    equity_fig.add_trace(
                        go.Scatter(
                            x=equity_df.index,
                            y=equity_df[equity_col],
                            name='Strategy',
                            line=dict(color='#1f77b4', width=2)
                        )
                    )
                
                if buyhold_col:
                    # Add buy and hold curve
                    equity_fig.add_trace(
                        go.Scatter(
                            x=equity_df.index,
                            y=equity_df[buyhold_col],
                            name='Buy & Hold',
                            line=dict(color='#ff7f0e', width=2, dash='dash')
                        )
                    )
                
                # Format the chart
                equity_fig.update_layout(
                    title='Equity Curve: Strategy vs Buy & Hold',
                    xaxis_title='Date',
                    yaxis_title='Equity Value (USDT)',
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
                
                st.plotly_chart(equity_fig, use_container_width=True)
                
                # Drawdown chart
                if 'drawdown' in equity_df.columns:
                    drawdown_fig = go.Figure()
                    
                    drawdown_fig.add_trace(
                        go.Scatter(
                            x=equity_df.index,
                            y=equity_df['drawdown'] * 100,  # Convert to percentage
                            name='Drawdown',
                            fill='tozeroy',
                            line=dict(color='#d62728')
                        )
                    )
                    
                    drawdown_fig.update_layout(
                        title='Drawdown Analysis',
                        xaxis_title='Date',
                        yaxis_title='Drawdown (%)',
                        template='plotly_white',
                        yaxis=dict(
                            autorange='reversed',  # Invert Y-axis to show drawdowns
                            tickformat='.1f'
                        ),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(drawdown_fig, use_container_width=True)
                
                # BTC Holdings chart
                btc_holdings_col = None
                for column in ['btc_holdings', 'btc_amount', 'holdings']:
                    if column in equity_df.columns:
                        btc_holdings_col = column
                        break
                
                if btc_holdings_col is not None and not equity_df[btc_holdings_col].isnull().all():
                    holdings_fig = go.Figure()
                    
                    holdings_fig.add_trace(
                        go.Scatter(
                            x=equity_df.index,
                            y=equity_df[btc_holdings_col],
                            name='BTC Holdings',
                            line=dict(color='#2ca02c', width=2),
                            fill='tozeroy'
                        )
                    )
                    
                    holdings_fig.update_layout(
                        title='Bitcoin Capital Growth Over Time',
                        xaxis_title='Date',
                        yaxis_title='BTC Amount',
                        template='plotly_white',
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    # Add initial vs final BTC annotation
                    if not equity_df[btc_holdings_col].empty:
                        initial_btc = equity_df[btc_holdings_col].iloc[0] if not equity_df[btc_holdings_col].iloc[0] == 0 else equity_df[btc_holdings_col].iloc[1]
                        final_btc = equity_df[btc_holdings_col].iloc[-1]
                        if initial_btc > 0 and final_btc > 0:
                            btc_growth_pct = (final_btc / initial_btc - 1) * 100
                            holdings_fig.add_annotation(
                                x=1,
                                y=1,
                                xref="paper",
                                yref="paper",
                                text=f"BTC Growth: {btc_growth_pct:.2f}% (Initial: {initial_btc:.6f} → Final: {final_btc:.6f})",
                                showarrow=False,
                                font=dict(
                                    size=12,
                                    color="#2ca02c"
                                ),
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="#2ca02c",
                                borderwidth=1,
                                borderpad=4,
                                align="right"
                            )
                    
                    st.plotly_chart(holdings_fig, use_container_width=True)
                
                # If BTC holdings not directly available, try to calculate it
                elif 'position_value' in equity_df.columns and 'close' in equity_df.columns:
                    close_prices = equity_df['close'].replace(0, np.nan).fillna(method='ffill')
                    derived_holdings = equity_df['position_value'] / close_prices
                    derived_holdings = derived_holdings.fillna(0).replace([np.inf, -np.inf], 0)
                    
                    if not derived_holdings.isnull().all() and derived_holdings.sum() > 0:
                        holdings_fig = go.Figure()
                        
                        holdings_fig.add_trace(
                            go.Scatter(
                                x=equity_df.index,
                                y=derived_holdings,
                                name='BTC Holdings (Calculated)',
                                line=dict(color='#2ca02c', width=2),
                                fill='tozeroy'
                            )
                        )
                        
                        holdings_fig.update_layout(
                            title='Bitcoin Capital Growth Over Time (Calculated)',
                            xaxis_title='Date',
                            yaxis_title='BTC Amount',
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        # Add initial vs final BTC annotation
                        if not derived_holdings.empty:
                            initial_btc = derived_holdings.iloc[0] if not derived_holdings.iloc[0] == 0 else derived_holdings.iloc[1]
                            final_btc = derived_holdings.iloc[-1]
                            if initial_btc > 0 and final_btc > 0:
                                btc_growth_pct = (final_btc / initial_btc - 1) * 100
                                holdings_fig.add_annotation(
                                    x=1,
                                    y=1,
                                    xref="paper",
                                    yref="paper",
                                    text=f"BTC Growth: {btc_growth_pct:.2f}% (Initial: {initial_btc:.6f} → Final: {final_btc:.6f})",
                                    showarrow=False,
                                    font=dict(
                                        size=12,
                                        color="#2ca02c"
                                    ),
                                    bgcolor="rgba(255, 255, 255, 0.8)",
                                    bordercolor="#2ca02c",
                                    borderwidth=1,
                                    borderpad=4,
                                    align="right"
                                )
                        
                        st.plotly_chart(holdings_fig, use_container_width=True)
            else:
                st.warning("No equity curve data available for this result.")
        
        # Tab 3: Trade Analysis
        with result_tab3:
            st.markdown("### Trade Analysis")
            
            # Check if we have trades data
            if 'trades' in result and result['trades']:
                # Convert data to DataFrame if it's in records format
                if isinstance(result['trades'], list):
                    trades_df = pd.DataFrame(result['trades'])
                    
                    # Convert date strings to datetime
                    date_col = next((col for col in ['timestamp', 'date', 'buy_date'] 
                                    if col in trades_df.columns), None)
                    
                    if date_col and trades_df[date_col].dtype == 'object':
                        trades_df[date_col] = pd.to_datetime(trades_df[date_col])
                else:
                    trades_df = result['trades']
                
                # Display trade statistics
                st.markdown("#### Trade Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                try:
                    # Calculate trade statistics
                    profitable_trades = trades_df[trades_df['profit_pct'] > 0]
                    losing_trades = trades_df[trades_df['profit_pct'] <= 0]
                    
                    with col1:
                        st.metric(
                            "Total Trades",
                            f"{len(trades_df)}"
                        )
                    
                    with col2:
                        st.metric(
                            "Winning Trades",
                            f"{len(profitable_trades)}",
                            delta=f"{len(profitable_trades) / len(trades_df) * 100:.1f}%" if len(trades_df) > 0 else None
                        )
                    
                    with col3:
                        avg_win = profitable_trades['profit_pct'].mean() if not profitable_trades.empty else 0
                        st.metric(
                            "Avg. Win",
                            f"{avg_win:.2f}%"
                        )
                    
                    with col4:
                        avg_loss = losing_trades['profit_pct'].mean() if not losing_trades.empty else 0
                        st.metric(
                            "Avg. Loss",
                            f"{avg_loss:.2f}%"
                        )
                except:
                    st.warning("Could not calculate trade statistics from the available data.")
                
                # Display the trades table
                st.markdown("#### Trade List")
                
                # Format trades for display
                display_trades_df = trades_df.copy()
                
                # Convert date columns to readable format
                for col in display_trades_df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            display_trades_df[col] = pd.to_datetime(display_trades_df[col]).dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pass
                
                # Format prices with proper precision
                for col in ['price', 'entry_price', 'exit_price', 'buy_price', 'sell_price']:
                    if col in display_trades_df.columns:
                        try:
                            display_trades_df[col] = display_trades_df[col].apply(lambda x: f"${x:.2f}")
                        except:
                            pass
                
                # Format profit columns
                for col in ['profit_pct', 'profit_percent']:
                    if col in display_trades_df.columns:
                        try:
                            display_trades_df[col] = display_trades_df[col].apply(lambda x: f"{x:.2f}%")
                        except:
                            pass
                
                # Display the trades table
                st.dataframe(display_trades_df, use_container_width=True)
            else:
                st.warning("No trade data available for this result.")
        
        # Tab 4: Advanced Analytics
        with result_tab4:
            st.markdown("### Advanced Analytics")
            
            # Check if we have equity curve data for advanced analytics
            if 'equity_curve' in result and result['equity_curve']:
                # Convert data to DataFrame if it's in records format
                if isinstance(result['equity_curve'], list):
                    equity_df = pd.DataFrame(result['equity_curve'])
                    # Convert date strings to datetime
                    if 'date' in equity_df.columns:
                        equity_df['date'] = pd.to_datetime(equity_df['date'])
                        equity_df.set_index('date', inplace=True)
                else:
                    equity_df = result['equity_curve']
                
                # Monthly returns heatmap
                equity_col = next((col for col in ['equity', 'portfolio_value', 'value'] 
                                 if col in equity_df.columns), None)
                
                if equity_col and not equity_df.empty:
                    try:
                        st.markdown("#### Monthly Returns Heatmap")
                        
                        # Calculate monthly returns
                        equity_df = equity_df.copy()
                        if equity_df.index.dtype != 'datetime64[ns]':
                            equity_df.index = pd.to_datetime(equity_df.index)
                        
                        # Direct calculation for return if data is short (1 month or less)
                        if len(equity_df) >= 2:
                            # Calculate overall return directly
                            initial_value = equity_df[equity_col].iloc[0]
                            final_value = equity_df[equity_col].iloc[-1]
                            total_return_pct = (final_value / initial_value - 1) * 100
                            
                            # Get the month and year for the current data
                            start_date = equity_df.index[0]
                            end_date = equity_df.index[-1]
                            
                            # Always create at least one monthly return even if within the same month
                            # First check if we have more than one month
                            if (end_date.year > start_date.year) or (end_date.month > start_date.month):
                                # Use standard monthly resampling if spanning multiple months
                                monthly_equity = equity_df[equity_col].resample('ME').last()
                                
                                # Check if we have capital injection data
                                has_injection_data = False
                                injection_col = None
                                for col in ['capital_injection', 'deployment', 'injection', 'additional_capital']:
                                    if col in equity_df.columns:
                                        # Only mark as having injections if there are actual non-zero values
                                        if equity_df[col].sum() > 0:
                                            has_injection_data = True
                                            injection_col = col
                                            break
                                        
                                if has_injection_data and injection_col is not None:
                                    # Sum injections by month to account for capital deployments
                                    monthly_injections = equity_df[injection_col].fillna(0).resample('ME').sum()
                                    
                                    # Log the injection data for debugging
                                    st.write(f"Using '{injection_col}' for capital injection tracking")
                                    st.write(f"Total capital injection: ${equity_df[injection_col].sum():.2f}")
                                    
                                    # Calculate actual returns excluding capital injections
                                    prev_equity = monthly_equity.shift(1)
                                    actual_growth = monthly_equity - prev_equity - monthly_injections
                                    monthly_pct_change = (actual_growth / prev_equity * 100).fillna(0)
                                else:
                                    # If no injection data, use simple percentage change
                                    # This might overstate returns if capital was added
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
                        
                        # Create a Series with year and month as MultiIndex
                        monthly_returns = pd.Series(
                            monthly_pct_change.values,
                            index=pd.MultiIndex.from_arrays([
                                monthly_pct_change.index.year,
                                monthly_pct_change.index.strftime('%b')
                            ], names=['Year', 'Month'])
                        )
                        
                        # Convert to DataFrame and pivot
                        monthly_df = monthly_returns.reset_index()
                        monthly_df.columns = ['Year', 'Month', 'Return']
                        
                        # Get ordered months
                        ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        # Create pivot table (years as rows, months as columns)
                        monthly_pivot = pd.pivot_table(
                            monthly_df,
                            values='Return',
                            index='Year',
                            columns='Month',
                            aggfunc='first'
                        )
                        
                        # Reorder months
                        monthly_pivot = monthly_pivot.reindex(columns=ordered_months)
                        
                        # Create heatmap
                        monthly_fig = go.Figure(data=go.Heatmap(
                            z=monthly_pivot.values,
                            x=monthly_pivot.columns,
                            y=monthly_pivot.index,
                            colorscale=[
                                [0, 'rgb(165,0,38)'],  # Deep red for most negative
                                [0.25, 'rgb(215,48,39)'],  # Red
                                [0.5, 'rgb(255,255,191)'],  # Light yellow/white
                                [0.75, 'rgb(26,152,80)'],  # Green
                                [1, 'rgb(0,104,55)']  # Deep green for most positive
                            ],
                            colorbar=dict(
                                title='Return (%)',
                                titleside='top',
                                tickformat='.1f'
                            ),
                            text=[[f"{val:.2f}%" if not pd.isna(val) else "" for val in row] for row in monthly_pivot.values],
                            hoverinfo='text',
                            hoverongaps=False
                        ))
                        
                        monthly_fig.update_layout(
                            title='Monthly Returns Heatmap',
                            xaxis_title='Month',
                            yaxis_title='Year',
                            xaxis=dict(
                                tickangle=-30,
                                tickmode='array',
                                tickvals=ordered_months
                            ),
                            margin=dict(l=10, r=10, t=40, b=10),
                            height=400
                        )
                        
                        st.plotly_chart(monthly_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate monthly returns heatmap: {str(e)}")
                
                # Return distribution
                try:
                    st.markdown("#### Return Distribution")
                    
                    # Calculate daily returns
                    if equity_col and not equity_df.empty:
                        # Check if we have capital injection data for daily returns too
                        has_injection_data = False
                        for col in ['capital_injection', 'deployment', 'injection', 'additional_capital']:
                            if col in equity_df.columns:
                                # Only count as true if there are actual non-zero values
                                if equity_df[col].sum() > 0:
                                    has_injection_data = True
                                    injection_col = col
                                    break
                                
                        if has_injection_data and injection_col is not None:
                            # Get injections and calculate adjusted daily returns
                            injections = equity_df[injection_col].fillna(0)
                            prev_equity = equity_df[equity_col].shift(1)
                            actual_growth = equity_df[equity_col] - prev_equity - injections
                            daily_returns = (actual_growth / prev_equity * 100).dropna()
                            
                            # Add a note that returns are adjusted for capital injections
                            st.info(f"Daily returns adjusted for {equity_df[injection_col].sum():.2f} in capital injections")
                        else:
                            # Standard percentage change if no injection data
                            daily_returns = equity_df[equity_col].pct_change().dropna() * 100
                        
                        if len(daily_returns) > 5:  # Need enough data points
                            fig = go.Figure()
                            
                            # Histogram for daily returns
                            fig.add_trace(go.Histogram(
                                x=daily_returns,
                                name='Daily Returns',
                                nbinsx=30,
                                histnorm='probability',
                                marker_color='#1f77b4'
                            ))
                            
                            # Format the chart
                            fig.update_layout(
                                title='Daily Return Distribution',
                                xaxis_title='Daily Return (%)',
                                yaxis_title='Frequency',
                                template='plotly_white',
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            # Add vertical line at 0
                            fig.add_vline(x=0, line_dash="dash", line_color="red")
                            
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate return distribution chart: {str(e)}")
                
            else:
                st.warning("No equity curve data available for advanced analytics.")
    else:
        # For other types of results, display in a simpler format
        st.markdown("### Result Details")
        
        # Display metrics based on the result type
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Return",
                f"{result.get('return_pct', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                "Max Drawdown",
                f"{result.get('max_drawdown_pct', 0):.2f}%"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{result.get('win_rate', 0):.2f}%"
            )
        
        with col4:
            st.metric(
                "Trades",
                f"{result.get('trades_count', 0)}"
            )
        
        # Display parameters
        if 'parameters' in result:
            st.markdown("### Parameters")
            
            params_df = pd.DataFrame({
                'Parameter': list(result['parameters'].keys()),
                'Value': [str(v) for v in result['parameters'].values()]
            })
            
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        # Display data based on result type
        if 'equity_curve' in result and result['equity_curve']:
            st.markdown("### Equity Curve")
            
            # Convert data to DataFrame if needed
            if isinstance(result['equity_curve'], list):
                equity_df = pd.DataFrame(result['equity_curve'])
            else:
                equity_df = result['equity_curve']
            
            # Plot equity curve
            if not equity_df.empty:
                st.line_chart(equity_df.set_index('date')['equity'] if 'date' in equity_df.columns else equity_df['equity'])
        
        if 'trades' in result and result['trades']:
            st.markdown("### Trades")
            
            # Convert data to DataFrame if needed
            if isinstance(result['trades'], list):
                trades_df = pd.DataFrame(result['trades'])
            else:
                trades_df = result['trades']
            
            # Display trades
            if not trades_df.empty:
                st.dataframe(trades_df)

# Function to format datetime objects
def format_datetime(dt):
    if isinstance(dt, (datetime, date)):
        return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)

# Load all saved results
backtest_results = results_manager.list_backtest_results(limit=100)
optimization_results = results_manager.list_optimization_results(limit=100)

# Scan for V7 strategy backtest results
v7_results = []
v7_pattern = "v7_strategy_backtest_"
results_dir = "results"

if os.path.exists(results_dir):
    for filename in os.listdir(results_dir):
        if filename.startswith(v7_pattern) and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                # Try to parse the timestamp from the filename
                timestamp_str = filename.replace(v7_pattern, "").replace(".json", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                
                # Load and parse the JSON data
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Create a result entry similar to the other results
                v7_result = {
                    'filepath': filepath,
                    'strategy': "V7 Strategy",
                    'timestamp': timestamp,
                    'type': 'backtest',
                    'name': f"V7 Strategy Backtest ({timestamp_str})",
                    'return_pct': data.get('return_pct', 0.0) if 'return_pct' in data else 
                                 data.get('parameters', {}).get('return_pct', 0.0),
                    'max_drawdown_pct': data.get('max_drawdown_pct', 0.0),
                    'win_rate': data.get('win_rate', 0.0),
                    'trades_count': len(data.get('trades', [])),
                    'parameters': data.get('parameters', {}),
                    'data': data
                }
                
                v7_results.append(v7_result)
            except Exception as e:
                st.error(f"Error loading V7 result {filename}: {str(e)}")

# Function to initialize session state variables if not present
def initialize_session_state():
    if 'viewing_result_details' not in st.session_state:
        st.session_state.viewing_result_details = None
    
    if 'saved_v7_results_data' not in st.session_state:
        st.session_state.saved_v7_results_data = {}

# Initialize session state
initialize_session_state()

# Combine all results
all_saved_results = backtest_results + optimization_results + v7_results

# Add session-stored v7 results if available
if st.session_state.saved_v7_results_data:
    for key, data in st.session_state.saved_v7_results_data.items():
        # Extract timestamp from the key (format: v7_strategy_backtest_YYYYMMDD_HHMM)
        timestamp_str = key.replace("v7_strategy_backtest_", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
        except:
            timestamp = datetime.now()
            
        # Check if this result is already in the list
        if not any(r.get('name', '') == key for r in all_saved_results):
            # Create a result entry with all necessary data
            result_entry = {
                'name': key,
                'strategy': 'V7 Strategy',
                'timestamp': timestamp,
                'return_pct': data.get('return_pct', 0),
                'max_drawdown_pct': data.get('max_drawdown_pct', 0),
                'win_rate': data.get('win_rate', 0),
                'trades_count': data.get('trades_count', 0),
                'sharpe_ratio': data.get('sharpe_ratio', 0),
                'type': 'backtest',
                'parameters': data.get('parameters', {}),
                'equity_curve': data.get('equity_curve', []),
                'trades': data.get('trades', []),
                'is_session_data': True  # Mark as session data
            }
            all_saved_results.append(result_entry)

# Sort results by timestamp (newest first)
all_saved_results = sorted(all_saved_results, key=lambda x: x['timestamp'] if isinstance(x['timestamp'], (datetime, date)) else datetime.now(), reverse=True)

# Function to initialize session state for checkboxes
if 'initialized' not in st.session_state:
    for i in range(len(all_saved_results)):
        st.session_state[f"selected_{i}"] = False
    st.session_state.initialized = True

# Check if viewing a specific result
if 'viewing_result_details' in st.session_state and st.session_state.viewing_result_details is not None:
    # Display detailed view of the result
    view_detailed_result(st.session_state.viewing_result_details)
# Main content
elif not all_saved_results:
    st.info("No saved results found. Run a backtest or Monte Carlo simulation and save the results to compare them here.")
else:
    # Create tabs for different comparison views
    tab1, tab2, tab3, tab4 = st.tabs(["Results Overview", "Summary Analysis", "Detailed Comparison", "Export Results"])
    
    # Tab 1: Results Overview
    with tab1:
        st.markdown("### Saved Results")
        
        # Display cards for each saved result
        for i, result in enumerate(all_saved_results):
            with st.container():
                st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
                
                # Determine result type
                result_type = result.get('type', 'backtest')  # Default to backtest if not specified
                
                # Create a header row with name and timestamp
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"#### {result['name']}")
                    st.markdown(f"**Type:** {'Monte Carlo Simulation' if result_type == 'monte_carlo' else 'Backtest' if result_type == 'backtest' else 'Optimization'}")
                
                with col2:
                    timestamp = result['timestamp']
                    if isinstance(timestamp, datetime):
                        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
                    else:
                        timestamp_str = str(timestamp)
                    st.markdown(f"**Date:** {timestamp_str}")
                    
                    # Show strategy and parameters
                    if 'strategy' in result:
                        st.markdown(f"**Strategy:** {result['strategy']}")
                
                with col3:
                    col3a, col3b = st.columns([1, 1])
                    with col3a:
                        # Add checkbox for result selection
                        st.checkbox("Select", key=f"selected_{i}")
                    with col3b:
                        # Add a view details button
                        if st.button("View", key=f"view_details_{i}"):
                            # Store the selected result in session state
                            st.session_state.viewing_result_details = result
                            # Redirect to detail view
                            st.rerun()
                
                # Display result metrics and charts based on type
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    if result_type == 'monte_carlo':
                        # Monte Carlo metrics
                        summary = result.get('summary', {})
                        
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                        with metrics_col1:
                            mean_return = summary.get('return_pct', {}).get('mean', 0)
                            st.metric(
                                "Mean Return",
                                f"{mean_return:.2f}%",
                                delta=f"{mean_return - 0:.2f}%" if mean_return != 0 else None
                            )
                        
                        with metrics_col2:
                            st.metric(
                                "Probability of Profit",
                                f"{summary.get('probability_of_profit', 0):.1f}%"
                            )
                        
                        with metrics_col3:
                            mean_drawdown = summary.get('max_drawdown_pct', {}).get('mean', 0)
                            st.metric(
                                "Mean Max Drawdown",
                                f"{mean_drawdown:.2f}%"
                            )
                        
                        with metrics_col4:
                            st.metric(
                                "Iterations",
                                f"{len(result.get('results', []))}"
                            )
                    
                    elif result_type == 'optimization':
                        # Optimization metrics
                        top_results = result.get('results', [])
                        if top_results:
                            top_result = top_results[0]  # Best result
                            
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            
                            with metrics_col1:
                                return_pct = top_result.get('return_pct', 0)
                                st.metric(
                                    "Best Return",
                                    f"{return_pct:.2f}%",
                                    delta=f"{return_pct - 0:.2f}%" if return_pct != 0 else None
                                )
                            
                            with metrics_col2:
                                st.metric(
                                    "Max Drawdown",
                                    f"{top_result.get('max_drawdown_pct', 0):.2f}%"
                                )
                            
                            with metrics_col3:
                                st.metric(
                                    "Win Rate",
                                    f"{top_result.get('win_rate', 0):.2f}%"
                                )
                            
                            with metrics_col4:
                                st.metric(
                                    "Combinations",
                                    f"{len(top_results)}"
                                )
                    
                    else:
                        # Regular backtest metrics
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
                        
                        with metrics_col1:
                            return_pct = result.get('return_pct', 0)
                            st.metric(
                                "USDT Return",
                                f"{return_pct:.2f}%",
                                delta=f"{return_pct - 0:.2f}%" if return_pct != 0 else None
                            )
                        
                        with metrics_col2:
                            # Calculate BTC profit using trades data if available
                            btc_profit = 0
                            initial_btc = 0
                            final_btc = 0
                            
                            # Try to get BTC profit information
                            if 'btc_profit_pct' in result:
                                btc_profit = result.get('btc_profit_pct', 0)
                            elif 'trades' in result and isinstance(result['trades'], list) and len(result['trades']) > 0:
                                # Calculate BTC profit from trades
                                trades = result['trades']
                                # Sum all BTC bought
                                for trade in trades:
                                    if 'type' in trade and trade['type'] == 'buy' and 'quantity' in trade:
                                        initial_btc += float(trade['quantity'])
                                
                                # Find final BTC amount from equity curve if available
                                if 'equity_curve' in result and result['equity_curve']:
                                    equity_df = result['equity_curve']
                                    if isinstance(equity_df, list) and len(equity_df) > 0:
                                        equity_df = pd.DataFrame(equity_df)
                                    
                                    # Check for BTC holdings column
                                    for col in ['btc_holdings', 'btc_amount', 'holdings']:
                                        if col in equity_df.columns:
                                            final_btc = equity_df[col].iloc[-1] if not equity_df.empty else 0
                                            break
                                
                                # Calculate BTC profit as percentage
                                if initial_btc > 0 and final_btc > 0:
                                    btc_profit = (final_btc / initial_btc - 1) * 100
                            
                            st.metric(
                                "BTC Growth",
                                f"{btc_profit:.2f}%" if btc_profit != 0 else "N/A",
                                delta=f"{btc_profit:.2f}%" if btc_profit > 0 else None
                            )
                        
                        with metrics_col3:
                            st.metric(
                                "Max Drawdown",
                                f"{result.get('max_drawdown_pct', 0):.2f}%"
                            )
                        
                        with metrics_col4:
                            st.metric(
                                "Win Rate",
                                f"{result.get('win_rate', 0):.2f}%"
                            )
                        
                        with metrics_col5:
                            trades_count = result.get('trades_count', 0)
                            st.metric(
                                "Trades",
                                f"{trades_count}"
                            )
                
                with col2:
                    if result_type == 'monte_carlo':
                        # Simple distribution chart for Monte Carlo
                        if 'results' in result and len(result['results']) > 0:
                            returns = [r.get('return_pct', 0) for r in result['results']]
                            
                            fig = px.histogram(
                                x=returns,
                                nbins=30,
                                labels={'x': 'Return (%)'},
                                title='Return Distribution',
                                color_discrete_sequence=['#f7931a']
                            )
                            
                            # Add mean line
                            fig.add_vline(
                                x=np.mean(returns), 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text=f"Mean: {np.mean(returns):.2f}%",
                                annotation_position="top right"
                            )
                            
                            fig.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif result_type == 'optimization':
                        # Parameter heatmap for top parameters
                        if 'results' in result and len(result['results']) > 0:
                            param_keys = [k for k in result['results'][0]['params'].keys()]
                            if len(param_keys) >= 2:
                                # Create dataframe for plot
                                param_data = []
                                for r in result['results'][:20]:  # Top 20 results
                                    row = {k: r['params'][k] for k in param_keys}
                                    row['Return (%)'] = r.get('return_pct', 0)
                                    row['Max Drawdown (%)'] = r.get('max_drawdown_pct', 0)
                                    param_data.append(row)
                                
                                param_df = pd.DataFrame(param_data)
                                
                                # Pick the top 2 most varying parameters
                                param_variances = {}
                                for param in param_keys:
                                    if len(set(param_df[param])) > 1:
                                        param_variances[param] = param_df[param].var()
                                
                                if len(param_variances) >= 2:
                                    top_params = sorted(param_variances.items(), key=lambda x: x[1], reverse=True)[:2]
                                    x_param, y_param = top_params[0][0], top_params[1][0]
                                    
                                    fig = px.scatter(
                                        param_df, 
                                        x=x_param,
                                        y=y_param,
                                        color='Return (%)',
                                        size='Return (%)',
                                        hover_data=['Max Drawdown (%)'],
                                        color_continuous_scale='Viridis',
                                        title=f'Parameter Space: {x_param} vs {y_param}'
                                    )
                                    
                                    fig.update_layout(
                                        height=200,
                                        margin=dict(l=0, r=0, t=30, b=0)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Not enough parameter variation for parameter space visualization.")
                            else:
                                st.info("Not enough parameters for parameter space visualization.")
                    
                    else:
                        # Equity curve for backtest
                        if 'equity_curve' in result and result['equity_curve']:
                            # Convert dict-based equity curve to DataFrame if necessary
                            if isinstance(result['equity_curve'], dict):
                                equity_curve = pd.DataFrame(result['equity_curve'])
                            else:
                                equity_curve = result['equity_curve']
                            
                            # Create a simplified equity curve
                            fig = go.Figure()
                            
                            # Add equity curve
                            fig.add_trace(go.Scatter(
                                x=list(range(len(equity_curve))),
                                y=equity_curve['equity'],
                                mode='lines',
                                name='Equity',
                                line=dict(color='#f7931a', width=2)
                            ))
                            
                            fig.update_layout(
                                title='Equity Curve',
                                height=200,
                                margin=dict(l=0, r=0, t=30, b=0),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # View details button
                if st.button("View Details", key=f"view_{i}"):
                    with st.expander("Result Details", expanded=True):
                        if result_type == 'monte_carlo':
                            # Monte Carlo details
                            st.markdown("#### Monte Carlo Simulation Details")
                            
                            # Display parameters
                            st.markdown("##### Configuration")
                            param_cols = st.columns(3)
                            
                            with param_cols[0]:
                                st.markdown(f"**Iterations:** {len(result.get('results', []))}")
                            
                            with param_cols[1]:
                                st.markdown(f"**Initial Capital:** ${result.get('parameters', {}).get('initial_capital', 0):.2f}")
                            
                            with param_cols[2]:
                                if 'parameters' in result and 'method' in result['parameters']:
                                    st.markdown(f"**Method:** {result['parameters']['method']}")
                            
                            # Show distribution chart
                            if 'results' in result and len(result['results']) > 0:
                                returns = [r.get('return_pct', 0) for r in result['results']]
                                drawdowns = [r.get('max_drawdown_pct', 0) for r in result['results']]
                                
                                dist_col1, dist_col2 = st.columns(2)
                                
                                with dist_col1:
                                    fig = PerformancePlots.plot_return_distribution(
                                        returns,
                                        "Return Distribution"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with dist_col2:
                                    fig = PerformancePlots.plot_return_distribution(
                                        drawdowns,
                                        "Drawdown Distribution"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Risk metrics
                                st.markdown("##### Risk Metrics")
                                summary = result.get('summary', {})
                                
                                risk_col1, risk_col2, risk_col3 = st.columns(3)
                                
                                with risk_col1:
                                    st.metric(
                                        "Probability of Profit",
                                        f"{summary.get('probability_of_profit', 0):.1f}%"
                                    )
                                
                                with risk_col2:
                                    st.metric(
                                        "Prob. Return > 10%",
                                        f"{summary.get('probability_return_over_10', 0):.1f}%"
                                    )
                                
                                with risk_col3:
                                    st.metric(
                                        "Prob. Drawdown > 20%",
                                        f"{summary.get('probability_drawdown_over_20', 0):.1f}%"
                                    )
                                
                                # Show sample equity curves
                                st.markdown("##### Sample Equity Curves")
                                
                                if 'equity_curves' in result and result['equity_curves']:
                                    # Plot a few sample equity curves
                                    fig = PerformancePlots.plot_monte_carlo_equity_curves(
                                        result['results'][:20],  # First 20 simulations
                                        num_curves=10
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        elif result_type == 'optimization':
                            # Optimization details
                            st.markdown("#### Optimization Details")
                            
                            # Display parameters
                            st.markdown("##### Configuration")
                            if 'parameters' in result:
                                params = result['parameters']
                                param_cols = st.columns(3)
                                
                                with param_cols[0]:
                                    st.markdown(f"**Strategy:** {result.get('strategy', 'Unknown')}")
                                
                                with param_cols[1]:
                                    st.markdown(f"**Initial Capital:** ${params.get('initial_capital', 0):.2f}")
                                
                                with param_cols[2]:
                                    if 'method' in params:
                                        st.markdown(f"**Method:** {params['method']}")
                            
                            # Parameter ranges
                            if 'parameter_ranges' in result:
                                st.markdown("##### Parameter Ranges")
                                ranges = result['parameter_ranges']
                                
                                range_data = []
                                for param, values in ranges.items():
                                    if isinstance(values, list) and len(values) > 0:
                                        range_data.append({
                                            'Parameter': param,
                                            'Min': min(values),
                                            'Max': max(values),
                                            'Values': ", ".join([str(v) for v in values])
                                        })
                                
                                range_df = pd.DataFrame(range_data)
                                st.dataframe(range_df, use_container_width=True)
                            
                            # Top results
                            if 'results' in result and len(result['results']) > 0:
                                st.markdown("##### Top Results")
                                
                                top_results = result['results'][:10]  # Top 10 results
                                
                                # Create DataFrame for display
                                top_data = []
                                for i, r in enumerate(top_results):
                                    row = {'Rank': i + 1}
                                    row.update({k: v for k, v in r['params'].items()})
                                    row['Return (%)'] = r.get('return_pct', 0)
                                    row['Max Drawdown (%)'] = r.get('max_drawdown_pct', 0)
                                    row['Win Rate (%)'] = r.get('win_rate', 0)
                                    row['Trades'] = r.get('trades_count', 0)
                                    
                                    top_data.append(row)
                                
                                top_df = pd.DataFrame(top_data)
                                st.dataframe(top_df, use_container_width=True)
                                
                                # Parameter impact visualization
                                st.markdown("##### Parameter Impact")
                                
                                param_keys = [k for k in result['results'][0]['params'].keys()]
                                if len(param_keys) > 0:
                                    param_to_analyze = st.selectbox(
                                        "Select parameter to analyze",
                                        param_keys
                                    )
                                    
                                    # Create dataframe for plot
                                    param_data = []
                                    for r in result['results']:
                                        row = {
                                            'Parameter': r['params'][param_to_analyze],
                                            'Return (%)': r.get('return_pct', 0),
                                            'Max Drawdown (%)': r.get('max_drawdown_pct', 0),
                                            'Name': f"{r['params']}"
                                        }
                                        param_data.append(row)
                                    
                                    param_df = pd.DataFrame(param_data)
                                    
                                    # Create scatter plot
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=param_df['Parameter'],
                                        y=param_df['Return (%)'],
                                        mode='markers',
                                        marker=dict(
                                            size=10,
                                            color=param_df['Max Drawdown (%)'],
                                            colorscale='Viridis',
                                            colorbar=dict(title="Max Drawdown (%)"),
                                            opacity=0.8
                                        ),
                                        text=param_df['Name'],
                                        hovertemplate=
                                        '<b>%{text}</b><br>' +
                                        f'{param_to_analyze}: %{{x}}<br>' +
                                        'Return: %{y:.2f}%<br>' +
                                        'Max Drawdown: %{marker.color:.2f}%<br>' +
                                        '<extra></extra>'
                                    ))
                                    
                                    fig.update_layout(
                                        title=f'Return vs. {param_to_analyze}',
                                        xaxis_title=param_to_analyze,
                                        yaxis_title='Return (%)',
                                        template='plotly_white',
                                        margin=dict(l=0, r=0, t=40, b=0)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            # Backtest details
                            st.markdown("#### Backtest Details")
                            
                            # Display parameters
                            st.markdown("##### Configuration")
                            if 'parameters' in result:
                                params = result['parameters']
                                param_cols = st.columns(3)
                                
                                with param_cols[0]:
                                    st.markdown(f"**Strategy:** {result.get('strategy', 'Unknown')}")
                                
                                with param_cols[1]:
                                    st.markdown(f"**Initial Capital:** ${params.get('initial_capital', 0):.2f}")
                                
                                with param_cols[2]:
                                    if 'period' in params:
                                        st.markdown(f"**Period:** {params['period']}")
                            
                            # Parameter values
                            if 'parameters' in result:
                                st.markdown("##### Parameters")
                                
                                # Filter out non-parameters
                                param_dict = {k: v for k, v in result['parameters'].items() 
                                            if k not in ['initial_capital', 'period']}
                                
                                if param_dict:
                                    param_data = [{'Parameter': k, 'Value': v} for k, v in param_dict.items()]
                                    param_df = pd.DataFrame(param_data)
                                    st.dataframe(param_df, use_container_width=True)
                            
                            # Performance metrics
                            st.markdown("##### Performance Metrics")
                            
                            metric_cols = st.columns(4)
                            
                            with metric_cols[0]:
                                st.metric(
                                    "Return (%)",
                                    f"{result.get('return_pct', 0):.2f}%"
                                )
                            
                            with metric_cols[1]:
                                st.metric(
                                    "Max Drawdown (%)",
                                    f"{result.get('max_drawdown_pct', 0):.2f}%"
                                )
                            
                            with metric_cols[2]:
                                st.metric(
                                    "Win Rate (%)",
                                    f"{result.get('win_rate', 0):.2f}%"
                                )
                            
                            with metric_cols[3]:
                                st.metric(
                                    "Trades",
                                    f"{result.get('trades_count', 0)}"
                                )
                            
                            # Show equity curve and drawdown
                            st.markdown("##### Equity and Drawdown")
                            
                            eq_col1, eq_col2 = st.columns(2)
                            
                            # Equity curve
                            with eq_col1:
                                if 'equity_curve' in result and result['equity_curve']:
                                    # Convert dict-based equity curve to DataFrame if necessary
                                    if isinstance(result['equity_curve'], dict):
                                        equity_curve = pd.DataFrame(result['equity_curve'])
                                    else:
                                        equity_curve = result['equity_curve']
                                    
                                    # Add a toggle for normalized view
                                    normalize_equity = st.checkbox("Normalize equity curves (show percentage change)", value=True,
                                                                help="When checked, both curves start at 100% for easier comparison")
                                    
                                    fig = PerformancePlots.plot_equity_curve(equity_curve, normalized=normalize_equity)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Drawdown chart
                            with eq_col2:
                                if 'equity_curve' in result and result['equity_curve']:
                                    # Convert dict-based equity curve to DataFrame if necessary
                                    if isinstance(result['equity_curve'], dict):
                                        equity_curve = pd.DataFrame(result['equity_curve'])
                                    else:
                                        equity_curve = result['equity_curve']
                                    
                                    fig = PerformancePlots.plot_drawdown_chart(equity_curve)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Trades visualization if available
                            if 'trades' in result and result['trades']:
                                st.markdown("##### Trades")
                                
                                # Convert dict-based trades to DataFrame if necessary
                                if isinstance(result['trades'], dict):
                                    trades = pd.DataFrame(result['trades'])
                                else:
                                    trades = result['trades']
                                
                                # Display trades table
                                st.dataframe(trades, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Button to clear all selections
        if st.button("Clear All Selections"):
            for i in range(len(all_saved_results)):
                st.session_state[f"selected_{i}"] = False
    
    # Tab 2: Summary Analysis
    with tab2:
        st.markdown("### Summary Analysis")
        
        # Get selected results for analysis
        selected_indices = [i for i in range(len(all_saved_results)) if st.session_state.get(f"selected_{i}", False)]
        selected_results = [all_saved_results[i] for i in selected_indices]
        
        if not selected_results:
            st.info("Please select at least one result for analysis from the Results Overview tab.")
        else:
            # Create analysis tabs
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Overall Performance", "Strategy Comparison", "Parameter Analysis"])
            
            # Tab 1: Overall Performance
            with analysis_tab1:
                st.markdown("#### Overall Performance Metrics")
                
                # Group results by type
                backtest_group = [r for r in selected_results if r.get('type', 'backtest') == 'backtest']
                monte_carlo_group = [r for r in selected_results if r.get('type') == 'monte_carlo']
                optimization_group = [r for r in selected_results if r.get('type') == 'optimization']
                
                # Create metrics based on result types
                metrics_data = []
                
                # Process backtest results
                for result in backtest_group:
                    metrics_data.append({
                        'Name': result['name'],
                        'Type': 'Backtest',
                        'Return (%)': result.get('return_pct', 0),
                        'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
                        'Win Rate (%)': result.get('win_rate', 0),
                        'Trades': result.get('trades_count', 0),
                        'Sharpe Ratio': result.get('sharpe_ratio', 0),
                        'Profit Factor': result.get('profit_factor', 0)
                    })
                
                # Process Monte Carlo results (use mean values)
                for result in monte_carlo_group:
                    summary = result.get('summary', {})
                    metrics_data.append({
                        'Name': result['name'],
                        'Type': 'Monte Carlo',
                        'Return (%)': summary.get('return_pct', {}).get('mean', 0),
                        'Max Drawdown (%)': summary.get('max_drawdown_pct', {}).get('mean', 0),
                        'Win Rate (%)': summary.get('win_rate', {}).get('mean', 0),
                        'Trades': summary.get('trades_count', {}).get('mean', 0),
                        'Sharpe Ratio': summary.get('sharpe_ratio', {}).get('mean', 0),
                        'Profit Factor': summary.get('profit_factor', {}).get('mean', 0)
                    })
                
                # Process optimization results (use best result)
                for result in optimization_group:
                    if 'results' in result and len(result['results']) > 0:
                        top_result = result['results'][0]  # Best result
                        metrics_data.append({
                            'Name': result['name'],
                            'Type': 'Optimization',
                            'Return (%)': top_result.get('return_pct', 0),
                            'Max Drawdown (%)': top_result.get('max_drawdown_pct', 0),
                            'Win Rate (%)': top_result.get('win_rate', 0),
                            'Trades': top_result.get('trades_count', 0),
                            'Sharpe Ratio': top_result.get('sharpe_ratio', 0),
                            'Profit Factor': top_result.get('profit_factor', 0)
                        })
                
                if metrics_data:
                    # Display comparison metrics table
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Create visual comparison charts
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Return comparison
                        return_fig = px.bar(
                            metrics_df,
                            x='Name',
                            y='Return (%)',
                            color='Type',
                            barmode='group',
                            title='Return Comparison',
                            color_discrete_sequence=['#f7931a', '#1f77b4', '#2ca02c']
                        )
                        
                        return_fig.update_layout(
                            xaxis_title='',
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(return_fig, use_container_width=True)
                    
                    with chart_col2:
                        # Drawdown comparison
                        dd_fig = px.bar(
                            metrics_df,
                            x='Name',
                            y='Max Drawdown (%)',
                            color='Type',
                            barmode='group',
                            title='Drawdown Comparison',
                            color_discrete_sequence=['#f7931a', '#1f77b4', '#2ca02c']
                        )
                        
                        dd_fig.update_layout(
                            xaxis_title='',
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(dd_fig, use_container_width=True)
                    
                    # Create scatter plot of return vs drawdown
                    risk_reward_fig = px.scatter(
                        metrics_df,
                        x='Max Drawdown (%)',
                        y='Return (%)',
                        color='Type',
                        size='Trades',
                        hover_data=['Name', 'Win Rate (%)'],
                        title='Risk-Reward Analysis',
                        color_discrete_sequence=['#f7931a', '#1f77b4', '#2ca02c']
                    )
                    
                    risk_reward_fig.update_layout(
                        template='plotly_white',
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(risk_reward_fig, use_container_width=True)
                else:
                    st.info("No valid results data found for analysis.")
            
            # Tab 2: Strategy Comparison
            with analysis_tab2:
                st.markdown("#### Strategy Comparison")
                
                # Extract strategy information
                strategy_data = []
                
                for result in selected_results:
                    result_type = result.get('type', 'backtest')
                    strategy_name = result.get('strategy', 'Unknown')
                    
                    if result_type == 'backtest':
                        strategy_data.append({
                            'Name': result['name'],
                            'Strategy': strategy_name,
                            'Return (%)': result.get('return_pct', 0),
                            'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
                            'Win Rate (%)': result.get('win_rate', 0),
                            'Trades': result.get('trades_count', 0)
                        })
                    elif result_type == 'optimization' and 'results' in result and len(result['results']) > 0:
                        top_result = result['results'][0]  # Best result
                        strategy_data.append({
                            'Name': result['name'],
                            'Strategy': strategy_name,
                            'Return (%)': top_result.get('return_pct', 0),
                            'Max Drawdown (%)': top_result.get('max_drawdown_pct', 0),
                            'Win Rate (%)': top_result.get('win_rate', 0),
                            'Trades': top_result.get('trades_count', 0)
                        })
                
                if strategy_data:
                    # Group by strategy
                    strategy_df = pd.DataFrame(strategy_data)
                    
                    # Display strategy comparison table
                    st.dataframe(strategy_df, use_container_width=True)
                    
                    # Create grouped bar chart by strategy
                    if len(set(strategy_df['Strategy'])) > 1:
                        strategy_fig = px.bar(
                            strategy_df,
                            x='Strategy',
                            y=['Return (%)', 'Max Drawdown (%)'],
                            barmode='group',
                            title='Strategy Performance Comparison',
                            color_discrete_sequence=['#f7931a', '#ef5350']
                        )
                        
                        strategy_fig.update_layout(
                            xaxis_title='',
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0),
                            legend_title=''
                        )
                        
                        st.plotly_chart(strategy_fig, use_container_width=True)
                        
                        # Win rate and trade count by strategy
                        trades_fig = px.scatter(
                            strategy_df,
                            x='Win Rate (%)',
                            y='Trades',
                            color='Strategy',
                            size='Return (%)',
                            hover_data=['Name', 'Max Drawdown (%)'],
                            title='Trade Efficiency by Strategy',
                            labels={'Win Rate (%)': 'Win Rate (%)', 'Trades': 'Number of Trades'}
                        )
                        
                        trades_fig.update_layout(
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(trades_fig, use_container_width=True)
                    else:
                        st.info("Only one strategy found in selected results. Please select results with different strategies for comparison.")
                else:
                    st.info("No backtest or optimization results found for strategy comparison.")
            
            # Tab 3: Parameter Analysis
            with analysis_tab3:
                st.markdown("#### Parameter Analysis")
                
                # Extract parameter information from backtest and optimization results
                param_results = []
                
                for result in selected_results:
                    result_type = result.get('type', 'backtest')
                    
                    if result_type == 'backtest' and 'parameters' in result:
                        # Extract key parameters
                        params = result['parameters']
                        param_row = {
                            'Name': result['name'],
                            'Type': 'Backtest',
                            'Strategy': result.get('strategy', 'Unknown'),
                            'Return (%)': result.get('return_pct', 0),
                            'Max Drawdown (%)': result.get('max_drawdown_pct', 0)
                        }
                        
                        # Add all parameters to the row
                        for param_key, param_value in params.items():
                            if param_key not in ['initial_capital', 'period']:
                                param_row[param_key] = param_value
                        
                        param_results.append(param_row)
                    
                    elif result_type == 'optimization' and 'results' in result:
                        # Extract top results
                        for i, opt_result in enumerate(result['results'][:5]):  # Top 5 results
                            if 'params' in opt_result:
                                param_row = {
                                    'Name': f"{result['name']} (Rank {i+1})",
                                    'Type': 'Optimization',
                                    'Strategy': result.get('strategy', 'Unknown'),
                                    'Return (%)': opt_result.get('return_pct', 0),
                                    'Max Drawdown (%)': opt_result.get('max_drawdown_pct', 0)
                                }
                                
                                # Add parameters to the row
                                for param_key, param_value in opt_result['params'].items():
                                    param_row[param_key] = param_value
                                
                                param_results.append(param_row)
                
                if param_results:
                    param_df = pd.DataFrame(param_results)
                    
                    # Get all parameter columns
                    param_cols = [col for col in param_df.columns if col not in ['Name', 'Type', 'Strategy', 'Return (%)', 'Max Drawdown (%)']]
                    
                    if param_cols:
                        # Display parameter comparison table
                        st.dataframe(param_df, use_container_width=True)
                        
                        # Select parameter for visualization
                        if len(param_cols) > 0:
                            param_to_analyze = st.selectbox(
                                "Select parameter to analyze",
                                param_cols
                            )
                            
                            # Create visual comparison
                            if param_to_analyze in param_df.columns:
                                try:
                                    # Convert to numeric if possible for better visualization
                                    param_df[param_to_analyze] = pd.to_numeric(param_df[param_to_analyze], errors='ignore')
                                    
                                    # Create scatter plot
                                    param_fig = px.scatter(
                                        param_df,
                                        x=param_to_analyze,
                                        y='Return (%)',
                                        color='Strategy',
                                        size='Max Drawdown (%)',
                                        hover_data=['Name', 'Type'],
                                        title=f'Parameter Impact: {param_to_analyze}'
                                    )
                                    
                                    param_fig.update_layout(
                                        template='plotly_white',
                                        margin=dict(l=0, r=0, t=40, b=0)
                                    )
                                    
                                    st.plotly_chart(param_fig, use_container_width=True)
                                    
                                    # Create box plot for parameter distributions
                                    if isinstance(param_df[param_to_analyze].iloc[0], (int, float)):
                                        box_fig = px.box(
                                            param_df,
                                            x='Strategy',
                                            y=param_to_analyze,
                                            color='Type',
                                            points='all',
                                            title=f'Parameter Distribution: {param_to_analyze}'
                                        )
                                        
                                        box_fig.update_layout(
                                            template='plotly_white',
                                            margin=dict(l=0, r=0, t=40, b=0)
                                        )
                                        
                                        st.plotly_chart(box_fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not create visualization: {str(e)}")
                    else:
                        st.info("No parameter data found in selected results.")
                else:
                    st.info("No parameter data found in selected results.")
    
    # Tab 3: Detailed Comparison
    with tab3:
        st.markdown("### Detailed Results Comparison")
        
        # Get selected results for comparison
        selected_indices = [i for i in range(len(all_saved_results)) if st.session_state.get(f"selected_{i}", False)]
        selected_results = [all_saved_results[i] for i in selected_indices]
        
        if len(selected_results) < 2:
            st.info("Please select at least two results for detailed comparison.")
        else:
            # Create comparison tabs
            comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Return Analysis", "Risk Metrics", "Parameter Impact"])
            
            # Tab 1: Return Analysis
            with comp_tab1:
                st.markdown("#### Return Metrics Comparison")
                
                # Create return metrics table
                return_data = []
                
                for result in selected_results:
                    result_type = result.get('type', 'backtest')
                    
                    if result_type == 'monte_carlo':
                        # Monte Carlo return metrics
                        summary = result.get('summary', {})
                        row = {
                            'Name': result['name'],
                            'Type': 'Monte Carlo',
                            'Mean Return (%)': summary.get('return_pct', {}).get('mean', 0),
                            'Median Return (%)': summary.get('return_pct', {}).get('median', 0),
                            'Min Return (%)': summary.get('return_pct', {}).get('min', 0),
                            'Max Return (%)': summary.get('return_pct', {}).get('max', 0),
                            'Std Dev': summary.get('return_pct', {}).get('std', 0),
                            'Probability of Profit (%)': summary.get('probability_of_profit', 0)
                        }
                    elif result_type == 'optimization':
                        # Optimization return metrics (take best result)
                        top_results = result.get('results', [])
                        if top_results:
                            best_return = top_results[0].get('return_pct', 0)
                            worst_return = top_results[-1].get('return_pct', 0)
                            returns = [r.get('return_pct', 0) for r in top_results]
                            
                            row = {
                                'Name': result['name'],
                                'Type': 'Optimization',
                                'Mean Return (%)': np.mean(returns),
                                'Median Return (%)': np.median(returns),
                                'Min Return (%)': np.min(returns),
                                'Max Return (%)': np.max(returns),
                                'Std Dev': np.std(returns),
                                'Probability of Profit (%)': (np.array(returns) > 0).mean() * 100
                            }
                        else:
                            continue
                    else:
                        # Regular backtest return metrics
                        row = {
                            'Name': result['name'],
                            'Type': 'Backtest',
                            'Mean Return (%)': result.get('return_pct', 0),
                            'Median Return (%)': result.get('return_pct', 0),
                            'Min Return (%)': result.get('return_pct', 0),
                            'Max Return (%)': result.get('return_pct', 0),
                            'Std Dev': 0,
                            'Probability of Profit (%)': 100 if result.get('return_pct', 0) > 0 else 0
                        }
                    
                    return_data.append(row)
                
                # Display return metrics table
                return_df = pd.DataFrame(return_data)
                st.dataframe(return_df, use_container_width=True)
                
                # Create visual comparison
                st.markdown("#### Return Comparison Visualization")
                
                # Bar chart for returns
                return_fig = px.bar(
                    return_df,
                    x='Name',
                    y=['Mean Return (%)', 'Median Return (%)'],
                    barmode='group',
                    error_y=return_df['Std Dev'],
                    color_discrete_sequence=['#f7931a', '#1f77b4'],
                    title='Return Comparison'
                )
                
                return_fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Return (%)',
                    template='plotly_white',
                    boxmode='group',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(return_fig, use_container_width=True)
                
                # Box plot for Monte Carlo and optimization results
                mc_opt_results = [r for r in selected_results if r.get('type') in ['monte_carlo', 'optimization']]
                
                if mc_opt_results:
                    box_data = []
                    
                    for result in mc_opt_results:
                        result_type = result.get('type')
                        
                        if result_type == 'monte_carlo' and 'results' in result:
                            for r in result['results']:
                                box_data.append({
                                    'Name': result['name'],
                                    'Return (%)': r.get('return_pct', 0)
                                })
                        elif result_type == 'optimization' and 'results' in result:
                            for r in result['results']:
                                box_data.append({
                                    'Name': result['name'],
                                    'Return (%)': r.get('return_pct', 0)
                                })
                    
                    if box_data:
                        box_df = pd.DataFrame(box_data)
                        
                        box_fig = px.box(
                            box_df,
                            x='Name',
                            y='Return (%)',
                            title='Return Distribution',
                            color='Name'
                        )
                        
                        box_fig.update_layout(
                            showlegend=False,
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(box_fig, use_container_width=True)
            
            # Tab 2: Risk Metrics
            with comp_tab2:
                st.markdown("#### Risk Metrics Comparison")
                
                # Create risk metrics table
                risk_data = []
                
                for result in selected_results:
                    result_type = result.get('type', 'backtest')
                    
                    if result_type == 'monte_carlo':
                        # Monte Carlo risk metrics
                        summary = result.get('summary', {})
                        row = {
                            'Name': result['name'],
                            'Type': 'Monte Carlo',
                            'Mean Max Drawdown (%)': summary.get('max_drawdown_pct', {}).get('mean', 0),
                            'Worst Drawdown (%)': summary.get('max_drawdown_pct', {}).get('max', 0),
                            'Sharpe Ratio': summary.get('sharpe_ratio', {}).get('mean', 0),
                            'Mean Win Rate (%)': summary.get('win_rate', {}).get('mean', 0),
                            'Profit Factor': summary.get('profit_factor', {}).get('mean', 0),
                            'High Risk Event (%)': summary.get('probability_drawdown_over_20', 0)
                        }
                    elif result_type == 'optimization':
                        # Optimization risk metrics (take best result)
                        top_results = result.get('results', [])
                        if top_results:
                            best_result = top_results[0]
                            drawdowns = [r.get('max_drawdown_pct', 0) for r in top_results]
                            
                            row = {
                                'Name': result['name'],
                                'Type': 'Optimization',
                                'Mean Max Drawdown (%)': np.mean(drawdowns),
                                'Worst Drawdown (%)': np.max(drawdowns),
                                'Sharpe Ratio': best_result.get('sharpe_ratio', 0),
                                'Mean Win Rate (%)': best_result.get('win_rate', 0),
                                'Profit Factor': best_result.get('profit_factor', 0),
                                'High Risk Event (%)': (np.array(drawdowns) > 20).mean() * 100
                            }
                        else:
                            continue
                    else:
                        # Regular backtest risk metrics
                        row = {
                            'Name': result['name'],
                            'Type': 'Backtest',
                            'Mean Max Drawdown (%)': result.get('max_drawdown_pct', 0),
                            'Worst Drawdown (%)': result.get('max_drawdown_pct', 0),
                            'Sharpe Ratio': result.get('sharpe_ratio', 0),
                            'Mean Win Rate (%)': result.get('win_rate', 0),
                            'Profit Factor': result.get('profit_factor', 0),
                            'High Risk Event (%)': 100 if result.get('max_drawdown_pct', 0) > 20 else 0
                        }
                    
                    risk_data.append(row)
                
                # Display risk metrics table
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)
                
                # Create visual comparison
                st.markdown("#### Risk Profile Visualization")
                
                # Bar chart for drawdown
                drawdown_fig = px.bar(
                    risk_df,
                    x='Name',
                    y='Mean Max Drawdown (%)',
                    color='Type',
                    title='Drawdown Comparison',
                    color_discrete_sequence=['#f7931a', '#1f77b4', '#2ca02c']
                )
                
                drawdown_fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Max Drawdown (%)',
                    template='plotly_white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(drawdown_fig, use_container_width=True)
                
                # Scatter plot for risk-reward
                risk_reward_fig = px.scatter(
                    risk_df,
                    x='Mean Max Drawdown (%)',
                    y='Mean Win Rate (%)',
                    color='Type',
                    size='Profit Factor',
                    hover_data=['Name', 'Sharpe Ratio'],
                    title='Risk-Reward Profile',
                    color_discrete_sequence=['#f7931a', '#1f77b4', '#2ca02c']
                )
                
                risk_reward_fig.update_layout(
                    template='plotly_white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(risk_reward_fig, use_container_width=True)
            
            # Tab 3: Parameter Impact
            with comp_tab3:
                st.markdown("#### Parameter Impact Analysis")
                
                # Need to collect parameter info from backtests and optimizations
                param_impact_data = []
                
                for result in selected_results:
                    result_type = result.get('type', 'backtest')
                    
                    if result_type == 'backtest' and 'parameters' in result:
                        # Extract key parameters
                        params = result['parameters']
                        param_row = {
                            'Name': result['name'],
                            'Type': 'Backtest',
                            'Return (%)': result.get('return_pct', 0),
                            'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
                            'Win Rate (%)': result.get('win_rate', 0)
                        }
                        
                        # Add all parameters to the row
                        for param_key, param_value in params.items():
                            if param_key not in ['initial_capital', 'period']:
                                param_row[param_key] = param_value
                        
                        param_impact_data.append(param_row)
                    
                    elif result_type == 'optimization' and 'results' in result:
                        # Extract top results
                        for i, opt_result in enumerate(result['results'][:5]):  # Top 5 results
                            if 'params' in opt_result:
                                param_row = {
                                    'Name': f"{result['name']} (Rank {i+1})",
                                    'Type': 'Optimization',
                                    'Return (%)': opt_result.get('return_pct', 0),
                                    'Max Drawdown (%)': opt_result.get('max_drawdown_pct', 0),
                                    'Win Rate (%)': opt_result.get('win_rate', 0)
                                }
                                
                                # Add parameters to the row
                                for param_key, param_value in opt_result['params'].items():
                                    param_row[param_key] = param_value
                                
                                param_impact_data.append(param_row)
                
                if param_impact_data:
                    param_impact_df = pd.DataFrame(param_impact_data)
                    
                    # Get all parameter columns
                    param_cols = [col for col in param_impact_df.columns if col not in ['Name', 'Type', 'Return (%)', 'Max Drawdown (%)', 'Win Rate (%)']]
                    
                    if param_cols:
                        # Display parameter comparison table
                        st.dataframe(param_impact_df, use_container_width=True)
                        
                        # Select parameters for visualization
                        if len(param_cols) >= 2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                x_param = st.selectbox(
                                    "X-axis parameter",
                                    param_cols,
                                    key="x_param"
                                )
                            
                            with col2:
                                y_param = st.selectbox(
                                    "Y-axis parameter",
                                    [p for p in param_cols if p != x_param],
                                    key="y_param"
                                )
                            
                            # Create 3D scatter plot
                            if x_param in param_impact_df.columns and y_param in param_impact_df.columns:
                                try:
                                    # Convert to numeric if possible for better visualization
                                    param_impact_df[x_param] = pd.to_numeric(param_impact_df[x_param], errors='ignore')
                                    param_impact_df[y_param] = pd.to_numeric(param_impact_df[y_param], errors='ignore')
                                    
                                    # Create 3D scatter plot
                                    fig = px.scatter_3d(
                                        param_impact_df,
                                        x=x_param,
                                        y=y_param,
                                        z='Return (%)',
                                        color='Type',
                                        size='Win Rate (%)',
                                        hover_data=['Name', 'Max Drawdown (%)'],
                                        title='Parameter Space Visualization'
                                    )
                                    
                                    fig.update_layout(
                                        scene=dict(
                                            xaxis_title=x_param,
                                            yaxis_title=y_param,
                                            zaxis_title='Return (%)'
                                        ),
                                        template='plotly_white',
                                        margin=dict(l=0, r=0, t=40, b=0),
                                        height=700
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not create 3D visualization: {str(e)}")
                        elif len(param_cols) == 1:
                            param = param_cols[0]
                            
                            try:
                                # Convert to numeric if possible for better visualization
                                param_impact_df[param] = pd.to_numeric(param_impact_df[param], errors='ignore')
                                
                                # Create scatter plot
                                fig = px.scatter(
                                    param_impact_df,
                                    x=param,
                                    y='Return (%)',
                                    color='Type',
                                    size='Max Drawdown (%)',
                                    hover_data=['Name', 'Win Rate (%)'],
                                    title=f'Parameter Impact: {param}'
                                )
                                
                                fig.update_layout(
                                    template='plotly_white',
                                    margin=dict(l=0, r=0, t=40, b=0)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not create visualization: {str(e)}")
                    else:
                        st.info("No parameter data found in selected results.")
                else:
                    st.info("No backtest or optimization results found with parameter data.")
    
    # Tab 4: Export Results
    with tab4:
        st.markdown("### Export Results")
        
        # Get selected results for export
        selected_indices = [i for i in range(len(all_saved_results)) if st.session_state.get(f"selected_{i}", False)]
        selected_results = [all_saved_results[i] for i in selected_indices]
        
        if not selected_results:
            st.info("Please select at least one result to export.")
        else:
            # Create two columns for different operations
            export_col, save_col = st.columns(2)
            
            # Export options in the first column
            with export_col:
                st.markdown("#### Export to File")
                
                export_format = st.radio(
                    "Choose export format",
                    ["CSV", "JSON"],
                    index=0
                )
                
                export_what = st.radio(
                    "Choose what to export",
                    ["Summary Metrics", "Full Results", "Trade Data"],
                    index=0
                )
                
                if st.button("Generate Export"):
                    if export_what == "Summary Metrics":
                        # Export summary metrics
                        summary_data = []
                        
                        for result in selected_results:
                            result_type = result.get('type', 'backtest')
                            
                            if result_type == 'monte_carlo':
                                summary = result.get('summary', {})
                                row = {
                                    'Name': result['name'],
                                    'Type': 'Monte Carlo',
                                    'Date': format_datetime(result['timestamp']),
                                    'Mean Return (%)': summary.get('return_pct', {}).get('mean', 0),
                                    'Mean Max Drawdown (%)': summary.get('max_drawdown_pct', {}).get('mean', 0),
                                    'Mean Win Rate (%)': summary.get('win_rate', {}).get('mean', 0),
                                    'Probability of Profit (%)': summary.get('probability_of_profit', 0)
                                }
                            elif result_type == 'optimization' and 'results' in result and len(result['results']) > 0:
                                top_result = result['results'][0]
                                row = {
                                    'Name': result['name'],
                                    'Type': 'Optimization',
                                    'Date': format_datetime(result['timestamp']),
                                    'Best Return (%)': top_result.get('return_pct', 0),
                                    'Best Drawdown (%)': top_result.get('max_drawdown_pct', 0),
                                    'Best Win Rate (%)': top_result.get('win_rate', 0),
                                    'Combinations': len(result['results'])
                                }
                            else:
                                row = {
                                    'Name': result['name'],
                                    'Type': 'Backtest',
                                    'Date': format_datetime(result['timestamp']),
                                    'Return (%)': result.get('return_pct', 0),
                                    'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
                                    'Win Rate (%)': result.get('win_rate', 0),
                                    'Trades': result.get('trades_count', 0)
                                }
                            
                            summary_data.append(row)
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            
                            if export_format == "CSV":
                                csv = summary_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name="backtest_summary_metrics.csv",
                                    mime="text/csv"
                                )
                            else:
                                json_data = summary_df.to_json(orient="records")
                                st.download_button(
                                    label="Download JSON",
                                    data=json_data,
                                    file_name="backtest_summary_metrics.json",
                                    mime="application/json"
                                )
                    
                    elif export_what == "Full Results":
                        # Export full results
                        if export_format == "JSON":
                            # Simplify results to make them JSON serializable
                            simplified_results = []
                            
                            for result in selected_results:
                                # Make a copy to avoid modifying original
                                serializable_result = {}
                                
                                # Copy basic fields
                                for key, value in result.items():
                                    if key == 'timestamp' and isinstance(value, (datetime, date)):
                                        serializable_result[key] = format_datetime(value)
                                    elif key == 'equity_curve' and isinstance(value, pd.DataFrame):
                                        # Convert DataFrame to dict
                                        serializable_result[key] = value.to_dict(orient="records")
                                    elif key == 'trades' and isinstance(value, pd.DataFrame):
                                        # Convert DataFrame to dict
                                        serializable_result[key] = value.to_dict(orient="records")
                                    else:
                                        serializable_result[key] = value
                                
                                simplified_results.append(serializable_result)
                            
                            import json
                            
                            json_data = json.dumps(simplified_results)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name="backtest_full_results.json",
                                mime="application/json"
                            )
                        else:
                            st.warning("Full results can only be exported as JSON due to the complex data structure.")
                    
                    elif export_what == "Trade Data":
                        # Export trade data
                        trade_dfs = []
                        
                        for result in selected_results:
                            if 'trades' in result and result['trades'] is not None:
                                # Convert dict-based trades to DataFrame if necessary
                                if isinstance(result['trades'], dict):
                                    trades_df = pd.DataFrame(result['trades'])
                                else:
                                    trades_df = result['trades']
                                
                                # Add result name as a column
                                trades_df['Result Name'] = result['name']
                                
                                trade_dfs.append(trades_df)
                        
                        if trade_dfs:
                            # Combine all trade DataFrames
                            combined_trades = pd.concat(trade_dfs, ignore_index=True)
                            
                            if export_format == "CSV":
                                csv = combined_trades.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name="backtest_trades.csv",
                                    mime="text/csv"
                                )
                            else:
                                json_data = combined_trades.to_json(orient="records")
                                st.download_button(
                                    label="Download JSON",
                                    data=json_data,
                                    file_name="backtest_trades.json",
                                    mime="application/json"
                                )
                        else:
                            st.warning("No trade data found in the selected results.")
            
            # Save/delete options in the second column
            with save_col:
                st.markdown("#### Manage Results")
                
                # Delete selected results
                if st.button("Delete Selected Results", type="primary", help="This will permanently delete the selected results"):
                    for result in selected_results:
                        if 'filepath' in result:
                            # Try to delete the file
                            try:
                                os.remove(result['filepath'])
                                st.success(f"Deleted: {result['name']}")
                            except Exception as e:
                                st.error(f"Error deleting {result['name']}: {str(e)}")
                    
                    # Clear selections and show message
                    for i in range(len(all_saved_results)):
                        st.session_state[f"selected_{i}"] = False
                    
                    st.success("Selected results have been deleted. Please refresh the page to update the list.")
                    st.button("Refresh Page", on_click=lambda: None)