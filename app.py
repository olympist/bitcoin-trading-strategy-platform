import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
from pathlib import Path
import glob

# Simplified API handling without direct CORS manipulation
# This approach will work across different Streamlit versions

# Create API endpoints
def setup_api_endpoints():
    
    # Get query parameters
    query_params = st.experimental_get_query_params()
    
    # Check if this is an API request
    if "api" in query_params:
        endpoint = query_params.get("api", [""])[0]
        
        # API endpoint: Get market data
        if endpoint == "market_data":
            symbol = query_params.get("symbol", ["BTCUSDT"])[0]
            try:
                from utils.binance_data import get_market_data
                market_data = get_market_data(symbol, "1h", use_real_data=True, use_historical_data=True)
                st.json(market_data)
                st.stop()
            except Exception as e:
                st.json({"error": str(e)})
                st.stop()
        
        # API endpoint: Get optimization results
        elif endpoint == "optimization_results":
            try:
                # Get optimization results from the results folder
                results_files = glob.glob("results/optimization/*.json")
                results_list = []
                
                for file in results_files:
                    try:
                        with open(file, "r") as f:
                            data = json.load(f)
                            # Extract basic metadata
                            meta = {
                                "id": os.path.basename(file).replace(".json", ""),
                                "strategy_name": data.get("strategy_name", "Unknown"),
                                "timestamp": data.get("timestamp", "Unknown"),
                                "dataset_name": data.get("dataset_name", "Unknown"),
                                "tests_count": len(data.get("results", [])),
                                "best_result": None
                            }
                            
                            # Find best result
                            results = data.get("results", [])
                            if results:
                                # Sort by return percentage
                                sorted_results = sorted(results, key=lambda x: x.get("return_pct", 0), reverse=True)
                                if sorted_results:
                                    meta["best_result"] = sorted_results[0]
                            
                            results_list.append(meta)
                    except Exception as e:
                        continue
                
                # Sort by timestamp (newest first)
                results_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                st.json(results_list)
                st.stop()
            except Exception as e:
                st.json({"error": str(e)})
                st.stop()
        
        # API endpoint: Get optimization detail
        elif endpoint == "optimization_detail":
            result_id = query_params.get("id", [""])[0]
            if result_id:
                try:
                    file_path = f"results/optimization/{result_id}.json"
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            st.json(data)
                            st.stop()
                    else:
                        st.json({"error": f"Result with ID {result_id} not found"})
                        st.stop()
                except Exception as e:
                    st.json({"error": str(e)})
                    st.stop()
            else:
                st.json({"error": "Missing result ID"})
                st.stop()
        
        # API endpoint: Run backtest
        elif endpoint == "run_backtest":
            try:
                # Get request body
                request_json = st.experimental_get_query_params().get("data", ["{}"])[0]
                request_data = json.loads(request_json)
                
                # Process backtest request
                # This is a placeholder - you would implement the actual backtest logic here
                response = {
                    "status": "success",
                    "message": "Backtest completed",
                    "results": {
                        "return_pct": 15.5,
                        "max_drawdown": 8.2,
                        "trades_count": 12
                    }
                }
                
                st.json(response)
                st.stop()
            except Exception as e:
                st.json({"error": str(e)})
                st.stop()
                
        # API endpoint: Health check
        elif endpoint == "health":
            st.json({"status": "ok"})
            st.stop()
            
        # Unknown API endpoint
        else:
            st.json({"error": f"Unknown API endpoint: {endpoint}"})
            st.stop()

# V8 strategy BotManager has been removed

# Set page configuration
st.set_page_config(
    page_title="Cryptocurrency Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup API endpoints first - this should run before any other Streamlit commands
setup_api_endpoints()

# Apply custom styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .strategy-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .strategy-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .strategy-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .strategy-description {
        color: #6c757d;
        margin-bottom: 15px;
    }
    .strategy-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("Cryptocurrency Trading Strategy Platform")
st.markdown("### Advanced Trading Strategy Backtesting & Deployment")

# Check for API keys
if 'api_key_checked' not in st.session_state:
    api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
    api_secret = os.environ.get('BINANCE_TESTNET_SECRET')
    
    if api_key and api_secret:
        st.session_state.has_api_keys = True
    else:
        st.session_state.has_api_keys = False
    
    st.session_state.api_key_checked = True

# Display API status
if not st.session_state.has_api_keys:
    st.warning("⚠️ Binance TestNet API keys not detected. Some functionality may be limited to simulation mode.")

# Strategy selection section
st.markdown("## Available Strategy Systems")

# Create a grid for main strategy cards
col1, col2 = st.columns(2)

with col1:
    # V7 Strategy Card
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-title">V7 Time-Optimized Strategy</div>
        <div class="strategy-description">
            Enhanced multi-bot system with precise time control. Configure custom start dates/times
            and operation intervals for each bot to optimize trading performance.
        </div>
        <a href="/v7_strategy_dashboard" target="_self">
            <button class="strategy-button">Launch V7 Dashboard</button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # V5 Strategy Card
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-title">V5 Strategy</div>
        <div class="strategy-description">
            Multi-bot capability with variable start dates and comprehensive risk management including
            fixed stop-loss, trailing stop, and time-based position closing.
        </div>
        <a href="/parameter_optimization_simple" target="_self">
            <button class="strategy-button">Launch V5 Dashboard</button>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Backtesting Card
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-title">Advanced Backtesting</div>
        <div class="strategy-description">
            Comprehensive backtesting tools with Monte Carlo simulation, optimization algorithms,
            and detailed performance metrics. Test strategies against historical data with precise controls.
        </div>
        <a href="/backtest" target="_self">
            <button class="strategy-button">Launch Backtester</button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto Optimization Card
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-title">Auto Optimization V2.0</div>
        <div class="strategy-description">
            Intelligent parameter optimization with automated batch generation, cross-dataset analysis,
            and progress tracking. Find optimal strategy parameters across market conditions.
        </div>
        <a href="/auto_optimization" target="_self">
            <button class="strategy-button">Launch Auto Optimizer</button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Quick stats section
st.markdown("## Market Overview")

try:
    # Use historical data by default
    from utils.binance_data import get_market_data
    market_data = get_market_data("BTCUSDT", "1h", use_real_data=True, use_historical_data=True)
    
    if market_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("BTC Price", f"${market_data['current_price']:,.2f}", 
                     f"{market_data['change_24h_percent']*100:.2f}%")
        
        with col2:
            st.metric("24h High", f"${market_data['high_24h']:,.2f}")
        
        with col3:
            st.metric("24h Low", f"${market_data['low_24h']:,.2f}")
        
        with col4:
            st.metric("Data Source", market_data.get('data_source', 'Unknown'))
        
        # Display a small chart using Plotly for better control
        if 'chart_data' in market_data and not market_data['chart_data'].empty:
            try:
                import plotly.graph_objects as go
                
                # Filter for non-infinite, non-null values to prevent warnings
                valid_chart_data = market_data['chart_data'].copy()
                valid_chart_data = valid_chart_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if not valid_chart_data.empty and 'time' in valid_chart_data.columns and 'price' in valid_chart_data.columns:
                    # Verify that columns have the correct data types
                    if not pd.api.types.is_datetime64_any_dtype(valid_chart_data['time']):
                        valid_chart_data['time'] = pd.to_datetime(valid_chart_data['time'])
                        
                    valid_chart_data['price'] = pd.to_numeric(valid_chart_data['price'], errors='coerce')
                    valid_chart_data = valid_chart_data.dropna()
                    
                    if not valid_chart_data.empty:
                        # Create a Plotly figure for more control
                        fig = go.Figure()
                        
                        # Add the price line
                        fig.add_trace(
                            go.Scatter(
                                x=valid_chart_data['time'],
                                y=valid_chart_data['price'],
                                mode='lines',
                                name='BTC Price',
                                line=dict(color='#1F4287', width=2)
                            )
                        )
                        
                        # Configure the layout
                        fig.update_layout(
                            title="Bitcoin Price Chart",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            template="plotly_white",
                            height=400,
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        # Explicitly set axis ranges
                        if len(valid_chart_data) > 1:
                            time_min = valid_chart_data['time'].min()
                            time_max = valid_chart_data['time'].max()
                            price_min = valid_chart_data['price'].min() * 0.99  # Add 1% padding
                            price_max = valid_chart_data['price'].max() * 1.01  # Add 1% padding
                            
                            fig.update_xaxes(range=[time_min, time_max])
                            fig.update_yaxes(range=[price_min, price_max])
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not display chart: Invalid data after cleaning")
                else:
                    st.info("Could not display chart: Missing required columns")
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
                st.info("Please check if the historical data files are available in the data folder.")
except Exception as e:
    st.error(f"Could not load market data: {str(e)}")
    st.info("Please check if the historical data files are available in the data folder.")

# Footer
st.markdown("---")
st.markdown("Developed by the Bitcoin Trading Strategy Platform Team | © 2025")