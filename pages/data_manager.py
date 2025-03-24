import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_fetcher import DataFetcher
from utils.data_validator import DataValidator

st.set_page_config(
    page_title="Data Manager - Bitcoin Backtesting Platform",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data_files' not in st.session_state:
    st.session_state.data_files = []
if 'selected_data' not in st.session_state:
    st.session_state.selected_data = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Initialize the data utilities
data_fetcher = DataFetcher()
data_validator = DataValidator()

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
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown('<p class="section-header">Data Management</p>', unsafe_allow_html=True)
st.markdown("Fetch, validate, and manage Bitcoin market data from multiple sources.")

# Create tabs for different data management functions
tab1, tab2, tab3 = st.tabs(["Fetch Data", "Manage Saved Data", "Validate & View Data"])

# Tab 1: Fetch Data
with tab1:
    st.markdown("### Fetch New Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        exchange = st.selectbox(
            "Exchange",
            ["binance", "binance_testnet", "bitstamp", "okx", "kraken", "bybit", "bitget"],
            index=0
        )
        
        # Display info about testnet if selected
        if exchange == "binance_testnet":
            st.info("Using Binance Testnet with configured API keys.")
        
        proxy = st.text_input(
            "Proxy URL (for Binance)",
            placeholder="http://your-proxy-url:port",
            help="Required if Binance is geo-restricted in your region",
            disabled=exchange != "binance"
        )
    
    with col2:
        symbol = st.selectbox(
            "Symbol",
            ["BTC/USDT", "BTC/USD", "BTC/USDC"],
            index=0
        )
        
        # Add option to choose between days slider or specific date range
        date_option = st.radio(
            "Date Selection Method",
            ["Use Days", "Use Date Range"],
            index=0,
            horizontal=True
        )
        
        if date_option == "Use Days":
            days = st.slider(
                "Days to Fetch",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of days of historical data to fetch"
            )
            # Set start_date and end_date as None in this case
            start_date = None
            end_date = None
        else:
            # Calculate default end date (today) and start date (30 days ago)
            end_date_default = datetime.now()
            start_date_default = end_date_default - timedelta(days=30)
            
            start_date = st.date_input(
                "Start Date",
                value=start_date_default,
                help="Beginning date for data fetching"
            )
            
            end_date = st.date_input(
                "End Date",
                value=end_date_default,
                help="Ending date for data fetching"
            )
            
            # Calculate days between dates for the API call
            if end_date >= start_date:
                days = (end_date - start_date).days + 1  # +1 to include end date
            else:
                st.warning("End date must be after start date. Using 1 day as minimum.")
                days = 1
    
    with col3:
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=4,
            help="Candle timeframe"
        )
        
        fetch_button = st.button("Fetch Data", use_container_width=True)
    
    # Progress and status indicators
    if fetch_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update status
            status_text.text("Initializing exchange connection...")
            progress_bar.progress(10)
            
            # Initialize exchange (with proxy if provided and selected exchange is Binance)
            current_proxy = proxy if exchange == "binance" and proxy else None
            data_fetcher.initialize_exchange(exchange, current_proxy)
            
            # Update status
            status_text.text(f"Fetching {timeframe} data for {symbol} from {exchange}...")
            progress_bar.progress(30)
            
            # Fetch the data with specific date range if provided
            df = data_fetcher.fetch_historical_data(
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                proxy=current_proxy,
                start_date=start_date if date_option == "Use Date Range" else None,
                end_date=end_date if date_option == "Use Date Range" else None
            )
            
            if df is None or df.empty:
                st.error(f"Failed to fetch data from {exchange}. Please check your connection and parameters.")
                progress_bar.progress(100)
            else:
                # Update status
                status_text.text("Validating data...")
                progress_bar.progress(60)
                
                # Validate the data
                is_valid, cleaned_df, validation_results = data_validator.validate_data(
                    df, timeframe, fix_issues=True
                )
                
                # Update status
                status_text.text("Saving data...")
                progress_bar.progress(80)
                
                # Save the data
                filepath = data_fetcher.save_data(cleaned_df, exchange, symbol, timeframe)
                
                # Refresh the list of data files
                st.session_state.data_files = data_fetcher.list_saved_data()
                
                # Update status
                status_text.text("Data fetched and saved successfully!")
                progress_bar.progress(100)
                
                # Display success message with data info
                st.success(f"Successfully fetched {len(cleaned_df)} {timeframe} candles for {symbol} from {exchange}")
                
                # Show validation results
                with st.expander("Data Validation Results", expanded=True):
                    for check, result in validation_results.items():
                        if result:
                            st.markdown(f"✅ **{check}**: Passed")
                        else:
                            st.markdown(f"⚠️ **{check}**: Failed (auto-fixed)")
                
                # Display sample of the data
                st.subheader("Sample Data")
                st.dataframe(cleaned_df.head(10))
                
                # Create and show a candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=cleaned_df.index,
                    open=cleaned_df['open'],
                    high=cleaned_df['high'],
                    low=cleaned_df['low'],
                    close=cleaned_df['close'],
                    increasing_line_color='#26a69a', 
                    decreasing_line_color='#ef5350'
                )])
                
                fig.update_layout(
                    title=f'{symbol} {timeframe} Price Data from {exchange}',
                    xaxis_title='Date',
                    yaxis_title='Price (USDT)',
                    height=400,
                    template='plotly_white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_bar.progress(100)
    
    # Multi-source data fetching
    with st.expander("Fetch from Multiple Sources", expanded=False):
        st.markdown("Fetch data from multiple exchanges for comparison and validation.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            exchanges = st.multiselect(
                "Exchanges",
                ["binance", "binance_testnet", "bitstamp", "okx", "kraken", "bybit", "bitget"],
                default=["binance", "binance_testnet", "bitstamp"]
            )
        
        with col2:
            multi_symbol = st.selectbox(
                "Symbol",
                ["BTC/USDT", "BTC/USD", "BTC/USDC"],
                index=0,
                key="multi_symbol"
            )
            
            multi_days = st.slider(
                "Days to Fetch",
                min_value=1,
                max_value=100,
                value=7,
                help="Number of days of historical data to fetch",
                key="multi_days"
            )
        
        with col3:
            multi_timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=4,
                help="Candle timeframe",
                key="multi_timeframe"
            )
            
            multi_proxy = st.text_input(
                "Proxy URL (for Binance)",
                placeholder="http://your-proxy-url:port",
                help="Required if Binance is geo-restricted in your region",
                key="multi_proxy"
            )
        
        multi_fetch_button = st.button("Fetch from Multiple Sources", use_container_width=True)
        
        if multi_fetch_button:
            # Create a progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update status
                status_text.text("Fetching data from multiple exchanges...")
                
                # Fetch data from multiple sources
                results = data_fetcher.fetch_from_multiple_sources(
                    symbol=multi_symbol,
                    timeframe=multi_timeframe,
                    days=multi_days,
                    exchanges=exchanges,
                    proxy=multi_proxy if multi_proxy else None
                )
                
                progress_bar.progress(50)
                
                if not results:
                    st.error("Failed to fetch data from any exchange.")
                    progress_bar.progress(100)
                else:
                    # Update status
                    status_text.text("Comparing data sources...")
                    
                    # Compare data sources
                    is_consistent, comparison_df = data_validator.compare_data_sources(results)
                    
                    progress_bar.progress(70)
                    
                    # Merge data sources with priority
                    status_text.text("Merging data from multiple sources...")
                    merged_df = data_validator.merge_data_sources(results, exchanges)
                    
                    progress_bar.progress(85)
                    
                    # Save the merged data
                    status_text.text("Saving merged data...")
                    filepath = data_fetcher.save_data(
                        merged_df, "multi_source", multi_symbol, multi_timeframe
                    )
                    
                    # Refresh the list of data files
                    st.session_state.data_files = data_fetcher.list_saved_data()
                    
                    progress_bar.progress(100)
                    status_text.text("Data fetched, merged, and saved successfully!")
                    
                    # Display success message
                    st.success(f"Successfully fetched and merged data from {len(results)} exchanges")
                    
                    # Show comparison results
                    with st.expander("Data Source Comparison", expanded=True):
                        if is_consistent:
                            st.markdown("✅ **Data Consistency**: Good consistency across sources")
                        else:
                            st.markdown("⚠️ **Data Consistency**: Significant differences detected between sources")
                        
                        # Show data counts from each source
                        st.markdown("#### Data points per source:")
                        for exchange, df in results.items():
                            st.markdown(f"- **{exchange}**: {len(df)} candles")
                        
                        # Plot comparison
                        st.markdown("#### Price comparison across sources:")
                        
                        fig = go.Figure()
                        for exchange, df in results.items():
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['close'],
                                mode='lines',
                                name=exchange
                            ))
                        
                        fig.update_layout(
                            title=f'{multi_symbol} Close Price Comparison',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=400,
                            template='plotly_white',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show merged data
                    st.subheader("Merged Data Sample")
                    st.dataframe(merged_df.head(10))
                    
                    # Show candlestick chart of merged data
                    fig = go.Figure(data=[go.Candlestick(
                        x=merged_df.index,
                        open=merged_df['open'],
                        high=merged_df['high'],
                        low=merged_df['low'],
                        close=merged_df['close'],
                        increasing_line_color='#26a69a', 
                        decreasing_line_color='#ef5350'
                    )])
                    
                    fig.update_layout(
                        title=f'Merged {multi_symbol} {multi_timeframe} Price Data',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        height=400,
                        template='plotly_white',
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                progress_bar.progress(100)

# Tab 2: Manage Saved Data
with tab2:
    st.markdown("### Manage Saved Data Files")
    
    # Refresh the list of data files
    if st.button("Refresh Data Files"):
        st.session_state.data_files = data_fetcher.list_saved_data()
    
    # If data files list is empty, load it
    if not st.session_state.data_files:
        st.session_state.data_files = data_fetcher.list_saved_data()
    
    # Check if we have data files
    if not st.session_state.data_files:
        st.info("No data files found. Use the 'Fetch Data' tab to download historical data.")
    else:
        # Create a dataframe for display
        df_files = pd.DataFrame(st.session_state.data_files)
        
        # Create a filtered view
        col1, col2, col3 = st.columns(3)
        
        with col1:
            exchange_filter = st.multiselect(
                "Filter by Exchange",
                options=sorted(df_files['exchange'].unique()),
                default=None
            )
        
        with col2:
            symbol_filter = st.multiselect(
                "Filter by Symbol",
                options=sorted(df_files['symbol'].unique()),
                default=None
            )
        
        with col3:
            timeframe_filter = st.multiselect(
                "Filter by Timeframe",
                options=sorted(df_files['timeframe'].unique()),
                default=None
            )
        
        # Apply filters
        filtered_df = df_files.copy()
        
        if exchange_filter:
            filtered_df = filtered_df[filtered_df['exchange'].isin(exchange_filter)]
        
        if symbol_filter:
            filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]
        
        if timeframe_filter:
            filtered_df = filtered_df[filtered_df['timeframe'].isin(timeframe_filter)]
        
        # Display the filtered dataframe
        if filtered_df.empty:
            st.info("No data files match the selected filters.")
        else:
            # Sort by creation date (newest first)
            filtered_df = filtered_df.sort_values('created', ascending=False)
            
            # Select columns to display
            display_df = filtered_df[['filename', 'exchange', 'symbol', 'timeframe', 'created', 'size_mb', 'rows']]
            
            # Display the dataframe
            st.dataframe(display_df, use_container_width=True)
            
            # Functionality to delete files
            with st.expander("Delete Data Files", expanded=False):
                st.warning("Warning: Deleting files is permanent and cannot be undone.")
                
                file_to_delete = st.selectbox(
                    "Select file to delete",
                    options=filtered_df['filename'].tolist()
                )
                
                if st.button("Delete Selected File"):
                    try:
                        # Get the filepath
                        filepath = filtered_df[filtered_df['filename'] == file_to_delete]['filepath'].iloc[0]
                        
                        # Delete the file
                        os.remove(filepath)
                        
                        # Refresh the list
                        st.session_state.data_files = data_fetcher.list_saved_data()
                        
                        # Update the filtered dataframe
                        filtered_df = filtered_df[filtered_df['filename'] != file_to_delete]
                        
                        st.success(f"File {file_to_delete} deleted successfully.")
                        
                        # Force a rerun to refresh the UI
                        st.experimental_rerun()
                    
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")

# Tab 3: Validate & View Data
with tab3:
    st.markdown("### View and Validate Data")
    
    # If data files list is empty, load it
    if not st.session_state.data_files:
        st.session_state.data_files = data_fetcher.list_saved_data()
    
    # Check if we have data files
    if not st.session_state.data_files:
        st.info("No data files found. Use the 'Fetch Data' tab to download historical data.")
    else:
        # Create a dataframe for selection
        df_files = pd.DataFrame(st.session_state.data_files)
        
        # File selection
        selected_file = st.selectbox(
            "Select Data File to View",
            options=df_files['filename'].tolist(),
            format_func=lambda x: f"{x} - {df_files.loc[df_files['filename'] == x, 'created'].iloc[0]}"
        )
        
        # Get the filepath
        filepath = df_files[df_files['filename'] == selected_file]['filepath'].iloc[0]
        
        # Load the data
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
            
            # Display file info
            file_info = df_files[df_files['filename'] == selected_file].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{file_info["exchange"]}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Exchange</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{file_info["symbol"]}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Symbol</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{file_info["timeframe"]}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Timeframe</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{len(df)}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Data Points</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Data validation
            validate_tab, explore_tab = st.tabs(["Validate Data", "Explore Data"])
            
            with validate_tab:
                st.markdown("#### Data Validation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fix_issues = st.checkbox("Fix issues automatically", value=True)
                
                with col2:
                    outlier_threshold = st.slider(
                        "Outlier Z-score threshold",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.1
                    )
                
                if st.button("Run Data Validation"):
                    # Run validation
                    is_valid, cleaned_df, validation_results = data_validator.validate_data(
                        df, file_info["timeframe"], outlier_threshold, fix_issues
                    )
                    
                    # Display validation results
                    if is_valid:
                        st.success("✅ Data validation passed!")
                    else:
                        st.warning("⚠️ Data validation found issues.")
                    
                    # Show detailed results
                    st.markdown("##### Validation Checks:")
                    for check, result in validation_results.items():
                        if result:
                            st.markdown(f"✅ **{check}**: Passed")
                        else:
                            st.markdown(f"❌ **{check}**: Failed")
                    
                    # If issues were fixed
                    if fix_issues and not is_valid:
                        # Compare original and cleaned data
                        st.markdown("##### Data Cleaning Results:")
                        
                        # Check for removed/interpolated values
                        diff_count = (df != cleaned_df).sum().sum()
                        
                        if diff_count > 0:
                            st.info(f"Fixed {diff_count} data points through cleaning.")
                            
                            # Option to save cleaned data
                            if st.button("Save Cleaned Data"):
                                # Generate cleaned filename
                                parts = selected_file.split(".")
                                cleaned_filename = f"{parts[0]}_cleaned.{parts[1]}"
                                cleaned_filepath = os.path.join(os.path.dirname(filepath), cleaned_filename)
                                
                                # Save cleaned data
                                cleaned_df.to_csv(cleaned_filepath)
                                
                                # Refresh the list
                                st.session_state.data_files = data_fetcher.list_saved_data()
                                
                                st.success(f"Cleaned data saved as {cleaned_filename}")
                        else:
                            st.info("No data points needed cleaning.")
            
            with explore_tab:
                st.markdown("#### Data Explorer")
                
                # Date range filter
                date_min = df.index.min().date()
                date_max = df.index.max().date()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=date_min,
                        min_value=date_min,
                        max_value=date_max
                    )
                
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=date_max,
                        min_value=date_min,
                        max_value=date_max
                    )
                
                # Filter data by date range
                mask = (df.index.date >= start_date) & (df.index.date <= end_date)
                filtered_df = df.loc[mask]
                
                # Display data stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", len(filtered_df))
                
                with col2:
                    date_range = (filtered_df.index.max() - filtered_df.index.min()).days
                    st.metric("Date Range", f"{date_range} days")
                
                with col3:
                    price_min = filtered_df['low'].min()
                    price_max = filtered_df['high'].max()
                    st.metric("Price Range", f"${price_min:.2f} - ${price_max:.2f}")
                
                with col4:
                    avg_volume = filtered_df['volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:.2f}")
                
                # Display the data table
                with st.expander("Data Table", expanded=False):
                    st.dataframe(filtered_df, use_container_width=True)
                
                # Create and show a candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=filtered_df.index,
                    open=filtered_df['open'],
                    high=filtered_df['high'],
                    low=filtered_df['low'],
                    close=filtered_df['close'],
                    increasing_line_color='#26a69a', 
                    decreasing_line_color='#ef5350'
                )])
                
                fig.update_layout(
                    title=f'{file_info["symbol"]} {file_info["timeframe"]} Price Data',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=500,
                    template='plotly_white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create and show a volume chart
                volume_fig = go.Figure(data=[go.Bar(
                    x=filtered_df.index,
                    y=filtered_df['volume'],
                    marker_color='#2196f3'
                )])
                
                volume_fig.update_layout(
                    title=f'{file_info["symbol"]} {file_info["timeframe"]} Volume',
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    height=300,
                    template='plotly_white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
