import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import uuid
import hashlib
import itertools
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import strategy modules
from strategy.optimized_alternative_dca_v5 import OptimizedAlternativeDCAv5
from backtesting.backtester import Backtester
from utils.data_fetcher import DataFetcher
from utils.results_manager import ResultsManager
from utils.optimization_strategies import (
    generate_exploration_batches,
    generate_conservative_batches,
    generate_aggressive_batches,
    generate_balanced_batches,
    generate_diversification_batches,
    generate_genetic_algorithm_batches
)
from visualization.performance_plots import PerformancePlots

# Initialize components
data_fetcher = DataFetcher()
results_manager = ResultsManager()

# Page title and layout
st.set_page_config(
    page_title="Auto Optimization",
    page_icon="🤖",
    layout="wide"
)

def select_optimization_mode(dataset_name):
    """
    UI for selecting the optimization mode and strategy.
    
    Args:
        dataset_name: Name of the dataset being optimized
        
    Returns:
        Dictionary with optimization mode configuration
    """
    st.subheader("Optimization Strategy")
    
    # Strategy selection
    strategy_options = [
        "Exploration (Initial Testing)",
        "Conservative (Risk Management)",
        "Aggressive (Maximum Returns)",
        "Balanced (Risk vs Return)",
        "Diversification (Wide Coverage)",
        "Genetic Algorithm (Adaptive)",
        "Custom (Manual Configuration)"
    ]
    
    selected_strategy = st.selectbox(
        "Select optimization strategy",
        options=strategy_options,
        index=0,
        help="Different strategies prioritize different aspects of trading performance"
    )
    
    # Strategy description
    strategy_descriptions = {
        "Exploration (Initial Testing)": "Tests wide parameter ranges to establish baseline performance. Good for initial optimization.",
        "Conservative (Risk Management)": "Prioritizes drawdown control and consistent returns over maximum profit.",
        "Aggressive (Maximum Returns)": "Focuses on maximizing return potential, potentially with higher drawdowns.",
        "Balanced (Risk vs Return)": "Seeks optimal trade-off between returns and risk measures.",
        "Diversification (Wide Coverage)": "Tests diverse parameter combinations to explore full parameter space.",
        "Genetic Algorithm (Adaptive)": "Uses previous results to evolve parameter combinations, focusing on promising areas.",
        "Custom (Manual Configuration)": "Manually configure parameter ranges and optimization settings."
    }
    
    st.info(strategy_descriptions[selected_strategy])
    
    # Additional configuration options based on strategy
    config = {"mode": selected_strategy}
    
    if selected_strategy == "Genetic Algorithm (Adaptive)":
        col1, col2 = st.columns(2)
        with col1:
            config["population_size"] = st.slider(
                "Population Size", 
                min_value=20, 
                max_value=100, 
                value=50,
                help="Number of parameter combinations to generate in each batch"
            )
        with col2:
            config["mutation_rate"] = st.slider(
                "Mutation Rate", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.2,
                help="Probability of random parameter mutations (0-0.5)"
            )
    
    # Show optimization priority weights
    st.subheader("Optimization Priorities")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        config["return_weight"] = st.slider(
            "Return Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.4,
            help="Importance of overall return percentage"
        )
    with col2:
        config["drawdown_weight"] = st.slider(
            "Drawdown Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2,
            help="Importance of minimizing drawdowns"
        )
    with col3:
        config["win_rate_weight"] = st.slider(
            "Win Rate Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2,
            help="Importance of trade win rate"
        )
    with col4:
        config["sharpe_weight"] = st.slider(
            "Sharpe Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2,
            help="Importance of risk-adjusted returns"
        )
    
    # Data tracking
    config["dataset"] = dataset_name
    config["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return config

def analyze_previous_results(flat_results, param_name, top_n=3, min_sample_size=10):
    """
    Analyze previous optimization results to find optimal parameter ranges with advanced
    statistical analysis and edge case handling.
    
    Args:
        flat_results: List of result dictionaries from previous batches
        param_name: Parameter name to analyze
        top_n: Number of top results to consider
        min_sample_size: Minimum required samples for statistical analysis
        
    Returns:
        min_val, max_val, optimal_values, confidence_score - a tuple containing min/max, 
        top values, and a confidence score (0-1) indicating reliability of the analysis
    """
    try:
        # Sort results by return percentage (primary metric)
        sorted_results = sorted(flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        
        # Extract values for this parameter
        param_values = [res['parameters'][param_name] for res in sorted_results if param_name in res['parameters']]
        
        if not param_values:
            return None, None, [], 0.0
        
        # Find min and max values
        min_val = min(param_values)
        max_val = max(param_values)
        
        # Get top N values (or fewer if not enough results)
        top_results = sorted_results[:min(top_n, len(sorted_results))]
        optimal_values = [res['parameters'][param_name] for res in top_results 
                          if param_name in res['parameters']]
        
        # Get all values and associated performance
        value_performance = []
        for res in sorted_results:
            if param_name in res['parameters']:
                value_performance.append({
                    'value': res['parameters'][param_name],
                    'return': res['metrics']['return_pct'],
                    'sharpe': res['metrics'].get('sharpe_ratio', 0),
                    'max_dd': res['metrics'].get('max_drawdown_pct', 0)
                })
        
        # Group by parameter value and calculate average performance
        value_groups = {}
        for item in value_performance:
            val = item['value']
            if val not in value_groups:
                value_groups[val] = {'count': 0, 'return': 0, 'sharpe': 0, 'max_dd': 0}
            value_groups[val]['count'] += 1
            value_groups[val]['return'] += item['return']
            value_groups[val]['sharpe'] += item['sharpe'] 
            value_groups[val]['max_dd'] += item['max_dd']
        
        # Calculate averages
        for val, metrics in value_groups.items():
            count = metrics['count']
            if count > 0:
                metrics['return'] /= count
                metrics['sharpe'] /= count
                metrics['max_dd'] /= count
        
        # Sort values by average return
        sorted_values = sorted(value_groups.items(), key=lambda x: x[1]['return'], reverse=True)
        
        # Check the distribution of top performers
        top_values_set = set(optimal_values)
        
        # Check if the best values are at the edges of the search space
        at_edges = min_val in top_values_set or max_val in top_values_set
        
        # Calculate confidence score based on:
        # 1. Sample size 
        # 2. Variance in performance
        # 3. Distribution of top values (are they clustered or spread out?)
        
        # Sample size confidence (scales with sqrt of sample size up to min_sample_size)
        sample_size_conf = min(1.0, (len(param_values) / min_sample_size) ** 0.5)
        
        # Performance spread confidence
        returns = [item['return'] for item in value_performance]
        performance_spread = max(returns) - min(returns) if returns else 0
        spread_conf = min(1.0, performance_spread / 10.0)  # Scale by expected spread of 10%
        
        # Cluster confidence (are top values close to each other?)
        if len(optimal_values) >= 3:
            sorted_top = sorted(optimal_values)
            range_all = max_val - min_val if max_val != min_val else 1.0
            range_top = sorted_top[-1] - sorted_top[0] if sorted_top[-1] != sorted_top[0] else range_all * 0.1
            cluster_conf = 1.0 - (range_top / range_all)
        else:
            cluster_conf = 0.5
        
        # Overall confidence score 
        confidence_score = (2 * sample_size_conf + spread_conf + cluster_conf) / 4
        
        # Handle potential edge cases (all values are the same)
        if len(set(optimal_values)) == 1 and len(param_values) > 1:
            # Add some variation by including the second best set of parameters
            more_results = sorted_results[top_n:min(top_n*2, len(sorted_results))]
            more_values = [res['parameters'][param_name] for res in more_results 
                           if param_name in res['parameters']]
            optimal_values.extend(more_values)
        
        # Remove duplicates while preserving order
        seen = set()
        optimal_values = [x for x in optimal_values if not (x in seen or seen.add(x))]
        
        # Expand the range if we're at the edges or have low confidence
        if at_edges or confidence_score < 0.4:
            # Extend search space by 20% in appropriate direction
            param_range = max_val - min_val
            if min_val in top_values_set:
                min_val = max(0, min_val - 0.2 * param_range)
            if max_val in top_values_set:
                max_val = max_val + 0.2 * param_range
        
        return min_val, max_val, optimal_values, confidence_score
    
    except Exception as e:
        st.warning(f"Error analyzing parameter {param_name}: {str(e)}")
        return None, None, [], 0.0

def generate_auto_batch_config(batch_iteration, price_data, dataset_name, previous_results=None):
    """
    Automatically generate batch configurations based on iteration number
    
    Args:
        batch_iteration: Current batch iteration (0 = first run)
        price_data: DataFrame with OHLCV data
        dataset_name: Name of the dataset being tested
        previous_results: Dictionary of previous batch results
        
    Returns:
        List of batch configuration dictionaries
    """
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
    
    # Create a deterministic ID based on dataset name for batch naming
    # Hash the dataset name to create a consistent identifier
    dataset_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
    
    # Generate batches based on iteration
    batches = []
    
    # First iteration (0) - test all parameters with broad ranges to establish baseline
    if batch_iteration == 0:
        # First batch: Test all 13 parameters with broad ranges
        batch1_optimize = {
            'investment_pct': [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0],  # Added edge values 1.0 and 100.0
            'price_drop_threshold': [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0],  # Added edge values 0.5 and 15.0
            'profit_threshold': [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0],  # Added edge values 0.5 and 15.0
            'stop_loss_fixed_pct': [5.0, 10.0, 15.0, 20.0, 25.0, 50.0],  # Added edge values 5.0 and 50.0
            'trailing_stop_pct': [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],  # Added edge values 0.2 and 5.0
            'max_position_duration': [30, 90, 180, 270, 360, 1000],  # Added edge values 30 and 1000
            'number_of_bots': [1, 2, 3, 5, 8]  # Added edge value 8
        }
        batches.append({
            'name': f"Auto_Batch1_{dataset_hash}_iter{batch_iteration}",
            'optimize': batch1_optimize,
            'fixed': {k: v for k, v in default_params.items() if k not in batch1_optimize},
            'combinations': 5000
        })
        
        # Second batch: Test boolean parameters and additional parameters
        batch2_optimize = {
            'partial_profit_taking': [True, False],
            'use_trailing_stop_main': [True, False],
            'min_trade_size': [1.0, 5.0, 10.0, 20.0, 50.0, 100.0],  # Added edge values 1.0 and 100.0
            'profit_taking_percentage': [10.0, 25.0, 50.0, 75.0, 90.0],  # Added edge values 10.0 and 90.0
            'trailing_stop_activation_pct': [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],  # Added edge values 0.2 and 5.0
            'trading_fee_pct': [0.05, 0.1, 0.2, 0.5, 1.0]  # Added edge value 1.0
        }
        batches.append({
            'name': f"Auto_Batch2_{dataset_hash}_iter{batch_iteration}",
            'optimize': batch2_optimize,
            'fixed': {k: v for k, v in default_params.items() if k not in batch2_optimize},
            'combinations': 5000
        })
        
        # Third batch: Test extreme combinations of core parameters
        batch3_optimize = {
            'investment_pct': [1.0, 10.0, 50.0, 100.0],
            'price_drop_threshold': [0.5, 2.0, 8.0, 15.0],
            'profit_threshold': [0.5, 3.0, 8.0, 15.0],
            'stop_loss_fixed_pct': [5.0, 15.0, 30.0, 50.0],
            'max_position_duration': [30, 180, 500, 1000]
        }
        batches.append({
            'name': f"Auto_Batch3_{dataset_hash}_iter{batch_iteration}",
            'optimize': batch3_optimize,
            'fixed': {k: v for k, v in default_params.items() if k not in batch3_optimize},
            'combinations': 5000
        })
    
    # Second iteration (1) - focus on the most promising parameters from first iteration
    elif batch_iteration == 1 and previous_results:
        # Process all previous results to find the most promising parameter ranges
        all_flat_results = []
        for batch_id, batch_results in previous_results.items():
            all_flat_results.extend(batch_results)
        
        # Get top overall parameters
        sorted_results = sorted(all_flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        if sorted_results:
            best_params = sorted_results[0]['parameters']
            
            # Create refined ranges based on top parameters
            # Group 1: Core trading parameters
            batch1_optimize = {}
            for param in ['investment_pct', 'price_drop_threshold', 'profit_threshold']:
                if param in best_params:
                    min_val, max_val, optimal_values, confidence = analyze_previous_results(all_flat_results, param, top_n=5)
                    
                    # Display confidence score in the UI
                    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"Parameter **{param}**: Confidence score <span style='color:{confidence_color};font-weight:bold'>{confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    if min_val is not None and max_val is not None:
                        # Adjust search strategy based on confidence score
                        if confidence > 0.7:
                            # High confidence - narrow search around optimal values
                            if isinstance(best_params[param], int):
                                # For integer parameters
                                optimal_val = best_params[param]
                                refined_values = [optimal_val]
                                
                                # Add only small variations for high confidence
                                for val in optimal_values:
                                    if val not in refined_values:
                                        refined_values.append(int(val))
                                
                                # Add ±1 for fine-tuning
                                refined_values.extend([max(1, optimal_val-1), optimal_val+1])
                                
                                # Remove duplicates and sort
                                batch1_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # For float parameters - narrow search
                                optimal_val = best_params[param]
                                # Test values at 10% intervals around optimal (narrower search)
                                variation = abs(optimal_val * 0.1)
                                refined_values = [round(optimal_val - variation, 2),
                                                optimal_val,
                                                round(optimal_val + variation, 2)]
                                
                                # Add only top values
                                for val in optimal_values[:3]:  # Only use top 3 values
                                    if val not in refined_values:
                                        refined_values.append(round(val, 2))
                                        
                                batch1_optimize[param] = sorted(refined_values)
                                
                        elif confidence > 0.4:
                            # Medium confidence - moderate search
                            if isinstance(best_params[param], int):
                                # Create a refined range around the optimal value
                                refined_values = [max(1, int(v-1)) for v in optimal_values] + \
                                               [int(v) for v in optimal_values] + \
                                               [int(v+1) for v in optimal_values]
                                # Remove duplicates and sort
                                batch1_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Create a refined range with standard granularity
                                optimal_val = best_params[param]
                                # Test values at 20% intervals around optimal
                                variation = abs(optimal_val * 0.2)
                                refined_values = [round(optimal_val - variation, 2),
                                                optimal_val,
                                                round(optimal_val + variation, 2)]
                                for val in optimal_values:
                                    if val not in refined_values:
                                        refined_values.append(round(val, 2))
                                batch1_optimize[param] = sorted(refined_values)
                        else:
                            # Low confidence - broader search
                            if isinstance(best_params[param], int):
                                # Create a wider range for exploration
                                optimal_val = best_params[param]
                                param_range = max_val - min_val
                                # Ensure we test a broader range
                                refined_values = [
                                    max(1, int(min_val - 0.1 * param_range)),
                                    max(1, int(min_val)),
                                    int(optimal_val),
                                    int(max_val),
                                    int(max_val + 0.1 * param_range)
                                ]
                                # Add specific values that worked well
                                for val in optimal_values:
                                    refined_values.append(int(val))
                                
                                # Remove duplicates and sort
                                batch1_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Wider range for float parameters
                                optimal_val = best_params[param]
                                param_range = max_val - min_val
                                # Test edge values and more points in between
                                refined_values = [
                                    round(max(0, min_val - 0.1 * param_range), 2),
                                    round(min_val, 2),
                                    round(min_val + 0.25 * param_range, 2),
                                    round(optimal_val, 2),
                                    round(min_val + 0.75 * param_range, 2),
                                    round(max_val, 2),
                                    round(max_val + 0.1 * param_range, 2)
                                ]
                                # Add top values
                                for val in optimal_values:
                                    refined_values.append(round(val, 2))
                                    
                                batch1_optimize[param] = sorted(list(set(refined_values)))
                    else:
                        # Use default values if no clear optimal range found
                        batch1_optimize[param] = [best_params.get(param, default_params[param])]
            
            batches.append({
                'name': f"Refined_Core_{dataset_hash}_iter{batch_iteration}",
                'optimize': batch1_optimize,
                'fixed': {k: best_params.get(k, v) for k, v in default_params.items() if k not in batch1_optimize},
                'combinations': 5000
            })
            
            # Group 2: Risk management parameters
            batch2_optimize = {}
            for param in ['stop_loss_fixed_pct', 'trailing_stop_pct', 'max_position_duration']:
                if param in best_params:
                    min_val, max_val, optimal_values, confidence = analyze_previous_results(all_flat_results, param, top_n=5)
                    
                    # Display confidence score in the UI
                    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"Parameter **{param}**: Confidence score <span style='color:{confidence_color};font-weight:bold'>{confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    if min_val is not None and max_val is not None:
                        # Adjust search strategy based on confidence score
                        if confidence > 0.7:
                            # High confidence - narrow search around optimal values
                            if param == 'max_position_duration':
                                # Integer parameter with smaller steps for high confidence
                                refined_values = [int(v) for v in optimal_values]
                                # Add minimal variations
                                for val in optimal_values:
                                    delta = max(1, int(val * 0.05))  # 5% variation
                                    refined_values.extend([max(30, int(val - delta)), int(val + delta)])
                                batch2_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Float parameters - narrow search
                                refined_values = [round(v, 2) for v in optimal_values]
                                # Add minimal variations
                                for val in optimal_values:
                                    delta = val * 0.05  # 5% variation
                                    refined_values.extend([round(max(0.1, val - delta), 2), round(val + delta, 2)])
                                batch2_optimize[param] = sorted(list(set(refined_values)))
                                
                        elif confidence > 0.4:
                            # Medium confidence - standard search
                            if param == 'max_position_duration':
                                # Integer parameter with moderate steps
                                refined_values = [int(v) for v in optimal_values]
                                # Add variations
                                for val in optimal_values:
                                    delta = int(val * 0.1)  # 10% variation
                                    refined_values.extend([max(30, int(val - delta)), int(val + delta)])
                                batch2_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Float parameters
                                refined_values = [round(v, 2) for v in optimal_values]
                                # Add variations
                                for val in optimal_values:
                                    delta = val * 0.1  # 10% variation
                                    refined_values.extend([round(max(0.1, val - delta), 2), round(val + delta, 2)])
                                batch2_optimize[param] = sorted(list(set(refined_values)))
                        else:
                            # Low confidence - wider search
                            if param == 'max_position_duration':
                                # Integer parameter with larger steps
                                param_range = max_val - min_val
                                refined_values = [
                                    max(30, int(min_val)),
                                    max(30, int(min_val + param_range * 0.25)),
                                    max(30, int((min_val + max_val) / 2)),
                                    int(max_val - param_range * 0.25),
                                    int(max_val)
                                ]
                                # Add optimal values
                                for val in optimal_values:
                                    refined_values.append(int(val))
                                batch2_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Float parameters with wider range
                                param_range = max_val - min_val
                                refined_values = [
                                    round(min_val, 2),
                                    round(min_val + param_range * 0.25, 2),
                                    round((min_val + max_val) / 2, 2),
                                    round(max_val - param_range * 0.25, 2),
                                    round(max_val, 2)
                                ]
                                # Add optimal values
                                for val in optimal_values:
                                    refined_values.append(round(val, 2))
                                batch2_optimize[param] = sorted(list(set(refined_values)))
                    else:
                        # Use default values if no clear optimal range found
                        batch2_optimize[param] = [best_params.get(param, default_params[param])]
            
            batches.append({
                'name': f"Refined_Risk_{dataset_hash}_iter{batch_iteration}",
                'optimize': batch2_optimize,
                'fixed': {k: best_params.get(k, v) for k, v in default_params.items() if k not in batch2_optimize},
                'combinations': 5000
            })
            
            # Group 3: Advanced parameters
            batch3_optimize = {}
            for param in ['number_of_bots', 'partial_profit_taking', 'min_trade_size', 
                          'profit_taking_percentage', 'trailing_stop_activation_pct']:
                if param in best_params:
                    min_val, max_val, optimal_values, confidence = analyze_previous_results(all_flat_results, param, top_n=5)
                    
                    # Display confidence score in the UI
                    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"Parameter **{param}**: Confidence score <span style='color:{confidence_color};font-weight:bold'>{confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    if min_val is not None and max_val is not None:
                        # Special handling for boolean parameter
                        if param == 'partial_profit_taking':
                            # Boolean parameter - always test both values
                            batch3_optimize[param] = [True, False]
                            continue
                            
                        # Adjust search strategy based on confidence score
                        if confidence > 0.7:
                            # High confidence - narrow search around optimal values
                            if param == 'number_of_bots':
                                # Integer parameter with minimal variation
                                refined_values = [int(v) for v in optimal_values]
                                # Only add ±1 for the top value 
                                top_val = int(optimal_values[0]) if optimal_values else 1
                                refined_values.extend([max(1, top_val - 1), top_val + 1])
                                batch3_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Float parameters - narrow search
                                optimal_val = optimal_values[0] if optimal_values else min_val
                                delta = optimal_val * 0.05  # 5% variation for high confidence
                                refined_values = [
                                    round(max(0.1, optimal_val - delta), 2),
                                    round(optimal_val, 2),
                                    round(optimal_val + delta, 2)
                                ]
                                # Add a few top values for diversity
                                for val in optimal_values[:2]:  # Only top 2 values
                                    refined_values.append(round(val, 2))
                                batch3_optimize[param] = sorted(list(set(refined_values)))
                                
                        elif confidence > 0.4:
                            # Medium confidence - moderate search
                            if param == 'number_of_bots':
                                # Integer parameter with moderate variation
                                refined_values = [int(v) for v in optimal_values]
                                # Add adjacent values to all optimal values
                                for val in optimal_values:
                                    refined_values.extend([max(1, int(val - 1)), int(val + 1)])
                                batch3_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Float parameters with moderate search
                                refined_values = [round(v, 2) for v in optimal_values]
                                # Add 10% variations
                                for val in optimal_values:
                                    delta = val * 0.1  # 10% variation
                                    refined_values.extend([round(max(0.1, val - delta), 2), round(val + delta, 2)])
                                batch3_optimize[param] = sorted(list(set(refined_values)))
                        else:
                            # Low confidence - wider search
                            if param == 'number_of_bots':
                                # Integer parameter with wider range
                                param_range = max_val - min_val
                                # Test a broader range
                                refined_values = [
                                    max(1, int(min_val)),
                                    max(1, int((min_val + max_val) / 2)),
                                    int(max_val)
                                ]
                                # Add optimal values from previous runs
                                for val in optimal_values:
                                    refined_values.append(int(val))
                                    # Add wider variations
                                    refined_values.extend([
                                        max(1, int(val - 2)),
                                        max(1, int(val - 1)),
                                        int(val + 1),
                                        int(val + 2)
                                    ])
                                batch3_optimize[param] = sorted(list(set(refined_values)))
                            else:
                                # Float parameters with wider range
                                param_range = max_val - min_val
                                # Test edge values and more points in between
                                refined_values = [
                                    round(min_val, 2),
                                    round((min_val + max_val) / 2, 2),
                                    round(max_val, 2)
                                ]
                                # Add optimal values
                                for val in optimal_values:
                                    refined_values.append(round(val, 2))
                                    # Add wider variations (20%)
                                    delta = val * 0.2
                                    refined_values.extend([
                                        round(max(0.1, val - delta), 2),
                                        round(val + delta, 2)
                                    ])
                                batch3_optimize[param] = sorted(list(set(refined_values)))
                    else:
                        # Use default values if no clear optimal range found
                        batch3_optimize[param] = [best_params.get(param, default_params[param])]
            
            batches.append({
                'name': f"Refined_Advanced_{dataset_hash}_iter{batch_iteration}",
                'optimize': batch3_optimize,
                'fixed': {k: best_params.get(k, v) for k, v in default_params.items() if k not in batch3_optimize},
                'combinations': 5000
            })
    
    # Third iteration and beyond - ultra-fine tuning of parameters
    elif batch_iteration >= 2 and previous_results:
        # Process all previous results to find the most promising parameter ranges
        all_flat_results = []
        for batch_id, batch_results in previous_results.items():
            all_flat_results.extend(batch_results)
        
        # Get top overall parameters
        sorted_results = sorted(all_flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        if sorted_results:
            best_params = sorted_results[0]['parameters']
            
            # Calculate parameter importance based on correlation with return percentage
            param_importance = {}
            param_confidence = {}  # Store confidence scores for each parameter
            
            # Display header for parameter importance section
            st.subheader("Parameter Importance Analysis for Ultra-Fine Tuning")
            
            for param in best_params.keys():
                if param in ['partial_profit_taking', 'use_trailing_stop_main']:
                    continue  # Skip boolean parameters for correlation analysis
                
                # Extract parameter values and returns for correlation analysis
                param_data = []
                return_data = []
                for result in all_flat_results:
                    if param in result['parameters'] and isinstance(result['parameters'][param], (int, float)):
                        param_data.append(result['parameters'][param])
                        return_data.append(result['metrics']['return_pct'])
                
                # Calculate confidence score based on data distribution and correlation strength
                _, _, _, confidence = analyze_previous_results(all_flat_results, param, top_n=5)
                param_confidence[param] = confidence
                
                if len(param_data) > 5:  # Need at least a few data points
                    try:
                        correlation = abs(np.corrcoef(param_data, return_data)[0, 1])
                        # Adjust importance by confidence - parameters with higher confidence get a boost
                        adjusted_importance = correlation * (0.5 + 0.5 * confidence)  # Scale importance by confidence
                        param_importance[param] = adjusted_importance if not np.isnan(adjusted_importance) else 0
                        
                        # Display importance and confidence information
                        importance_color = "green" if adjusted_importance > 0.5 else "orange" if adjusted_importance > 0.2 else "gray"
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                        
                        st.markdown(
                            f"Parameter **{param}**: " +
                            f"Importance <span style='color:{importance_color};font-weight:bold'>{adjusted_importance:.3f}</span>, " +
                            f"Confidence <span style='color:{confidence_color};font-weight:bold'>{confidence:.2f}</span>", 
                            unsafe_allow_html=True
                        )
                    except:
                        param_importance[param] = 0
            
            # Sort parameters by importance (now adjusted by confidence)
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create a single batch with the most important parameters
            important_params = [p[0] for p in sorted_params[:5] if p[1] > 0.1]  # Top 5 important parameters
            
            # If not enough important parameters found, use key parameters
            if len(important_params) < 3:
                important_params = ['investment_pct', 'price_drop_threshold', 'profit_threshold', 
                                   'stop_loss_fixed_pct', 'trailing_stop_pct'][:5]
            
            # Create ultra-fine tuning batch with confidence-based adjustments
            ultra_fine_optimize = {}
            for param in important_params:
                if param in best_params:
                    # Get best value and confidence for this parameter
                    best_value = best_params[param]
                    confidence = param_confidence.get(param, 0.5)  # Default to medium confidence if not available
                    
                    st.write(f"Fine-tuning **{param}** with confidence {confidence:.2f}")
                    
                    if isinstance(best_value, bool):
                        # Boolean parameters always test both values
                        ultra_fine_optimize[param] = [True, False]
                    elif isinstance(best_value, int):
                        # Integer parameters with confidence-based step sizes
                        if confidence > 0.7:
                            # High confidence - very narrow range
                            ultra_fine_optimize[param] = [
                                max(1, best_value - 1), 
                                best_value, 
                                best_value + 1
                            ]
                        elif confidence > 0.4:
                            # Medium confidence - standard range
                            ultra_fine_optimize[param] = [
                                max(1, best_value - 2), 
                                max(1, best_value - 1), 
                                best_value, 
                                best_value + 1, 
                                best_value + 2
                            ]
                        else:
                            # Low confidence - wider range
                            ultra_fine_optimize[param] = [
                                max(1, best_value - 3),
                                max(1, best_value - 2), 
                                max(1, best_value - 1), 
                                best_value, 
                                best_value + 1, 
                                best_value + 2,
                                best_value + 3
                            ]
                    else:
                        # Float parameters with confidence-based step sizes
                        if confidence > 0.7:
                            # High confidence - very small steps (2.5%)
                            step = best_value * 0.025
                            ultra_fine_optimize[param] = [
                                round(max(0.01, best_value - step*2), 2),
                                round(max(0.01, best_value - step), 2),
                                round(best_value, 2),
                                round(best_value + step, 2),
                                round(best_value + step*2, 2)
                            ]
                        elif confidence > 0.4:
                            # Medium confidence - small steps (5%)
                            step = best_value * 0.05
                            ultra_fine_optimize[param] = [
                                round(max(0.01, best_value - step*2), 2),
                                round(max(0.01, best_value - step), 2),
                                round(best_value, 2),
                                round(best_value + step, 2),
                                round(best_value + step*2, 2)
                            ]
                        else:
                            # Low confidence - larger steps (10%)
                            step = best_value * 0.1
                            ultra_fine_optimize[param] = [
                                round(max(0.01, best_value - step*3), 2),
                                round(max(0.01, best_value - step*2), 2),
                                round(max(0.01, best_value - step), 2),
                                round(best_value, 2),
                                round(best_value + step, 2),
                                round(best_value + step*2, 2),
                                round(best_value + step*3, 2)
                            ]
            
            # Add the ultra-fine tuning batch
            batches.append({
                'name': f"UltraFine_{dataset_hash}_iter{batch_iteration}",
                'optimize': ultra_fine_optimize,
                'fixed': {k: best_params.get(k, v) for k, v in default_params.items() if k not in ultra_fine_optimize},
                'combinations': 5000
            })
    
    # Third iteration (2) - hyper-fine parameter tuning based on iteration 1 results
    elif batch_iteration == 2 and previous_results:
        # Process all previous results to find the most promising parameter ranges
        all_flat_results = []
        for batch_id, batch_results in previous_results.items():
            all_flat_results.extend(batch_results)
        
        # Get top overall parameters from previous iterations
        sorted_results = sorted(all_flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        if sorted_results:
            # Use the top 3 parameter sets for this round
            top_param_sets = sorted_results[:3]
            best_params = top_param_sets[0]['parameters']
            
            # Create extremely narrow ranges around the best parameters
            # Use tighter steps for precision refinement
            hyper_fine_optimize = {}
            
            # Display top 3 parameter sets for reference
            st.subheader("Top 3 Parameter Sets")
            for i, params in enumerate(top_param_sets):
                metric_col = f"<span style='color:{'green' if i == 0 else 'orange' if i == 1 else 'blue'}'>"
                st.markdown(
                    f"{metric_col}Set {i+1}:</span> Return: {params['metrics']['return_pct']:.2f}%, " +
                    f"Drawdown: {params['metrics']['max_drawdown_pct']:.2f}%, " +
                    f"Trades: {params['metrics']['total_trades']}", 
                    unsafe_allow_html=True
                )
                
                # Show key parameters
                key_params = {k: v for k, v in params['parameters'].items() 
                            if k in ['investment_pct', 'price_drop_threshold', 'profit_threshold', 
                                    'stop_loss_fixed_pct', 'trailing_stop_pct']}
                st.json(key_params)
            
            # First, focus on the most critical parameters with extremely fine granularity
            critical_params = ['investment_pct', 'price_drop_threshold', 'profit_threshold', 
                             'stop_loss_fixed_pct', 'trailing_stop_pct']
            
            for param in critical_params:
                if param in best_params:
                    min_val, max_val, optimal_values, confidence = analyze_previous_results(
                        all_flat_results, param, top_n=3, min_sample_size=15
                    )
                    
                    # Display confidence score
                    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                    st.markdown(f"Parameter **{param}**: Hyper-fine tuning confidence <span style='color:{confidence_color};font-weight:bold'>{confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    # Extract best value
                    best_value = best_params[param]
                    
                    # Create micro-stepped values around the best value
                    if isinstance(best_value, int):
                        # For integer parameters - test exact values with minimal variation
                        step = max(1, int(best_value * 0.02))  # 2% step size, minimum 1
                        hyper_fine_optimize[param] = sorted(list(set([
                            max(1, best_value - step*2),
                            max(1, best_value - step),
                            best_value,
                            best_value + step,
                            best_value + step*2
                        ])))
                    else:
                        # For float parameters - use very small steps
                        step = max(0.05, best_value * 0.02)  # 2% step size, minimum 0.05
                        hyper_fine_optimize[param] = sorted([
                            round(max(0.01, best_value - step*2), 2),
                            round(max(0.01, best_value - step), 2),
                            round(best_value, 2),
                            round(best_value + step, 2),
                            round(best_value + step*2, 2)
                        ])
            
            # Batch for hyper-fine tuning of critical parameters
            batches.append({
                'name': f"HyperFine_Critical_{dataset_hash}_iter{batch_iteration}",
                'optimize': hyper_fine_optimize,
                'fixed': {k: best_params.get(k, v) for k, v in default_params.items() if k not in hyper_fine_optimize},
                'combinations': 5000
            })
            
            # Second batch: Test variations of the top 3 parameter sets
            # This creates "hybrids" of successful parameter combinations
            hybrid_optimize = {}
            
            # For each important parameter, collect the values from top 3 sets
            for param in ['investment_pct', 'price_drop_threshold', 'profit_threshold', 
                        'stop_loss_fixed_pct', 'trailing_stop_pct', 'max_position_duration']:
                param_values = []
                for params in top_param_sets:
                    if param in params['parameters']:
                        param_values.append(params['parameters'][param])
                
                # Remove duplicates
                param_values = sorted(list(set(param_values)))
                
                if len(param_values) > 1:
                    hybrid_optimize[param] = param_values
            
            # Add the hybrid optimization batch
            batches.append({
                'name': f"Hybrid_Top3_{dataset_hash}_iter{batch_iteration}",
                'optimize': hybrid_optimize,
                'fixed': {k: best_params.get(k, v) for k, v in default_params.items() if k not in hybrid_optimize},
                'combinations': 3000
            })
            
            # Third batch: Test secondary parameters keeping the best primary parameters fixed
            secondary_optimize = {}
            
            # Use secondary parameters
            for param in ['min_trade_size', 'partial_profit_taking', 'profit_taking_percentage', 
                        'use_trailing_stop_main', 'trailing_stop_activation_pct', 'number_of_bots']:
                
                if param in best_params:
                    min_val, max_val, optimal_values, confidence = analyze_previous_results(
                        all_flat_results, param, top_n=3
                    )
                    
                    # For boolean parameters
                    if param in ['partial_profit_taking', 'use_trailing_stop_main']:
                        secondary_optimize[param] = [True, False]
                        continue
                    
                    # For other parameters
                    best_value = best_params[param]
                    
                    # Create values based on parameter type
                    if isinstance(best_value, int):
                        step = max(1, int(best_value * 0.05))  # 5% step
                        secondary_optimize[param] = sorted(list(set([
                            max(1, best_value - step*2),
                            max(1, best_value - step),
                            best_value,
                            best_value + step,
                            best_value + step*2
                        ])))
                    else:
                        step = max(0.1, best_value * 0.05)  # 5% step, minimum 0.1
                        secondary_optimize[param] = sorted([
                            round(max(0.01, best_value - step*2), 2),
                            round(max(0.01, best_value - step), 2),
                            round(best_value, 2),
                            round(best_value + step, 2),
                            round(best_value + step*2, 2)
                        ])
            
            # Add the secondary parameters batch
            batches.append({
                'name': f"Secondary_Fine_{dataset_hash}_iter{batch_iteration}",
                'optimize': secondary_optimize,
                'fixed': {k: best_params.get(k, v) for k, v in default_params.items() if k not in secondary_optimize},
                'combinations': 3000
            })
    
    # Fourth iteration (3) - final refinement with extremely granular steps
    elif batch_iteration == 3 and previous_results:
        # Process all previous results to find the optimal parameter values
        all_flat_results = []
        for batch_id, batch_results in previous_results.items():
            all_flat_results.extend(batch_results)
        
        # Get top overall parameters from all previous iterations
        sorted_results = sorted(all_flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        
        if sorted_results:
            # Get the absolute best parameter set
            absolute_best_params = sorted_results[0]['parameters']
            
            st.subheader("Final Parameter Refinement")
            st.markdown(f"Best performance so far: **{sorted_results[0]['metrics']['return_pct']:.2f}%** return with " +
                       f"**{sorted_results[0]['metrics']['max_drawdown_pct']:.2f}%** max drawdown")
            
            # Show best parameters
            st.json(absolute_best_params)
            
            # Create ultra-fine batch with micro-steps around absolute best
            ultra_final_optimize = {}
            
            # Define critical parameters to fine-tune at this stage
            final_params = ['investment_pct', 'price_drop_threshold', 'profit_threshold', 
                           'trailing_stop_pct', 'profit_taking_percentage']
            
            for param in final_params:
                if param in absolute_best_params:
                    best_value = absolute_best_params[param]
                    
                    # Create micro-step values
                    if isinstance(best_value, int):
                        # For integers - extremely fine steps
                        step = max(1, int(best_value * 0.01))  # 1% step size, minimum 1
                        ultra_final_optimize[param] = sorted(list(set([
                            max(1, best_value - step),
                            best_value,
                            best_value + step
                        ])))
                    else:
                        # For floats - use extremely small steps
                        step = max(0.01, best_value * 0.01)  # 1% step size, minimum 0.01
                        ultra_final_optimize[param] = sorted([
                            round(max(0.01, best_value - step), 2),
                            round(best_value, 2),
                            round(best_value + step, 2)
                        ])
            
            # Add the ultra-final refinement batch
            batches.append({
                'name': f"Final_Refinement_{dataset_hash}_iter{batch_iteration}",
                'optimize': ultra_final_optimize,
                'fixed': {k: absolute_best_params.get(k, v) for k, v in default_params.items() if k not in ultra_final_optimize},
                'combinations': 3000
            })
            
            # Create a robustness test batch with slight variations
            # This tests how sensitive the strategy is to small parameter changes
            robustness_optimize = {}
            
            for param in ['investment_pct', 'price_drop_threshold', 'profit_threshold', 
                        'stop_loss_fixed_pct', 'trailing_stop_pct']:
                if param in absolute_best_params:
                    best_value = absolute_best_params[param]
                    
                    # Create random variations around the best value
                    if isinstance(best_value, int):
                        variations = []
                        for i in range(-3, 4):
                            if i != 0:  # Skip the best value itself as it's in the fixed params
                                variations.append(max(1, best_value + i))
                        robustness_optimize[param] = sorted(list(set(variations)))
                    else:
                        variations = []
                        for i in range(-3, 4):
                            if i != 0:  # Skip the best value
                                variations.append(round(max(0.01, best_value + (i * 0.01 * best_value)), 2))
                        robustness_optimize[param] = sorted(variations)
            
            # Add the robustness test batch
            batches.append({
                'name': f"Robustness_Test_{dataset_hash}_iter{batch_iteration}",
                'optimize': robustness_optimize,
                'fixed': {k: absolute_best_params.get(k, v) for k, v in default_params.items() if k not in robustness_optimize},
                'combinations': 4000
            })
    
    return batches

def get_dataset_info(price_data):
    """Get a summary of the dataset for record keeping"""
    if price_data is None or price_data.empty:
        return "Unknown dataset"
    
    # Get basic info
    start_date = price_data.index[0].strftime("%Y-%m-%d")
    end_date = price_data.index[-1].strftime("%Y-%m-%d")
    timeframe = "Unknown"
    symbol = "Unknown"
    
    # Try to extract timeframe and symbol from DataFrame
    if hasattr(price_data, 'name') and price_data.name:
        parts = price_data.name.split('_')
        if len(parts) > 1:
            symbol = parts[0]
            if len(parts) > 2:
                timeframe = parts[1]
    
    # Calculate total days
    days = (price_data.index[-1] - price_data.index[0]).days
    
    # Create dataset info string
    dataset_info = f"{symbol}_{timeframe}_{start_date}_to_{end_date}_{days}days"
    
    return dataset_info.replace("/", "")

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
                     max_combinations=10000, batch_id=None, task_id=None, batch_index=0):
    """
    Run an optimization over the parameter ranges provided with incremental result saving.
    
    Args:
        parameter_ranges: Dict of parameter names to lists of values
        fixed_parameters: Dict of parameters that stay constant
        price_data: DataFrame with price data
        initial_capital: Starting capital for backtest
        max_combinations: Maximum number of parameter combinations to test
        batch_id: Optional batch identifier
        task_id: Optional task ID for tracking progress
        batch_index: Batch index within the task
        
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
    status_summary = st.empty()
    
    # Check if we should resume from a saved state
    resumed_results = []
    start_index = 0
    
    if task_id:
        from utils.task_manager import TaskManager
        TaskManager.initialize_session_state()  # Ensure session state is initialized
        
        # Check if we have resume info for this task
        resume_info = st.session_state.resume_info.get(task_id, {})
        has_saved_results = resume_info.get('has_saved_results', False)
        
        # Get progress information
        progress = TaskManager.get_task_progress(task_id)
        is_resume_mode = progress and progress.get('resume_mode', False)
        
        if is_resume_mode:
            # Log that we're attempting to resume
            print(f"Attempting to resume task {task_id[:8]} batch {batch_id}")
            
            # Use our new helper method to load saved results from all possible sources
            batch_results = TaskManager.load_saved_results(task_id, batch_id, results_manager)
            
            # If we found results, use them
            if batch_results:
                resumed_results = batch_results
                start_index = len(batch_results)
                print(f"Resuming from test {start_index}/{len(combinations)} in batch {batch_id}")
                st.info(f"Resuming from test {start_index}/{len(combinations)} in batch {batch_id}")
                
                # Save these results in the task manager if they weren't there already
                if batch_id not in st.session_state.incremental_results.get(task_id, {}):
                    if task_id not in st.session_state.incremental_results:
                        st.session_state.incremental_results[task_id] = {}
                    st.session_state.incremental_results[task_id][batch_id] = batch_results
                    print(f"Restored {len(batch_results)} results for batch {batch_id} in task {task_id[:8]}")
            else:
                print(f"No saved results found for batch {batch_id}")
                st.warning(f"No saved results found for batch {batch_id}. Starting from beginning.")
    
    # Initialize results with any resumed data
    results = resumed_results.copy()
    
    # Check if all tests are already complete
    if start_index >= len(combinations):
        status_text.text(f"All tests already completed for batch {batch_id}!")
        
        # Sort and return the results
        sorted_results = sorted(results, key=lambda x: x['metrics']['return_pct'], reverse=True)
        return sorted_results
    
    # Run backtest for each combination
    for i, combo in enumerate(combinations[start_index:], start=start_index):
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
        
        # Add summary info for incremental progress
        if results:
            # Show top 3 results so far
            sorted_current = sorted(results, key=lambda x: x['metrics']['return_pct'], reverse=True)[:3]
            summary_text = "Current top results:\n"
            for j, res in enumerate(sorted_current):
                summary_text += f"#{j+1}: Return {res['metrics']['return_pct']:.2f}%, "
                summary_text += ", ".join([f"{p}={v}" for p, v in sorted(res['parameters'].items())])
                summary_text += "\n"
            status_summary.text(summary_text)
        
        # Run backtest with these parameters
        result = run_backtest(params, price_data, initial_capital)
        results.append(result)
        
        # Save result incrementally if we have a task ID
        if task_id:
            from utils.task_manager import TaskManager
            TaskManager.add_test_result(task_id, batch_index, i, result, results_manager)
            
            # Update task progress
            TaskManager.update_task_progress(
                task_id, 
                batch_index, 
                progress, 
                current_test=i,
                params=params
            )
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text(f"Optimization completed! Tested {len(combinations)} parameter combinations.")
    
    # Sort results by return
    sorted_results = sorted(results, key=lambda x: x['metrics']['return_pct'], reverse=True)
    
    # Store final batch results
    if batch_id:
        dataset_info = get_dataset_info(price_data)
        
        batch_info = {
            'batch_id': batch_id,
            'parameter_ranges': parameter_ranges,
            'fixed_parameters': fixed_parameters,
            'total_combinations': len(combinations),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset_info,
            'is_complete': True
        }
        results_manager.save_optimization_results(
            sorted_results,
            'OptimizedAlternativeDCAv5',
            parameter_ranges,
            batch_info
        )
    
    return sorted_results

def run_batch_optimization(batches, price_data, initial_capital=1000.0, max_combinations_per_batch=10000, 
                          task_id=None, resume_mode=False, max_tests_per_batch=1000):
    """
    Run multiple optimization batches sequentially with incremental saving and resume capability.
    
    Args:
        batches: List of batch dictionaries with parameter_ranges and fixed_parameters
        price_data: DataFrame with price data
        initial_capital: Starting capital for backtest
        max_combinations_per_batch: Maximum parameter combinations per batch
        task_id: Optional task ID for progress tracking
        resume_mode: Whether to try to resume from previously saved state
        max_tests_per_batch: Maximum number of tests to run before saving
        
    Returns:
        Dictionary of batch results
    """
    all_results = {}
    
    progress_bar = st.progress(0)
    batch_status = st.empty()
    
    # Show resume status if applicable
    if resume_mode and task_id:
        st.info(f"Attempting to resume optimization task {task_id[:8]}...")
    
    # Create a new task if needed
    if not task_id:
        from utils.task_manager import TaskManager
        # Initialize session state before accessing
        TaskManager.initialize_session_state()
        task_id = TaskManager.create_task(
            batches, 
            price_data, 
            initial_capital=initial_capital, 
            auto_adjust=True,
            max_tests_per_batch=max_tests_per_batch
        )
    elif resume_mode:
        # If resuming, we need to make sure the task is properly initialized
        from utils.task_manager import TaskManager
        # Initialize session state before accessing
        TaskManager.initialize_session_state()
        # Use the existing task_id but create a new task with it
        task_id = TaskManager.create_task(
            batches, 
            price_data, 
            initial_capital=initial_capital, 
            auto_adjust=True,
            max_tests_per_batch=max_tests_per_batch,
            resume_task_id=task_id
        )
        # Log resumption
        st.info(f"Resuming optimization task {task_id[:8]} with saved progress...")
        st.info(f"Created new optimization task with ID: {task_id[:8]}")
    
    for i, batch in enumerate(batches):
        batch_id = batch.get('name', f"Batch {i+1}")
        batch_status.text(f"Running {batch_id} ({i+1}/{len(batches)})")
        
        # Update task progress
        if task_id:
            from utils.task_manager import TaskManager
            # Initialize session state before accessing
            TaskManager.initialize_session_state()
            TaskManager.update_task_progress(task_id, i, 0.0)
            TaskManager.update_task_status(task_id, 'running', f"Running batch {i+1}/{len(batches)}")
        
        # Run this batch
        batch_results = optimize_strategy(
            batch['optimize'],
            batch['fixed'],
            price_data,
            initial_capital,
            batch['combinations'] if 'combinations' in batch else max_combinations_per_batch,
            batch_id=batch_id,
            task_id=task_id,
            batch_index=i
        )
        
        all_results[batch_id] = batch_results
        
        # Update overall progress
        progress_bar.progress((i + 1) / len(batches))
        
        # Add delay to allow UI updates
        time.sleep(0.1)
    
    # Mark task as completed
    if task_id:
        from utils.task_manager import TaskManager
        # Initialize session state before accessing
        TaskManager.initialize_session_state()
        TaskManager.update_task_status(task_id, 'completed', "All batches completed successfully")
    
    batch_status.text(f"Completed all {len(batches)} batches!")
    
    # Display overall best parameters found
    st.markdown("### Overall Best Parameters Found")
    
    # Flatten and sort all results by return percentage
    all_flat_results = []
    for batch_id, batch_results in all_results.items():
        all_flat_results.extend(batch_results)
    
    sorted_results = sorted(all_flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
    
    # Display top 3 parameter combinations
    for i, result in enumerate(sorted_results[:3]):
        st.markdown(f"**#{i+1} Return: {result['metrics']['return_pct']:.2f}%**")
        params_text = []
        for param, value in result['parameters'].items():
            params_text.append(f"{param}: {value}")
        st.markdown(", ".join(params_text))
    
    return all_results

def load_previous_results(dataset_info):
    """Load previous optimization results for the same dataset"""
    all_results = {}
    
    # Get all optimization results
    optimization_files = results_manager.list_optimization_results()
    
    # Filter to only include results for this dataset
    for opt_file in optimization_files:
        try:
            results = results_manager.load_optimization_results(opt_file['filepath'])
            if 'additional_info' in results and 'dataset' in results['additional_info']:
                if results['additional_info']['dataset'] == dataset_info:
                    # Load the batch results
                    batch_results = results['results']
                    all_results[results['additional_info']['batch_id']] = batch_results
        except Exception as e:
            # Skip files that can't be loaded
            continue
    
    return all_results

def display_optimization_analysis(results, dataset_info):
    """Display detailed analysis of optimization results"""
    if not results:
        st.warning("No results to analyze.")
        return
    
    # Flatten all results
    all_results = []
    for batch_id, batch_results in results.items():
        all_results.extend(batch_results)
    
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
    
    if not sorted_results:
        st.warning("No valid results found for analysis.")
        return
    
    st.subheader(f"Optimization Analysis for {dataset_info}")
    
    # Display best parameters in a nice format
    best_result = sorted_results[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Best Parameters Found")
        
        param_df = pd.DataFrame([best_result['parameters']]).T.reset_index()
        param_df.columns = ['Parameter', 'Value']
        
        # Group parameters into categories
        categories = {
            'Basic': ['investment_pct', 'price_drop_threshold', 'profit_threshold', 'trading_fee_pct'],
            'Risk Management': ['stop_loss_fixed_pct', 'trailing_stop_pct', 'max_position_duration'],
            'Advanced': ['number_of_bots', 'min_trade_size', 'partial_profit_taking',
                        'profit_taking_percentage', 'use_trailing_stop_main', 'trailing_stop_activation_pct']
        }
        
        # Create tabs for different parameter categories
        tabs = st.tabs(['All'] + list(categories.keys()))
        
        with tabs[0]:
            st.dataframe(param_df, use_container_width=True)
        
        for i, (category, params) in enumerate(categories.items(), start=1):
            with tabs[i]:
                category_params = param_df[param_df['Parameter'].isin(params)]
                st.dataframe(category_params, use_container_width=True)
    
    with col2:
        st.markdown("### Performance Metrics")
        
        # Create metrics display
        st.metric("Return (%)", f"{best_result['metrics']['return_pct']:.2f}%")
        st.metric("Max Drawdown (%)", f"{best_result['metrics']['max_drawdown_pct']:.2f}%")
        st.metric("Win Rate (%)", f"{best_result['metrics']['win_rate']:.2f}%")
        st.metric("Sharpe Ratio", f"{best_result['metrics']['sharpe_ratio']:.2f}")
        st.metric("Profit Factor", f"{best_result['metrics']['profit_factor']:.2f}")
        st.metric("Number of Trades", f"{best_result['metrics']['trades_count']}")
    
    # Parameter sensitivity analysis
    st.subheader("Parameter Sensitivity Analysis")
    
    # Collect parameter values and corresponding returns for correlation analysis
    param_data = {}
    for param in sorted(set([p for r in sorted_results for p in r['parameters'].keys()])):
        param_data[param] = {'values': [], 'returns': []}
    
    for result in sorted_results:
        for param, value in result['parameters'].items():
            if isinstance(value, (int, float)):  # Only include numeric parameters
                param_data[param]['values'].append(value)
                param_data[param]['returns'].append(result['metrics']['return_pct'])
    
    # Calculate correlations
    correlations = {}
    for param, data in param_data.items():
        if len(data['values']) > 5:  # Need at least a few data points
            try:
                corr = np.corrcoef(data['values'], data['returns'])[0, 1]
                correlations[param] = corr
            except:
                pass
    
    # Sort parameters by absolute correlation (importance)
    param_importance = sorted(correlations.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    
    # Create parameter importance bar chart
    if param_importance:
        param_names = [p[0] for p in param_importance]
        corr_values = [p[1] for p in param_importance]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=param_names,
            y=corr_values,
            marker_color=['green' if c > 0 else 'red' for c in corr_values],
            text=[f"{abs(c):.2f}" for c in corr_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Parameter Importance (Correlation with Return)",
            xaxis_title="Parameter",
            yaxis_title="Correlation",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Annotate correlation meaning
        st.info("""
        **Parameter Importance Explanation:**
        - **Positive correlation (green)**: Higher parameter values tend to increase returns
        - **Negative correlation (red)**: Lower parameter values tend to increase returns
        - **Higher absolute value**: Stronger influence on performance
        """)
    
    # Parameter exploration plots
    st.subheader("Parameter Exploration")
    
    # Select a parameter to visualize
    top_params = [p[0] for p in param_importance[:min(10, len(param_importance))]]
    selected_param = st.selectbox("Select parameter to explore", top_params)
    
    if selected_param in param_data and len(param_data[selected_param]['values']) > 0:
        # Create scatter plot of parameter value vs return
        fig = go.Figure()
        
        # Main scatter plot
        fig.add_trace(go.Scatter(
            x=param_data[selected_param]['values'],
            y=param_data[selected_param]['returns'],
            mode='markers',
            marker=dict(
                size=10,
                color=param_data[selected_param]['returns'],
                colorscale='Viridis',
                colorbar=dict(title="Return (%)"),
                showscale=True
            ),
            text=[f"Return: {r:.2f}%" for r in param_data[selected_param]['returns']],
            hoverinfo='text'
        ))
        
        # Add trend line
        if len(param_data[selected_param]['values']) > 2:  # Need at least 3 points for a trend line
            try:
                z = np.polyfit(param_data[selected_param]['values'], param_data[selected_param]['returns'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(param_data[selected_param]['values']), 
                                      max(param_data[selected_param]['values']), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                ))
            except:
                pass
        
        # Update layout
        fig.update_layout(
            title=f"Impact of {selected_param} on Return",
            xaxis_title=selected_param,
            yaxis_title="Return (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show optimal range for this parameter
        values = param_data[selected_param]['values']
        returns = param_data[selected_param]['returns']
        
        # Get top 3 performing values
        paired_data = list(zip(values, returns))
        paired_data.sort(key=lambda x: x[1], reverse=True)
        top_values = [v for v, _ in paired_data[:3]]
        
        st.markdown(f"**Optimal range for {selected_param}:** {min(top_values):.2f} to {max(top_values):.2f}")
        st.markdown(f"**Top performing values:** {', '.join([f'{v:.2f}' for v in top_values])}")
    
    # Cross-parameter analysis
    st.subheader("Cross-Parameter Analysis")
    
    if len(top_params) >= 2:
        # Select two parameters to compare
        col1, col2 = st.columns(2)
        with col1:
            x_param = st.selectbox("X-axis parameter", top_params, index=0)
        with col2:
            y_param = st.selectbox("Y-axis parameter", top_params, index=min(1, len(top_params)-1))
        
        if x_param != y_param:
            # Create cross-parameter data
            cross_data = {'x': [], 'y': [], 'return': [], 'text': []}
            
            for result in sorted_results:
                if x_param in result['parameters'] and y_param in result['parameters']:
                    x_val = result['parameters'][x_param]
                    y_val = result['parameters'][y_param]
                    
                    if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                        cross_data['x'].append(x_val)
                        cross_data['y'].append(y_val)
                        cross_data['return'].append(result['metrics']['return_pct'])
                        cross_data['text'].append(
                            f"{x_param}: {x_val}<br>{y_param}: {y_val}<br>Return: {result['metrics']['return_pct']:.2f}%"
                        )
            
            if cross_data['x']:
                # Create scatter plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=cross_data['x'],
                    y=cross_data['y'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=cross_data['return'],
                        colorscale='Viridis',
                        colorbar=dict(title="Return (%)"),
                        showscale=True
                    ),
                    text=cross_data['text'],
                    hoverinfo='text'
                ))
                
                # Find and mark the best point
                best_idx = cross_data['return'].index(max(cross_data['return']))
                
                fig.add_trace(go.Scatter(
                    x=[cross_data['x'][best_idx]],
                    y=[cross_data['y'][best_idx]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    name='Best Result'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Parameter Interaction: {x_param} vs {y_param}",
                    xaxis_title=x_param,
                    yaxis_title=y_param,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **How to read this chart:**
                - Each point represents a tested parameter combination
                - Color represents return percentage (darker = higher return)
                - The red star marks the best performing combination
                - Look for clusters of darker colors to identify optimal parameter regions
                """)
            else:
                st.warning("Not enough data for cross-parameter analysis of these parameters.")

def main():
    st.title("🤖 Automatic Strategy Optimization")
    st.markdown("""
    This tool automates the optimization process for the V5 strategy. It:
    
    1. **Automatically determines** which parameters to optimize and their ranges
    2. **Progressively refines** parameters based on results from previous iterations  
    3. **Persists all results** with dataset information for future reference
    4. **Visualizes improvement** across optimization iterations
    5. **Offers multiple optimization strategies** for different goals
    6. **Compares results across datasets** to find robust parameters
    7. **Tracks progress** of parameter convergence and performance improvements
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
                            
                            # Set a name for the dataset
                            df.name = selected_file.replace('.csv', '')
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
            
            df_name = f"{exchange}_{symbol.replace('/', '-')}_{timeframe}"
            
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
                        
                        # Set a name for the dataset
                        df.name = df_name
                        
                        # Option to save data
                        save_filename = f"{df_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
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
    
    # Main content area - tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Auto Optimization", "Results Analysis", "Progress Tracking", "Dataset Comparison"])
    
    with tab1:
        st.header("Automatic Parameter Optimization")
        st.markdown("""
        This tab runs automatic optimization with progressively refined parameters.
        The system automatically determines what parameters to optimize and their ranges.
        """)
        
        # Dataset information for record keeping
        if 'df' in locals() and df is not None and not df.empty:
            dataset_info = get_dataset_info(df)
            st.info(f"Current dataset: {dataset_info}")
            
            # Check for previous optimization results for this dataset
            previous_results = load_previous_results(dataset_info)
            
            if previous_results:
                st.success(f"Found {len(previous_results)} previous optimization batch(es) for this dataset")
                
                # Display previous iteration summary
                flat_results = []
                for batch_id, batch_results in previous_results.items():
                    top_result = sorted(batch_results, key=lambda x: x['metrics']['return_pct'], reverse=True)[0]
                    flat_results.append({
                        'batch_id': batch_id,
                        'return_pct': top_result['metrics']['return_pct'],
                        'max_drawdown_pct': top_result['metrics']['max_drawdown_pct'],
                        'win_rate': top_result['metrics']['win_rate'],
                        'sharpe_ratio': top_result['metrics']['sharpe_ratio']
                    })
                
                # Create progress summary
                st.subheader("Previous Optimization Progress")
                summary_df = pd.DataFrame(flat_results)
                st.dataframe(summary_df)
                
                # Calculate what iteration we're on based on previous results
                auto_iteration_number = len(previous_results) // 3  # Each iteration has about 3 batches
                
                # Allow user to choose a specific iteration or use the automatically determined one
                st.subheader("Optimization Iteration Selection")
                use_custom_iteration = st.checkbox("Manually select iteration", 
                                                  value=False,
                                                  help="Choose a specific iteration rather than the automatically determined next one")
                
                if use_custom_iteration:
                    # Show all available iterations (0-3)
                    iteration_options = [
                        "Iteration 0: Initial broad parameter exploration",
                        "Iteration 1: First parameter refinement",
                        "Iteration 2: Hyper-fine parameter tuning",
                        "Iteration 3: Final refinement and robustness testing"
                    ]
                    
                    selected_option = st.selectbox(
                        "Select optimization iteration to run",
                        options=iteration_options,
                        index=min(auto_iteration_number, 3),  # Default to auto-detected iteration, max index 3
                        help="Each iteration uses progressively narrower parameter ranges based on previous results"
                    )
                    
                    # Extract iteration number from selected option
                    iteration_number = int(selected_option.split(":")[0].split(" ")[1])
                    
                    # Warning if going back to earlier iterations
                    if iteration_number < auto_iteration_number:
                        st.warning(f"You're running iteration {iteration_number} even though data suggests you should be on iteration {auto_iteration_number}. This might be useful for exploring different parameter paths.")
                    
                    # Special warning for iteration 0 if we already have results
                    if iteration_number == 0 and auto_iteration_number > 0:
                        st.warning("Starting a new iteration 0 will explore different initial parameters than previous runs. This is useful for finding alternative parameter paths.")
                else:
                    # Use automatically determined iteration
                    iteration_number = auto_iteration_number
                    st.success(f"Automatically using iteration {iteration_number} based on previous optimization results")
                
                # Generate batches for the selected iteration
                next_batches = generate_auto_batch_config(iteration_number, df, dataset_info, previous_results)
                
                # Show what will be tested in the selected iteration
                st.subheader(f"Next Optimization: Iteration #{iteration_number}")
                for i, batch in enumerate(next_batches):
                    with st.expander(f"Batch {i+1}: {batch['name']}", expanded=False):
                        st.write("Parameters to optimize:")
                        for param, values in batch['optimize'].items():
                            st.write(f"- {param}: {values}")
            else:
                st.info("No previous optimization results found for this dataset. Starting fresh.")
                
                # Generate initial batches
                next_batches = generate_auto_batch_config(0, df, dataset_info)
                
                # Show what will be tested
                st.subheader("Initial Optimization Batches")
                for i, batch in enumerate(next_batches):
                    with st.expander(f"Batch {i+1}: {batch['name']}", expanded=False):
                        st.write("Parameters to optimize:")
                        for param, values in batch['optimize'].items():
                            st.write(f"- {param}: {values}")
            
            # Performance and batch settings
            with st.expander("Advanced Settings", expanded=False):
                # Add batch size control
                max_tests_per_batch = st.slider(
                    "Max tests per batch", 
                    min_value=100, 
                    max_value=5000, 
                    value=1000, 
                    step=100,
                    help="Maximum number of parameter combinations to test in each batch before saving results"
                )
                st.session_state['max_tests_per_batch'] = max_tests_per_batch
                
                # Option to resume from previous task
                has_previous_task = 'last_task_id' in st.session_state
                if has_previous_task:
                    st.info(f"Previous optimization task found with ID: {st.session_state.last_task_id[:8]}")
                    resume_previous = st.checkbox("Resume previous optimization?", 
                                                 value=True,
                                                 help="Continue optimization from where it was interrupted")
                else:
                    resume_previous = False
            
            # Get active task information from task manager (if any)
            from utils.task_manager import TaskManager
            # Initialize session state before accessing
            TaskManager.initialize_session_state()
            active_tasks = TaskManager.get_active_tasks()
            
            if active_tasks:
                st.subheader("Active Optimization Tasks")
                for task in active_tasks:
                    status_color = "orange" if task['status'] == 'pending' else "blue" if task['status'] == 'running' else "green"
                    
                    st.markdown(f"""
                    <div style="padding:10px; border-left:5px solid {status_color}; background-color:#f0f0f0; margin-bottom:10px;">
                        <b>Task ID:</b> {task['task_id'][:8]} | <b>Status:</b> <span style="color:{status_color}">{task['status']}</span><br/>
                        <b>Progress:</b> Batch {task['current_batch']+1}/{len(task['batches'])}, {int(task['batch_progress']*100)}% complete<br/>
                        <b>Current test:</b> {task.get('current_test', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show progress bar
                    progress = (task['current_batch'] + task['batch_progress']) / len(task['batches'])
                    st.progress(min(1.0, progress))
            
            # Offer both single-batch and multi-batch options
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to run just one round of optimization
                run_button_text = "Resume Optimization" if resume_previous else "Run Auto-Optimization"
                if st.button(run_button_text, key="run_auto_opt"):
                    if 'df' in locals() and df is not None and not df.empty:
                        with st.spinner("Running automatic optimization..."):
                            # Run batch optimization with the generated batches
                            max_tests_per_batch = st.session_state.get('max_tests_per_batch', 1000)
                            
                            # Use existing task ID for resume or create a new one
                            if resume_previous and 'last_task_id' in st.session_state:
                                task_id = st.session_state.last_task_id
                                resume_mode = True
                                st.info(f"Resuming optimization from task {task_id[:8]}...")
                            else:
                                task_id = str(uuid.uuid4())
                                resume_mode = False
                            
                            batch_results = run_batch_optimization(
                                next_batches,
                                df,
                                initial_capital,
                                max_combinations_per_batch=10000,
                                task_id=task_id,
                                resume_mode=resume_mode,
                                max_tests_per_batch=max_tests_per_batch
                            )
                            
                            # Store in session state
                            st.session_state.latest_results = batch_results
                            st.session_state.dataset_info = dataset_info
                            st.session_state.last_task_id = task_id
            
            with col2:
                # Option for continuous optimization with multiple iterations
                num_iterations = st.number_input("Number of Iterations", min_value=2, max_value=4, value=4,
                                              help="Run multiple iterations of optimization automatically (up to 4 iterations: initial, refinement, hyper-fine, and final)")
                
                if st.button("Run Multiple Iterations", key="run_multi_iter"):
                    if 'df' in locals() and df is not None and not df.empty:
                        # Initialize progress tracking
                        progress_bar = st.progress(0)
                        iteration_status = st.empty()
                        results_container = st.container()
                        
                        # Store combined results from all iterations
                        all_results = {}
                        
                        # Run through multiple iterations
                        for iteration in range(num_iterations):
                            iteration_number = iteration + 1
                            iteration_status.write(f"Running iteration {iteration_number}/{num_iterations}...")
                            
                            # Generate batches for this iteration based on previous results
                            if iteration == 0:
                                # First iteration uses the initial batches
                                iter_batches = next_batches
                            else:
                                # Process results from previous iterations to generate refined batches
                                iter_batches = generate_auto_batch_config(
                                    iteration,
                                    df,
                                    dataset_info['name'],
                                    all_results
                                )
                            
                            # Show current batches being processed
                            with results_container.expander(f"Iteration {iteration_number} Batches", expanded=False):
                                for i, batch in enumerate(iter_batches):
                                    st.write(f"Batch {i+1}: {batch['name']}")
                                    for param, values in batch['optimize'].items():
                                        st.write(f"- {param}: {values}")
                            
                            # Run batch optimization for this iteration
                            with st.spinner(f"Running iteration {iteration_number}/{num_iterations}..."):
                                max_tests_per_batch = st.session_state.get('max_tests_per_batch', 1000)
                                
                                # Create a unique task ID for this iteration or use the existing one from the first iteration
                                if iteration == 0:
                                    task_id = str(uuid.uuid4())
                                    st.session_state.multi_iter_task_id = task_id
                                    resume_mode = False
                                else:
                                    task_id = st.session_state.multi_iter_task_id
                                    resume_mode = True
                                
                                batch_results = run_batch_optimization(
                                    iter_batches,
                                    df,
                                    initial_capital,
                                    max_combinations_per_batch=10000,
                                    task_id=task_id,
                                    resume_mode=resume_mode,
                                    max_tests_per_batch=max_tests_per_batch
                                )
                                
                                # Add results to combined results
                                all_results.update(batch_results)
                                
                                # Store the latest results in session state
                                st.session_state.latest_results = all_results
                                st.session_state.dataset_info = dataset_info
                                st.session_state.last_task_id = task_id
                                
                                # Update progress
                                progress_bar.progress((iteration + 1) / num_iterations)
                        
                        # Final success message
                        iteration_status.success(f"Completed {num_iterations} optimization iterations!")
                        
                        # Display summary of improvements
                        with results_container:
                            st.subheader("Optimization Progress Summary")
                            
                            # Extract all results and sort by performance
                            flat_results = []
                            for batch_id, batch_results in all_results.items():
                                for result in batch_results:
                                    # Add iteration info based on batch name (new format with hash and iteration)
                                    # Parse iteration number from batch id (format: Name_hash_iterX)
                                    try:
                                        parts = batch_id.split('_')
                                        if len(parts) >= 3 and 'iter' in parts[-1]:
                                            iteration = parts[-1].replace('iter', '')
                                        else:
                                            iteration = '0'  # Default if not found
                                    except Exception:
                                        iteration = '0'  # Default if parsing fails
                                        
                                    # Get current timestamp for results
                                    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
                                    
                                    result['batch_info'] = {
                                        'id': batch_id,
                                        'iteration': iteration,
                                        'timestamp': current_time
                                    }
                                    flat_results.append(result)
                            
                            # Sort by return percentage
                            sorted_results = sorted(flat_results, key=lambda x: x['metrics']['return_pct'], reverse=True)
                            
                            # Show top 5 results
                            st.write("Top 5 Parameter Combinations:")
                            for i, result in enumerate(sorted_results[:5]):
                                st.write(f"**#{i+1}:** Return: {result['metrics']['return_pct']:.2f}%, " +
                                        f"Max DD: {result['metrics']['max_drawdown_pct']:.2f}%, " +
                                        f"Sharpe: {result['metrics'].get('sharpe_ratio', 0):.2f}")
                                
                                # Show parameters in a more compact format
                                param_str = ", ".join([f"{k}: {v}" for k, v in result['parameters'].items()])
                                st.write(f"Parameters: {param_str}")
                                st.write("---")
                        
                        # Find the best performance across all iterations
                        best_return = 0
                        best_params = {}
                        
                        for batch_id, results_list in all_results.items():
                            for result in results_list:
                                if result['metrics']['return_pct'] > best_return:
                                    best_return = result['metrics']['return_pct']
                                    best_params = result['parameters']
                        
                        # Display best parameters found
                        st.subheader("Best Parameters Found")
                        st.json(best_params)
                        
                        # Metrics for best parameters
                        st.metric(
                            label="Best Return", 
                            value=f"{best_return:.2f}%"
                        )
                else:
                    st.error("Please load a dataset first.")
        else:
            st.warning("Please load a dataset to begin optimization.")
    
    with tab2:
        st.header("Optimization Results Analysis")
        
        # List saved optimization results
        st.subheader("Saved Optimization Results")
        all_results = results_manager.list_optimization_results(strategy_name="OptimizedAlternativeDCAv5")
        
        if all_results:
            # Group results by dataset
            results_by_dataset = {}
            for result in all_results:
                try:
                    loaded_result = results_manager.load_optimization_results(result['filepath'])
                    if 'additional_info' in loaded_result and 'dataset' in loaded_result['additional_info']:
                        dataset = loaded_result['additional_info']['dataset']
                        if dataset not in results_by_dataset:
                            results_by_dataset[dataset] = []
                        results_by_dataset[dataset].append({
                            **result,
                            'loaded_data': loaded_result
                        })
                except:
                    # Skip results that can't be loaded
                    pass
            
            # Create a selectbox for datasets
            datasets = list(results_by_dataset.keys())
            if datasets:
                selected_dataset = st.selectbox("Select dataset to analyze", datasets)
                
                if selected_dataset in results_by_dataset:
                    dataset_results = results_by_dataset[selected_dataset]
                    
                    # Display basic info
                    st.write(f"Found {len(dataset_results)} optimization batches for this dataset")
                    
                    # Convert to format needed for display_optimization_analysis
                    formatted_results = {}
                    for result_data in dataset_results:
                        loaded_data = result_data['loaded_data']
                        batch_id = loaded_data['additional_info']['batch_id']
                        formatted_results[batch_id] = loaded_data['results']
                    
                    # Display analysis
                    display_optimization_analysis(formatted_results, selected_dataset)
            else:
                st.info("No optimization results found with dataset information.")
        else:
            st.info("No saved optimization results found.")
    
    with tab3:
        # Use the Optimization Progress Tracker module
        import pages.optimization_progress
        pages.optimization_progress.app()
        
    with tab4:
        # Use the Dataset Comparison module
        import pages.dataset_comparison
        pages.dataset_comparison.app()
        
if __name__ == "__main__":
    main()