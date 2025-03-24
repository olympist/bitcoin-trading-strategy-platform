import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.results_manager import ResultsManager
from visualization.performance_plots import PerformancePlots

# Initialize components
results_manager = ResultsManager()

def identify_optimization_runs(results):
    """
    Identify distinct optimization runs based on timestamps or dataset information.
    
    Args:
        results: List of loaded result dictionaries
        
    Returns:
        Dictionary mapping run IDs to lists of result files
    """
    # Group by dataset name or timestamp
    runs = {}
    
    for result in results:
        try:
            # Try to load the result to get dataset info
            loaded_result = results_manager.load_optimization_results(result['filepath'])
            
            # First try to use dataset information
            if ('additional_info' in loaded_result and 
                'dataset' in loaded_result['additional_info']):
                
                dataset_name = loaded_result['additional_info']['dataset']
                
                # Use dataset name as the primary grouping
                if dataset_name not in runs:
                    # Get timestamp or creation date for sorting
                    try:
                        creation_time = datetime.fromtimestamp(os.path.getctime(result['filepath']))
                    except:
                        creation_time = datetime.now()
                        
                    runs[dataset_name] = {
                        'start_time': creation_time,
                        'end_time': creation_time,
                        'results': [result],
                        'dataset': dataset_name
                    }
                else:
                    # Add to existing dataset group
                    runs[dataset_name]['results'].append(result)
                    
                    # Update end time if needed
                    try:
                        file_time = datetime.fromtimestamp(os.path.getctime(result['filepath']))
                        runs[dataset_name]['end_time'] = max(runs[dataset_name]['end_time'], file_time)
                    except:
                        pass
                
                continue  # Skip to next result since we've handled this one
            
            # Fall back to timestamp-based grouping if no dataset info
            timestamp = os.path.basename(result['filepath']).split('_')[0]
            
            # Convert to datetime object for better grouping
            if len(timestamp) >= 8:  # Make sure it's a valid timestamp
                try:
                    dt = datetime.strptime(timestamp, '%Y%m%d')
                    
                    # Group runs that are close in time (within 1 day)
                    found_group = False
                    for run_id, run_info in runs.items():
                        if abs((dt - run_info['start_time']).total_seconds()) < 86400:  # 24 hours
                            runs[run_id]['results'].append(result)
                            runs[run_id]['end_time'] = max(runs[run_id]['end_time'], dt)
                            found_group = True
                            break
                    
                    if not found_group:
                        # Create a new run
                        run_id = f"Run-{len(runs) + 1}"
                        runs[run_id] = {
                            'start_time': dt,
                            'end_time': dt,
                            'results': [result]
                        }
                except Exception:
                    # Skip if timestamp format is invalid
                    continue
        except Exception:
            # Skip results that can't be properly processed
            continue
    
    # Sort runs by start time
    sorted_runs = {}
    for i, (run_id, run_info) in enumerate(sorted(runs.items(), key=lambda x: x[1]['start_time'])):
        sorted_run_id = f"Run-{i + 1}"
        sorted_runs[sorted_run_id] = run_info
    
    return sorted_runs

def analyze_optimization_progress(run_results):
    """
    Analyze the progress of optimization runs.
    
    Args:
        run_results: List of results for a single optimization run
        
    Returns:
        Dictionary with progress metrics
    """
    # Initialize metrics
    progress = {
        'timestamps': [],
        'return_pct': [],
        'max_drawdown_pct': [],
        'win_rate': [],
        'sharpe_ratio': [],
        'best_parameters': {},
        'parameter_evolution': {}
    }
    
    # Collect all parameter names
    all_params = set()
    for result in run_results:
        try:
            # Load the result file
            loaded_result = results_manager.load_optimization_results(result['filepath'])
            
            # Extract parameter names from all results
            for batch_result in loaded_result['results']:
                all_params.update(batch_result['parameters'].keys())
        except:
            continue
    
    # Initialize parameter evolution tracking
    for param in all_params:
        progress['parameter_evolution'][param] = {
            'timestamps': [],
            'values': []
        }
    
    # Analyze each result chronologically
    for result in sorted(run_results, key=lambda x: x['timestamp']):
        try:
            # Extract timestamp
            timestamp = datetime.strptime(result['timestamp'].split('_')[0], '%Y%m%d')
            
            # Load the result file
            loaded_result = results_manager.load_optimization_results(result['filepath'])
            
            # Find the best result in this batch (by return)
            best_result = None
            best_return = -float('inf')
            
            for batch_result in loaded_result['results']:
                if batch_result['metrics']['return_pct'] > best_return:
                    best_return = batch_result['metrics']['return_pct']
                    best_result = batch_result
            
            if best_result:
                # Add metrics
                progress['timestamps'].append(timestamp)
                progress['return_pct'].append(best_result['metrics']['return_pct'])
                progress['max_drawdown_pct'].append(best_result['metrics']['max_drawdown_pct'])
                progress['win_rate'].append(best_result['metrics']['win_rate'])
                progress['sharpe_ratio'].append(best_result['metrics']['sharpe_ratio'])
                
                # Track best parameters
                if not progress['best_parameters'] or best_return > max(progress['return_pct'][:-1], default=0):
                    progress['best_parameters'] = best_result['parameters'].copy()
                
                # Track parameter evolution
                for param, value in best_result['parameters'].items():
                    if param in progress['parameter_evolution']:
                        progress['parameter_evolution'][param]['timestamps'].append(timestamp)
                        progress['parameter_evolution'][param]['values'].append(value)
        except:
            # Skip results that can't be loaded or analyzed
            continue
    
    return progress

def create_progress_visualization(progress):
    """
    Create visualizations of optimization progress.
    
    Args:
        progress: Dictionary with progress metrics
        
    Returns:
        Plotly figure object
    """
    # Check if we have enough data points
    if len(progress['timestamps']) < 2:
        return None
    
    # Create a metrics progress plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Return %", "Max Drawdown %", "Win Rate %", "Sharpe Ratio"),
        shared_xaxes=True
    )
    
    # Add traces for each metric
    fig.add_trace(
        go.Scatter(
            x=progress['timestamps'], 
            y=progress['return_pct'],
            mode='lines+markers',
            name='Return %',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=progress['timestamps'], 
            y=progress['max_drawdown_pct'],
            mode='lines+markers',
            name='Max Drawdown %',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=progress['timestamps'], 
            y=progress['win_rate'],
            mode='lines+markers',
            name='Win Rate %',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=progress['timestamps'], 
            y=progress['sharpe_ratio'],
            mode='lines+markers',
            name='Sharpe Ratio',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Optimization Progress Over Time",
        height=500,
        showlegend=False
    )
    
    return fig

def create_parameter_evolution_chart(progress, params_to_show=None):
    """
    Create a chart showing how parameter values have evolved during optimization.
    
    Args:
        progress: Dictionary with progress metrics
        params_to_show: Optional list of parameters to include (default: all)
        
    Returns:
        Plotly figure object
    """
    # Filter parameters if needed
    parameter_evolution = progress['parameter_evolution']
    
    if params_to_show:
        parameter_evolution = {k: v for k, v in parameter_evolution.items() if k in params_to_show}
    
    # Keep only parameters with enough data points
    parameter_evolution = {k: v for k, v in parameter_evolution.items() if len(v['timestamps']) >= 2}
    
    if not parameter_evolution:
        return None
    
    # Create subplots for each parameter
    num_params = len(parameter_evolution)
    
    # Organize parameters in a grid (up to 3 columns)
    cols = min(3, num_params)
    rows = (num_params + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=list(parameter_evolution.keys())
    )
    
    # Add traces for each parameter
    for i, (param_name, param_data) in enumerate(parameter_evolution.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=param_data['timestamps'], 
                y=param_data['values'],
                mode='lines+markers',
                name=param_name
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title="Parameter Evolution Over Optimization Iterations",
        height=max(300, rows * 250),
        showlegend=False
    )
    
    return fig

def calculate_improvement_metrics(progress):
    """
    Calculate improvement metrics from optimization progress data.
    
    Args:
        progress: Dictionary with progress metrics
        
    Returns:
        Dictionary with improvement metrics
    """
    if not progress['return_pct']:
        return {}
    
    # Calculate improvement metrics
    initial_metrics = {
        'return_pct': progress['return_pct'][0] if progress['return_pct'] else None,
        'max_drawdown_pct': progress['max_drawdown_pct'][0] if progress['max_drawdown_pct'] else None,
        'win_rate': progress['win_rate'][0] if progress['win_rate'] else None,
        'sharpe_ratio': progress['sharpe_ratio'][0] if progress['sharpe_ratio'] else None
    }
    
    final_metrics = {
        'return_pct': progress['return_pct'][-1] if progress['return_pct'] else None,
        'max_drawdown_pct': progress['max_drawdown_pct'][-1] if progress['max_drawdown_pct'] else None,
        'win_rate': progress['win_rate'][-1] if progress['win_rate'] else None,
        'sharpe_ratio': progress['sharpe_ratio'][-1] if progress['sharpe_ratio'] else None
    }
    
    # Calculate absolute change
    abs_change = {
        'return_pct': final_metrics['return_pct'] - initial_metrics['return_pct'] if final_metrics['return_pct'] is not None and initial_metrics['return_pct'] is not None else None,
        'max_drawdown_pct': final_metrics['max_drawdown_pct'] - initial_metrics['max_drawdown_pct'] if final_metrics['max_drawdown_pct'] is not None and initial_metrics['max_drawdown_pct'] is not None else None,
        'win_rate': final_metrics['win_rate'] - initial_metrics['win_rate'] if final_metrics['win_rate'] is not None and initial_metrics['win_rate'] is not None else None,
        'sharpe_ratio': final_metrics['sharpe_ratio'] - initial_metrics['sharpe_ratio'] if final_metrics['sharpe_ratio'] is not None and initial_metrics['sharpe_ratio'] is not None else None
    }
    
    # Calculate percent change (avoiding division by zero)
    pct_change = {
        'return_pct': (abs_change['return_pct'] / initial_metrics['return_pct'] * 100) if abs_change['return_pct'] is not None and initial_metrics['return_pct'] and initial_metrics['return_pct'] != 0 else None,
        'max_drawdown_pct': (abs_change['max_drawdown_pct'] / initial_metrics['max_drawdown_pct'] * 100) if abs_change['max_drawdown_pct'] is not None and initial_metrics['max_drawdown_pct'] and initial_metrics['max_drawdown_pct'] != 0 else None,
        'win_rate': (abs_change['win_rate'] / initial_metrics['win_rate'] * 100) if abs_change['win_rate'] is not None and initial_metrics['win_rate'] and initial_metrics['win_rate'] != 0 else None,
        'sharpe_ratio': (abs_change['sharpe_ratio'] / initial_metrics['sharpe_ratio'] * 100) if abs_change['sharpe_ratio'] is not None and initial_metrics['sharpe_ratio'] and initial_metrics['sharpe_ratio'] != 0 else None
    }
    
    return {
        'initial': initial_metrics,
        'final': final_metrics,
        'abs_change': abs_change,
        'pct_change': pct_change
    }

def app():
    st.header("Optimization Progress Tracker")
    st.markdown("""
    Track the progress of optimization runs over time to see improvement in metrics
    and parameter evolution.
    """)
    
    # List saved optimization results
    all_results = results_manager.list_optimization_results(strategy_name="OptimizedAlternativeDCAv5")
    
    if all_results:
        # Group results into distinct optimization runs
        runs = identify_optimization_runs(all_results)
        
        if runs:
            # Allow user to select a run
            run_options = list(runs.keys())
            selected_run = st.selectbox("Select optimization run to analyze", run_options)
            
            if selected_run and selected_run in runs:
                run_info = runs[selected_run]
                
                # Display basic run info
                st.info(f"Run duration: {run_info['start_time'].strftime('%Y-%m-%d')} to {run_info['end_time'].strftime('%Y-%m-%d')}")
                st.write(f"Number of optimization batches: {len(run_info['results'])}")
                
                # Analyze progress
                progress = analyze_optimization_progress(run_info['results'])
                
                if progress['timestamps']:
                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(["Progress Metrics", "Parameter Evolution", "Improvement Summary"])
                    
                    with tab1:
                        # Create progress visualization
                        fig = create_progress_visualization(progress)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                            # Display data table
                            data = {
                                'Timestamp': [ts.strftime('%Y-%m-%d') for ts in progress['timestamps']],
                                'Return (%)': [f"{x:.2f}" for x in progress['return_pct']],
                                'Max Drawdown (%)': [f"{x:.2f}" for x in progress['max_drawdown_pct']],
                                'Win Rate (%)': [f"{x:.2f}" for x in progress['win_rate']],
                                'Sharpe Ratio': [f"{x:.2f}" for x in progress['sharpe_ratio']]
                            }
                            
                            progress_df = pd.DataFrame(data)
                            st.dataframe(progress_df)
                        else:
                            st.warning("Not enough data points to visualize progress.")
                    
                    with tab2:
                        # Parameter evolution
                        # Allow user to select parameters to display
                        all_params = list(progress['parameter_evolution'].keys())
                        
                        if all_params:
                            selected_params = st.multiselect(
                                "Select parameters to visualize",
                                all_params,
                                default=all_params[:min(6, len(all_params))]
                            )
                            
                            if selected_params:
                                param_fig = create_parameter_evolution_chart(progress, selected_params)
                                if param_fig:
                                    st.plotly_chart(param_fig, use_container_width=True)
                                else:
                                    st.warning("Not enough data points to visualize parameter evolution.")
                            
                            # Display best parameters found
                            st.subheader("Best Parameters Found")
                            if progress['best_parameters']:
                                best_params_df = pd.DataFrame([
                                    {'Parameter': k, 'Value': v}
                                    for k, v in progress['best_parameters'].items()
                                ])
                                st.dataframe(best_params_df)
                        else:
                            st.info("No parameter evolution data available.")
                    
                    with tab3:
                        # Improvement summary
                        st.subheader("Optimization Improvement Summary")
                        
                        # Calculate improvement metrics
                        improvement = calculate_improvement_metrics(progress)
                        
                        if improvement:
                            # Display metrics using columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Initial vs Final Metrics")
                                
                                # Create comparison table
                                comparison_data = {
                                    'Metric': ['Return (%)', 'Max Drawdown (%)', 'Win Rate (%)', 'Sharpe Ratio'],
                                    'Initial': [
                                        f"{improvement['initial']['return_pct']:.2f}" if improvement['initial']['return_pct'] is not None else "N/A",
                                        f"{improvement['initial']['max_drawdown_pct']:.2f}" if improvement['initial']['max_drawdown_pct'] is not None else "N/A",
                                        f"{improvement['initial']['win_rate']:.2f}" if improvement['initial']['win_rate'] is not None else "N/A",
                                        f"{improvement['initial']['sharpe_ratio']:.2f}" if improvement['initial']['sharpe_ratio'] is not None else "N/A"
                                    ],
                                    'Final': [
                                        f"{improvement['final']['return_pct']:.2f}" if improvement['final']['return_pct'] is not None else "N/A",
                                        f"{improvement['final']['max_drawdown_pct']:.2f}" if improvement['final']['max_drawdown_pct'] is not None else "N/A",
                                        f"{improvement['final']['win_rate']:.2f}" if improvement['final']['win_rate'] is not None else "N/A",
                                        f"{improvement['final']['sharpe_ratio']:.2f}" if improvement['final']['sharpe_ratio'] is not None else "N/A"
                                    ]
                                }
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df)
                            
                            with col2:
                                st.subheader("Improvement Metrics")
                                
                                # Create improvement table
                                improvement_data = {
                                    'Metric': ['Return (%)', 'Max Drawdown (%)', 'Win Rate (%)', 'Sharpe Ratio'],
                                    'Absolute Change': [
                                        f"{improvement['abs_change']['return_pct']:.2f}" if improvement['abs_change']['return_pct'] is not None else "N/A",
                                        f"{improvement['abs_change']['max_drawdown_pct']:.2f}" if improvement['abs_change']['max_drawdown_pct'] is not None else "N/A",
                                        f"{improvement['abs_change']['win_rate']:.2f}" if improvement['abs_change']['win_rate'] is not None else "N/A",
                                        f"{improvement['abs_change']['sharpe_ratio']:.2f}" if improvement['abs_change']['sharpe_ratio'] is not None else "N/A"
                                    ],
                                    'Percent Change (%)': [
                                        f"{improvement['pct_change']['return_pct']:.2f}" if improvement['pct_change']['return_pct'] is not None else "N/A",
                                        f"{improvement['pct_change']['max_drawdown_pct']:.2f}" if improvement['pct_change']['max_drawdown_pct'] is not None else "N/A",
                                        f"{improvement['pct_change']['win_rate']:.2f}" if improvement['pct_change']['win_rate'] is not None else "N/A",
                                        f"{improvement['pct_change']['sharpe_ratio']:.2f}" if improvement['pct_change']['sharpe_ratio'] is not None else "N/A"
                                    ]
                                }
                                
                                improvement_df = pd.DataFrame(improvement_data)
                                st.dataframe(improvement_df)
                            
                            # Create overall improvement summary
                            avg_improvement = np.mean([
                                improvement['pct_change']['return_pct'] if improvement['pct_change']['return_pct'] is not None else 0,
                                -improvement['pct_change']['max_drawdown_pct'] if improvement['pct_change']['max_drawdown_pct'] is not None else 0,  # Negative for drawdown (lower is better)
                                improvement['pct_change']['win_rate'] if improvement['pct_change']['win_rate'] is not None else 0,
                                improvement['pct_change']['sharpe_ratio'] if improvement['pct_change']['sharpe_ratio'] is not None else 0
                            ])
                            
                            # Display improvement summary
                            st.subheader("Overall Optimization Effectiveness")
                            
                            # Determine effectiveness level
                            if avg_improvement > 20:
                                effectiveness = "High"
                                color = "green"
                            elif avg_improvement > 5:
                                effectiveness = "Medium"
                                color = "orange"
                            else:
                                effectiveness = "Low"
                                color = "red"
                            
                            st.markdown(f"""
                            <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white;">
                            <h3 style="margin: 0;">Optimization Effectiveness: {effectiveness}</h3>
                            <p>Average improvement across metrics: {avg_improvement:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Not enough data to calculate improvement metrics.")
                else:
                    st.warning("No progress data found for this optimization run.")
            else:
                st.warning("Please select an optimization run to analyze.")
        else:
            st.warning("Could not identify distinct optimization runs from the saved results.")
    else:
        st.info("No saved optimization results found.")

if __name__ == "__main__":
    app()