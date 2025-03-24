import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.results_manager import ResultsManager
from visualization.performance_plots import PerformancePlots

# Initialize components
results_manager = ResultsManager()

def create_parameter_correlation_matrix(results_data: List[Dict]):
    """
    Create a correlation matrix between parameter values and performance metrics.
    
    Args:
        results_data: List of result dictionaries containing parameters and metrics
        
    Returns:
        DataFrame with correlation values
    """
    # Extract parameters and metrics into a single dataframe
    rows = []
    
    for result in results_data:
        row = {}
        # Add parameters
        for param_name, param_value in result['parameters'].items():
            if isinstance(param_value, (int, float, bool)):
                row[param_name] = float(param_value) if not isinstance(param_value, bool) else int(param_value)
        
        # Add performance metrics
        for metric_name, metric_value in result['metrics'].items():
            if isinstance(metric_value, (int, float)):
                row[f"m_{metric_name}"] = metric_value
        
        rows.append(row)
    
    # Create dataframe
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr(method='pearson', numeric_only=True)
    
    # Filter to show only parameter-metric correlations
    param_cols = [col for col in correlation_matrix.columns if not col.startswith('m_')]
    metric_cols = [col for col in correlation_matrix.columns if col.startswith('m_')]
    
    return correlation_matrix.loc[param_cols, metric_cols]

def get_cross_dataset_robustness(results_by_dataset):
    """
    Analyze parameters across multiple datasets to find robust configurations.
    
    Args:
        results_by_dataset: Dictionary mapping dataset names to result lists
        
    Returns:
        DataFrame with robustness metrics
    """
    # Get all parameter names across all datasets
    all_params = set()
    
    for dataset_name, results in results_by_dataset.items():
        for result in results:
            all_params.update(result.get('parameters', {}).keys())
    
    # For each parameter, analyze consistency across datasets
    param_robustness = {}
    for param in all_params:
        dataset_values = {}
        
        # Get best parameter value for each dataset
        for dataset_name, results in results_by_dataset.items():
            if not results:
                continue
                
            # Find top 3 results by return percentage
            top_results = sorted(results, key=lambda x: x.get('metrics', {}).get('return_pct', 0), reverse=True)[:3]
            
            # Get parameter values from top results
            values = [r.get('parameters', {}).get(param) for r in top_results if param in r.get('parameters', {})]
            if values:
                dataset_values[dataset_name] = values
        
        # Skip parameters without enough data
        if len(dataset_values) < 2:
            continue
            
        # Calculate mean value for each dataset
        dataset_means = {ds: np.mean(vals) if all(isinstance(v, (int, float)) for v in vals) else None 
                         for ds, vals in dataset_values.items()}
        
        # Filter out None values
        dataset_means = {ds: val for ds, val in dataset_means.items() if val is not None}
        
        # Skip if not enough numeric values
        if len(dataset_means) < 2:
            continue
            
        # Calculate robustness metrics
        mean_value = np.mean(list(dataset_means.values()))
        min_value = min(dataset_means.values())
        max_value = max(dataset_means.values())
        
        # Coefficient of variation - lower is better
        std_value = np.std(list(dataset_means.values()))
        cv = (std_value / abs(mean_value)) if mean_value != 0 else float('inf')
        
        # Calculate agreement score (0-100)
        agreement_score = max(0, min(100, 100 - (cv * 100)))
        
        param_robustness[param] = {
            'mean': mean_value,
            'min': min_value,
            'max': max_value,
            'std': std_value,
            'cv': cv,
            'agreement': agreement_score,
            'datasets': len(dataset_means)
        }
    
    # Convert to DataFrame
    robustness_df = pd.DataFrame.from_dict(param_robustness, orient='index')
    
    # Sort by agreement score (descending)
    return robustness_df.sort_values('agreement', ascending=False)

def app():
    st.header("Cross-Dataset Comparison")
    st.markdown("""
    Compare optimization results across different datasets to find robust parameters 
    that perform well across market conditions.
    """)
    
    # List saved optimization results
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
                    
                    # Add flattened results
                    for batch_result in loaded_result['results']:
                        results_by_dataset[dataset].append(batch_result)
            except:
                # Skip results that can't be loaded
                pass
        
        datasets = list(results_by_dataset.keys())
        if len(datasets) > 1:
            # Allow user to select datasets to compare
            selected_datasets = st.multiselect(
                "Select datasets to compare", 
                datasets, 
                default=datasets[:min(3, len(datasets))]
            )
            
            if selected_datasets:
                # Create tabs for different analyses
                tab1, tab2, tab3 = st.tabs(["Parameter Comparison", "Correlation Analysis", "Robustness Score"])
                
                # Filter to selected datasets
                filtered_results = {ds: results_by_dataset[ds] for ds in selected_datasets if ds in results_by_dataset}
                
                with tab1:
                    st.subheader("Best Parameters By Dataset")
                    
                    # Collect best parameters for each dataset
                    comparison_data = []
                    all_params = set()
                    
                    for dataset in selected_datasets:
                        dataset_results = filtered_results.get(dataset, [])
                        
                        if not dataset_results:
                            continue
                            
                        # Find best result for this dataset
                        best_return = 0
                        best_params = None
                        best_metrics = None
                        
                        for result in dataset_results:
                            if result['metrics']['return_pct'] > best_return:
                                best_return = result['metrics']['return_pct']
                                best_params = result['parameters']
                                best_metrics = result['metrics']
                        
                        if best_params:
                            # Add to comparison data
                            comparison_data.append({
                                'dataset': dataset,
                                'params': best_params,
                                'metrics': best_metrics
                            })
                            # Track all parameters
                            all_params.update(best_params.keys())
                    
                    if comparison_data:
                        # Create comparison table
                        st.subheader("Best Parameters Comparison")
                        
                        # Create DataFrame with parameters and metrics
                        rows = []
                        for data in comparison_data:
                            row = {'Dataset': data['dataset']}
                            # Add metrics
                            row['Return (%)'] = f"{data['metrics']['return_pct']:.2f}"
                            row['Max DD (%)'] = f"{data['metrics']['max_drawdown_pct']:.2f}"
                            row['Win Rate (%)'] = f"{data['metrics']['win_rate']:.2f}"
                            row['Sharpe'] = f"{data['metrics']['sharpe_ratio']:.2f}"
                            
                            # Add parameters
                            for param in sorted(all_params):
                                if param in data['params']:
                                    row[param] = data['params'][param]
                                else:
                                    row[param] = "N/A"
                            
                            rows.append(row)
                        
                        comparison_df = pd.DataFrame(rows)
                        st.dataframe(comparison_df)
                        
                        # Parameter variability visualization
                        st.subheader("Parameter Variability Across Datasets")
                        
                        # Create DataFrame for parameter ranges
                        param_ranges = []
                        for param in sorted(all_params):
                            values = [data['params'].get(param) for data in comparison_data if param in data['params']]
                            if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
                                param_ranges.append({
                                    'Parameter': param,
                                    'Min': min(values),
                                    'Mean': np.mean(values),
                                    'Max': max(values),
                                    'Range (%)': (max(values) - min(values)) / np.mean(values) * 100 if np.mean(values) != 0 else 0
                                })
                        
                        if param_ranges:
                            # Sort by range percentage
                            param_ranges.sort(key=lambda x: x['Range (%)'], reverse=True)
                            
                            # Create visualization
                            fig = go.Figure()
                            
                            for i, param in enumerate(param_ranges):
                                fig.add_trace(go.Scatter(
                                    x=[param['Min'], param['Mean'], param['Max']],
                                    y=[i, i, i],
                                    mode='markers+lines',
                                    name=param['Parameter'],
                                    text=[f"Min: {param['Min']:.2f}", f"Mean: {param['Mean']:.2f}", f"Max: {param['Max']:.2f}"],
                                    marker=dict(size=[8, 12, 8])
                                ))
                            
                            # Update layout
                            fig.update_layout(
                                title="Parameter Value Ranges Across Datasets",
                                xaxis_title="Parameter Value",
                                yaxis=dict(
                                    tickmode='array',
                                    tickvals=list(range(len(param_ranges))),
                                    ticktext=[p['Parameter'] for p in param_ranges]
                                ),
                                height=max(400, len(param_ranges) * 30),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    st.subheader("Parameter Performance Correlation")
                    
                    # Create correlation analyses for each dataset
                    for dataset in selected_datasets:
                        dataset_results = filtered_results.get(dataset, [])
                        
                        if not dataset_results:
                            continue
                            
                        st.subheader(f"Correlation Analysis: {dataset}")
                        
                        # Create correlation matrix
                        corr_matrix = create_parameter_correlation_matrix(dataset_results)
                        
                        if not corr_matrix.empty:
                            # Clean up metric names for display
                            corr_matrix.columns = [col.replace('m_', '') for col in corr_matrix.columns]
                            
                            # Display correlation matrix
                            fig = px.imshow(
                                corr_matrix,
                                color_continuous_scale='RdBu_r',
                                labels=dict(x="Performance Metric", y="Parameter", color="Correlation"),
                                zmin=-1, zmax=1
                            )
                            
                            fig.update_layout(
                                title=f"Parameter-Performance Correlation: {dataset}",
                                height=max(400, len(corr_matrix) * 25)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Highlight top correlations
                            st.subheader("Top Parameter Correlations")
                            
                            # Melt the correlation matrix to get parameter-metric pairs
                            melted = pd.melt(
                                corr_matrix.reset_index(), 
                                id_vars='index',
                                var_name='Metric',
                                value_name='Correlation'
                            )
                            melted.columns = ['Parameter', 'Metric', 'Correlation']
                            
                            # Get absolute correlation and sort
                            melted['Abs_Correlation'] = melted['Correlation'].abs()
                            melted = melted.sort_values('Abs_Correlation', ascending=False)
                            
                            # Display top correlations
                            st.dataframe(melted.head(10)[['Parameter', 'Metric', 'Correlation']])
                        else:
                            st.info("Not enough data for correlation analysis.")
                
                with tab3:
                    st.subheader("Parameter Robustness Analysis")
                    
                    # Calculate robustness scores
                    robustness_df = get_cross_dataset_robustness(filtered_results)
                    
                    if not robustness_df.empty:
                        # Format for display
                        display_df = robustness_df.copy()
                        display_df['agreement'] = display_df['agreement'].map(lambda x: f"{x:.1f}%")
                        display_df['cv'] = display_df['cv'].map(lambda x: f"{x:.3f}")
                        
                        # Display robustness table
                        st.dataframe(display_df[['mean', 'min', 'max', 'agreement', 'datasets']])
                        
                        # Visualize agreement scores
                        st.subheader("Parameter Agreement Visualization")
                        
                        # Create agreement bar chart
                        param_names = robustness_df.index.tolist()
                        agreement_values = robustness_df['agreement'].values
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=param_names,
                            y=agreement_values,
                            marker_color=['green' if a > 75 else 'orange' if a > 50 else 'red' for a in agreement_values],
                            text=[f"{a:.1f}%" for a in agreement_values],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title="Parameter Agreement Across Datasets",
                            xaxis_title="Parameter",
                            yaxis_title="Agreement (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Generate recommended robust parameters
                        st.subheader("Recommended Robust Parameters")
                        
                        # Use parameters with high agreement
                        robust_params = {param: row['mean'] 
                                        for param, row in robustness_df.iterrows() 
                                        if row['agreement'] > 50}
                        
                        # Display recommended parameters
                        robust_df = pd.DataFrame([{'Parameter': k, 'Value': v} for k, v in robust_params.items()])
                        st.dataframe(robust_df)
                        
                        # Option to run backtest with robust parameters
                        # (Omitted because needs the main backtest function from auto_optimization)
                    else:
                        st.info("Not enough data across datasets for robustness analysis.")
            else:
                st.info("Please select at least one dataset to compare.")
        else:
            st.info("Need at least two datasets with optimization results for comparison.")
    else:
        st.info("No saved optimization results found.")

if __name__ == "__main__":
    app()