import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# Import project modules
from utils.results_manager import ResultsManager
from utils.meta_analysis import MetaAnalysis

# Initialize ResultsManager
results_manager = ResultsManager()

# Configure page
st.set_page_config(
    page_title="Optimization Analysis",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("Parameter Optimization Analysis")

# Sidebar for enhanced controls
with st.sidebar:
    st.header("Analysis Options")
    
    # Advanced analysis options
    st.subheader("Advanced Analysis")
    
    view_mode = st.radio(
        "Analysis View Mode",
        ["Single Result Analysis", "Cross-Dataset Comparison", "Parameter Importance", "Evolution Analysis"]
    )
    
    # Performance metric to optimize for
    primary_metric = st.selectbox(
        "Primary Optimization Metric",
        ["return_pct", "sharpe_ratio", "calmar_ratio", "sortino_ratio", "win_rate", "profit_factor"],
        index=0,
        help="Select the primary metric that will be used for ranking optimization results"
    )
    
    # Advanced display options
    st.subheader("Display Options")
    
    # Option for how many top results to show
    top_n_results = st.slider(
        "Top Results to Show",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of top results to display in tables and charts"
    )
    
    # Outlier filtering
    filter_outliers = st.checkbox(
        "Filter Extreme Outliers",
        value=True,
        help="Remove extreme outlier results that might skew the analysis"
    )
    
    # Advanced correlation analysis
    enable_advanced_correlation = st.checkbox(
        "Enable Advanced Correlation Analysis",
        value=True,
        help="Perform deeper analysis of parameter interactions and correlations"
    )

# Sidebar: Parameter descriptions
with st.sidebar:
    st.header("Parameter Explanations")
    
    with st.expander("Investment Percentage"):
        st.write("""
        **Investment Percentage** determines what percentage of available cash to use for each buy.
        
        - **Higher values** (e.g., 20-40%): More aggressive investment style, deploys capital faster, potentially higher returns in strong uptrends but may run out of capital sooner
        - **Lower values** (e.g., 5-10%): More conservative approach, deploys capital more gradually, better for prolonged downtrends
        
        Optimal range is typically 10-20% for balanced performance.
        """)
    
    with st.expander("Price Drop Threshold"):
        st.write("""
        **Price Drop Threshold** sets the minimum price decrease (%) required to trigger a buy.
        
        - **Higher values** (e.g., 3-5%): More selective buying on larger drops, fewer trades, potentially better entry prices
        - **Lower values** (e.g., 0.5-1%): More frequent buying on smaller dips, more trades, better for capturing small fluctuations
        
        Optimal range is typically 1.5-3% for most market conditions.
        """)
    
    with st.expander("Profit Threshold"):
        st.write("""
        **Profit Threshold** is the target profit percentage that triggers selling.
        
        - **Higher values** (e.g., 5-10%): Holds positions longer for larger gains, potentially missing profit-taking opportunities
        - **Lower values** (e.g., 1-2%): Takes profits more quickly, higher turnover, better for ranging markets
        
        Optimal range is typically 2-4% to balance profit-taking with holding through minor fluctuations.
        """)
    
    with st.expander("Stop Loss Fixed Percentage"):
        st.write("""
        **Stop Loss Fixed Percentage** sets the maximum loss before exiting a position.
        
        - **Higher values** (e.g., 20-25%): More room for price fluctuations, fewer stopped out trades, but potentially larger losses
        - **Lower values** (e.g., 10-15%): Tighter risk control, smaller maximum loss, but may get stopped out during normal volatility
        
        Optimal range is typically 15-20% for most cryptocurrency markets.
        """)
    
    with st.expander("Trailing Stop Percentage"):
        st.write("""
        **Trailing Stop Percentage** is used to lock in profits as price moves upward.
        
        - **Higher values** (e.g., 2-3%): More room for price to fluctuate before selling, potentially capturing larger uptrends
        - **Lower values** (e.g., 0.5-1%): Locks in profits more aggressively, better at capturing local tops
        
        Optimal range is typically 1.2-2% based on Bitcoin volatility patterns.
        """)
    
    with st.expander("Maximum Position Duration"):
        st.write("""
        **Maximum Position Duration** sets the maximum number of days to hold a position.
        
        - **Higher values** (e.g., 180-360 days): More patience with positions, better for long-term accumulation
        - **Lower values** (e.g., 30-90 days): Faster capital turnover, forces exit from stagnant positions
        
        Optimal range is typically 150-210 days for balanced performance.
        """)
    
    with st.expander("Number of Bots"):
        st.write("""
        **Number of Bots** determines how many separate trading bots to run simultaneously.
        
        - **Higher values** (e.g., 3-5 bots): More diversified entries, smoother equity curve, better risk distribution
        - **Lower values** (e.g., 1-2 bots): Simpler management, more concentrated capital deployment
        
        Optimal range is typically 2-3 bots for most strategies.
        """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Parameter Impact Analysis", "Correlation Analysis", "Performance Distribution", "Meta-Analysis"])

# Meta-Analysis tab implementation (defined here, shown when tab4 is active)
def show_meta_analysis():
    st.header("Meta-Analysis of All Optimization Results")
    
    # Fetch all optimization results
    with st.spinner("Loading all optimization results..."):
        all_optimization_files = results_manager.list_optimization_results()
        
        if not all_optimization_files:
            st.warning("No optimization results found.")
        else:
            # Load and process all optimization results
            all_results, all_meta_data, errors = MetaAnalysis.load_and_process_results(
                results_manager, all_optimization_files)
            
            # Display any errors encountered
            if errors:
                st.error(f"Encountered {len(errors)} errors while loading results:")
                for error in errors[:5]:  # Show only first 5 errors to avoid clutter
                    st.error(error)
            
            # Convert results to DataFrame for analysis
            if all_results:
                meta_df = pd.DataFrame(all_meta_data)
                results_meta_df = pd.DataFrame(all_results)
                
                # Display metadata summary
                st.subheader("Optimization Test Overview")
                st.dataframe(meta_df)
                
                # Select which parameters and metrics to analyze
                st.subheader("Parameter Performance Across All Tests")
                
                # Find parameter and metric columns
                param_cols = [col for col in results_meta_df.columns if col.startswith('param_')]
                metric_cols = [col for col in results_meta_df.columns if col.startswith('metric_')]
                
                # Find common parameters across multiple tests
                common_params = MetaAnalysis.find_common_parameters(results_meta_df)
                
                if not common_params:
                    st.warning("No common parameters found across multiple tests.")
                else:
                    # Select parameter for analysis
                    param_labels = [col.replace('param_', '') for col in common_params]
                    selected_meta_param = st.selectbox("Select parameter to analyze", param_labels, key="meta_param")
                    selected_meta_param_col = f"param_{selected_meta_param}"
                    
                    # Select metric for analysis
                    metric_labels = [col.replace('metric_', '') for col in metric_cols]
                    selected_meta_metric = st.selectbox(
                        "Select metric to analyze", 
                        metric_labels, 
                        index=metric_labels.index('return_pct') if 'metric_return_pct' in metric_cols else 0, 
                        key="meta_metric"
                    )
                    selected_meta_metric_col = f"metric_{selected_meta_metric}"
                    
                    # Filter rows where the selected parameter exists
                    filtered_df = results_meta_df.dropna(subset=[selected_meta_param_col])
                    
                    if filtered_df.empty:
                        st.warning(f"No data available for parameter '{selected_meta_param}'.")
                    else:
                        # Create box plot by dataset
                        st.subheader(f"Performance of {selected_meta_param} across datasets")
                        
                        # Group by dataset and parameter value
                        fig = MetaAnalysis.parameter_performance_across_datasets(
                            filtered_df, selected_meta_param_col, selected_meta_metric_col)
                        st.plotly_chart(fig)
                        
                        # Robust parameter analysis
                        st.subheader("Robust Parameter Analysis")
                        
                        # Group by parameter value and calculate statistics across all datasets
                        param_stats = MetaAnalysis.calculate_parameter_stats(
                            filtered_df, selected_meta_param_col, selected_meta_metric_col)
                        
                        # Display statistics
                        st.dataframe(param_stats)
                        
                        # Bar chart of parameter performance
                        fig = MetaAnalysis.parameter_performance_chart(
                            param_stats, selected_meta_param_col, selected_meta_metric.replace('_', ' '))
                        st.plotly_chart(fig)
                        
                        # Show most robust parameter value
                        robust_row = param_stats[param_stats['cv'] == param_stats['cv'].min()]
                        best_row = param_stats[param_stats['mean'] == param_stats['mean'].max()]
                        
                        if not robust_row.empty and not best_row.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.info(f"""
                                **Most Robust Value:** {robust_row.iloc[0][selected_meta_param_col]}
                                - Mean {selected_meta_metric}: {robust_row.iloc[0]['mean']:.2f}
                                - Coefficient of Variation: {robust_row.iloc[0]['cv']:.2f}
                                - Tested {int(robust_row.iloc[0]['count'])} times
                                """)
                                
                            with col2:
                                st.success(f"""
                                **Best Performing Value:** {best_row.iloc[0][selected_meta_param_col]}
                                - Mean {selected_meta_metric}: {best_row.iloc[0]['mean']:.2f}
                                - Standard Deviation: {best_row.iloc[0]['std']:.2f}
                                - Tested {int(best_row.iloc[0]['count'])} times
                                """)
                        
                        # Parameter consistency analysis across datasets
                        st.subheader("Parameter Consistency Analysis")
                        
                        # Calculate top performing parameter value for each dataset
                        dataset_best_params = MetaAnalysis.find_best_parameters_per_dataset(
                            filtered_df, selected_meta_param_col, selected_meta_metric_col)
                        
                        # Display results
                        if dataset_best_params:
                            best_param_df = pd.DataFrame(list(dataset_best_params.items()), columns=['Dataset', f'Best {selected_meta_param} Value'])
                            st.write(f"Best {selected_meta_param} value for each dataset:")
                            st.dataframe(best_param_df)
                            
                            # Check agreement
                            if len(set(dataset_best_params.values())) == 1:
                                st.success(f"**Perfect Agreement!** The value {list(dataset_best_params.values())[0]} performs best across all datasets.")
                            else:
                                # Count occurrences of each best value
                                value_counts = {}
                                for value in dataset_best_params.values():
                                    if value in value_counts:
                                        value_counts[value] += 1
                                    else:
                                        value_counts[value] = 1
                                
                                # Find most common best value
                                most_common_value = max(value_counts.items(), key=lambda x: x[1])
                                agreement_pct = most_common_value[1] / len(dataset_best_params) * 100
                                
                                st.info(f"""
                                **Partial Agreement:** The value {most_common_value[0]} performs best in {most_common_value[1]} 
                                out of {len(dataset_best_params)} datasets ({agreement_pct:.1f}% agreement).
                                """)
                
                # Parameter importance analysis
                st.subheader("Overall Parameter Importance Analysis")
                
                # Calculate parameter importance based on correlations
                importance_data = MetaAnalysis.calculate_parameter_importance(
                    results_meta_df, common_params, metric_cols)
                
                if importance_data:
                    # Create heatmap of parameter-metric correlations
                    fig = MetaAnalysis.create_correlation_heatmap(importance_data['correlations'])
                    st.plotly_chart(fig)
                    
                    # Bar chart of parameter importance
                    fig = MetaAnalysis.create_importance_chart(importance_data['importance'])
                    st.plotly_chart(fig)
                    
                    # Display top parameters
                    st.subheader("Most Important Parameters")
                    st.write("Parameters with the strongest influence on performance metrics:")
                    
                    # Show top 5 parameters or all if less than 5
                    top_n = min(5, len(importance_data['importance']))
                    for i in range(top_n):
                        param = importance_data['importance'].iloc[i]['Parameter']
                        score = importance_data['importance'].iloc[i]['Abs_Correlation']
                        st.write(f"{i+1}. **{param}** - Importance Score: {score:.3f}")
                else:
                    st.warning("Not enough data to calculate parameter correlations.")
            else:
                st.warning("No valid optimization results could be loaded.")

with tab1:
    st.header("Parameter Impact Analysis")
    
    # Load optimization results
    st.subheader("Select Optimization Results")
    
    # Get list of available optimization results
    optimization_files = results_manager.list_optimization_results()
    
    if optimization_files:
        # Options for selectbox - handle missing keys gracefully
        options = []
        for meta in optimization_files:
            strategy = meta.get('strategy', 'Unknown Strategy')
            timestamp = meta.get('timestamp', 'Unknown Date')
            options.append(f"{timestamp} - {strategy}")
            
        selected_result = st.selectbox("Select optimization results", options)
        
        # Get selected result file path
        selected_idx = options.index(selected_result)
        result_file = optimization_files[selected_idx]['filepath']
        
        # Load result data
        with st.spinner("Loading optimization results..."):
            result_data = results_manager.load_optimization_results(result_file)
            
            if 'results' in result_data and result_data['results']:
                # Convert results to DataFrame for analysis
                results_df = pd.DataFrame([
                    {
                        **{f"param_{k}": v for k, v in r['parameters'].items()},
                        **{f"metric_{k}": v for k, v in r['metrics'].items()}
                    }
                    for r in result_data['results']
                ])
                
                # Parameter impact visualization
                st.subheader("Parameter Impact on Performance")
                
                # Select parameters and metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    param_options = [col for col in results_df.columns if col.startswith('param_')]
                    param_labels = [col.replace('param_', '') for col in param_options]
                    selected_param = st.selectbox("Select parameter", param_labels)
                    selected_param_col = f"param_{selected_param}"
                
                with col2:
                    metric_options = [col for col in results_df.columns if col.startswith('metric_')]
                    metric_labels = [col.replace('metric_', '') for col in metric_options]
                    selected_metric = st.selectbox("Select metric", metric_labels, index=metric_labels.index('return_pct') if 'return_pct' in metric_labels else 0)
                    selected_metric_col = f"metric_{selected_metric}"
                
                # Create parameter impact plot
                if selected_param_col in results_df.columns and selected_metric_col in results_df.columns:
                    # Check if we have enough unique values for the parameter
                    unique_values = results_df[selected_param_col].nunique()
                    
                    if unique_values >= 3:
                        # Create box plot using Seaborn and Matplotlib
                        plt.figure(figsize=(10, 6))
                        sns.set_style("darkgrid")
                        
                        # Sort parameter values (important for proper visualization)
                        param_values = sorted(results_df[selected_param_col].unique())
                        
                        # Create box plot with individual points
                        ax = sns.boxplot(x=selected_param_col, y=selected_metric_col, data=results_df, order=param_values)
                        sns.stripplot(x=selected_param_col, y=selected_metric_col, data=results_df, color="black", alpha=0.5, order=param_values)
                        
                        # Set titles and labels
                        plt.title(f"Impact of {selected_param} on {selected_metric}", fontsize=14)
                        plt.xlabel(selected_param, fontsize=12)
                        plt.ylabel(selected_metric, fontsize=12)
                    else:
                        # Not enough unique values, create alternative visualization
                        st.warning(f"Not enough unique values for {selected_param} to create a meaningful box plot.")
                        
                        # Show available values
                        st.subheader(f"Available Values for {selected_param}")
                        unique_param_values = sorted(results_df[selected_param_col].unique())
                        
                        for val in unique_param_values:
                            # For each parameter value, show the corresponding metric values
                            filtered_df = results_df[results_df[selected_param_col] == val]
                            avg_metric = filtered_df[selected_metric_col].mean()
                            min_metric = filtered_df[selected_metric_col].min()
                            max_metric = filtered_df[selected_metric_col].max()
                            
                            st.write(f"**{selected_param} = {val}**")
                            st.write(f"- Average {selected_metric}: {avg_metric:.2f}")
                            st.write(f"- Range: {min_metric:.2f} to {max_metric:.2f}")
                        
                        # Create bar chart for limited values
                        plt.figure(figsize=(10, 6))
                        
                        # Calculate average metric value per parameter value
                        param_impact = results_df.groupby(selected_param_col)[selected_metric_col].mean().reset_index()
                        
                        # Create bar chart
                        plt.bar(
                            [str(val) for val in param_impact[selected_param_col]], 
                            param_impact[selected_metric_col],
                            color='skyblue'
                        )
                        
                        # Add data labels
                        for i, val in enumerate(param_impact[selected_metric_col]):
                            plt.text(i, val + 0.01 * (param_impact[selected_metric_col].max()), 
                                    f"{val:.2f}", ha='center')
                        
                        # Set titles and labels
                        plt.title(f"Average {selected_metric} by {selected_param}", fontsize=14)
                        plt.xlabel(selected_param, fontsize=12)
                        plt.ylabel(f"Average {selected_metric}", fontsize=12)
                        plt.ylim(0, param_impact[selected_metric_col].max() * 1.2)
                        
                        # Add trend line
                        sns.regplot(x=results_df[selected_param_col].astype(float), y=results_df[selected_metric_col], 
                                   scatter=False, ci=None, color='red')
                        
                        # Save figure to bytes buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        # Display image
                        st.image(Image.open(buf))
                        
                        # Interpretation
                        st.subheader("Interpretation")
                        
                        # Calculate correlation and best value
                        correlation = results_df[[selected_param_col, selected_metric_col]].corr().iloc[0, 1]
                        
                        # Group by parameter and calculate average metric value
                        param_impact = results_df.groupby(selected_param_col)[selected_metric_col].mean().reset_index()
                        
                        # Find optimal parameter value (for maximizing the metric)
                        if selected_metric in ['return_pct', 'win_rate', 'profit_factor', 'sharpe_ratio']:
                            # For metrics where higher is better
                            best_value = param_impact.loc[param_impact[selected_metric_col].idxmax()][selected_param_col]
                            worst_value = param_impact.loc[param_impact[selected_metric_col].idxmin()][selected_param_col]
                            best_metric = param_impact[selected_metric_col].max()
                            worst_metric = param_impact[selected_metric_col].min()
                        else:
                            # For metrics where lower is better (like drawdown)
                            best_value = param_impact.loc[param_impact[selected_metric_col].idxmin()][selected_param_col]
                            worst_value = param_impact.loc[param_impact[selected_metric_col].idxmax()][selected_param_col]
                            best_metric = param_impact[selected_metric_col].min()
                            worst_metric = param_impact[selected_metric_col].max()
                        
                        # Relationship description
                        if abs(correlation) > 0.7:
                            relationship = "strong"
                        elif abs(correlation) > 0.3:
                            relationship = "moderate"
                        else:
                            relationship = "weak"
                        
                        if correlation > 0:
                            direction = "positive"
                        else:
                            direction = "negative"
                        
                        # Write interpretation
                        st.write(f"""
                        **Correlation: {correlation:.2f}** - There is a {relationship} {direction} correlation between {selected_param} and {selected_metric}.
                        
                        **Optimal value: {best_value}** - This value produced the best average {selected_metric} of {best_metric:.2f}.
                        
                        **Worst value: {worst_value}** - This value produced the worst average {selected_metric} of {worst_metric:.2f}.
                        """)
                        
                        # Parameter recommendation
                        st.info(f"**Recommendation:** Based on this analysis, a {selected_param} value of **{best_value}** is optimal for maximizing {selected_metric}.")
                else:
                    st.error("Selected parameter or metric not found in the results.")
                
                # Heat map visualization for two parameters
                st.subheader("Parameter Interaction Analysis")
                
                # Select two parameters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    param1_options = [col for col in results_df.columns if col.startswith('param_')]
                    param1_labels = [col.replace('param_', '') for col in param1_options]
                    selected_param1 = st.selectbox("Select first parameter", param1_labels, key="param1")
                    selected_param1_col = f"param_{selected_param1}"
                
                with col2:
                    param2_options = [col for col in results_df.columns if col.startswith('param_') and col != selected_param1_col]
                    param2_labels = [col.replace('param_', '') for col in param2_options]
                    selected_param2 = st.selectbox("Select second parameter", param2_labels, key="param2")
                    selected_param2_col = f"param_{selected_param2}"
                
                with col3:
                    metric_options = [col for col in results_df.columns if col.startswith('metric_')]
                    metric_labels = [col.replace('metric_', '') for col in metric_options]
                    selected_metric2 = st.selectbox("Select metric", metric_labels, index=metric_labels.index('return_pct') if 'return_pct' in metric_labels else 0, key="metric2")
                    selected_metric2_col = f"metric_{selected_metric2}"
                
                # Create heatmap
                if (selected_param1_col in results_df.columns and 
                    selected_param2_col in results_df.columns and 
                    selected_metric2_col in results_df.columns):
                    
                    # Check if we have enough unique values
                    unique_values1 = results_df[selected_param1_col].nunique()
                    unique_values2 = results_df[selected_param2_col].nunique()
                    
                    if unique_values1 >= 2 and unique_values2 >= 2:
                        # Create pivot table
                        heatmap_data = results_df.pivot_table(
                            index=selected_param1_col, 
                            columns=selected_param2_col, 
                            values=selected_metric2_col,
                            aggfunc='mean'
                        )
                        
                        # Create heatmap
                        fig = px.imshow(
                            heatmap_data,
                            labels=dict(
                                x=selected_param2,
                                y=selected_param1,
                                color=selected_metric2
                            ),
                            x=sorted(results_df[selected_param2_col].unique()),
                            y=sorted(results_df[selected_param1_col].unique()),
                            color_continuous_scale="RdBu_r" if selected_metric2 in ['max_drawdown_pct'] else "Blues",
                            aspect="auto"
                        )
                        
                        # Add values to heatmap cells
                        for i in range(len(heatmap_data.index)):
                            for j in range(len(heatmap_data.columns)):
                                try:
                                    value = heatmap_data.iloc[i, j]
                                    if not np.isnan(value):
                                        fig.add_annotation(
                                            x=heatmap_data.columns[j],
                                            y=heatmap_data.index[i],
                                            text=f"{value:.2f}",
                                            showarrow=False,
                                            font=dict(color="white" if abs(value) > (heatmap_data.max().max() / 2) else "black")
                                        )
                                except:
                                    pass
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Interaction Effect of {selected_param1} and {selected_param2} on {selected_metric2}",
                            width=800,
                            height=600
                        )
                        
                        # Display figure
                        st.plotly_chart(fig)
                        
                        # Find best combination
                        if not heatmap_data.empty:
                            # For metrics where higher is better
                            if selected_metric2 in ['return_pct', 'win_rate', 'profit_factor', 'sharpe_ratio']:
                                best_idx = heatmap_data.stack().idxmax()
                                best_value = heatmap_data.stack().max()
                                optimize_for = "maximize"
                            else:
                                # For metrics where lower is better
                                best_idx = heatmap_data.stack().idxmin()
                                best_value = heatmap_data.stack().min()
                                optimize_for = "minimize"
                            
                            if isinstance(best_idx, tuple) and len(best_idx) == 2:
                                best_param1, best_param2 = best_idx
                                
                                st.info(f"""
                                **Best Parameter Combination:**
                                - {selected_param1}: **{best_param1}**
                                - {selected_param2}: **{best_param2}**
                                - {selected_metric2}: **{best_value:.2f}**
                                
                                This combination gives the optimal {selected_metric2} value.
                                """)
                    else:
                        st.warning("Not enough unique values for the selected parameters to create a meaningful heatmap.")
                else:
                    st.error("Selected parameters or metric not found in the results.")
            else:
                st.error("No results found in the selected file.")
    else:
        st.warning("No optimization results found. Run optimization first.")

with tab2:
    st.header("Parameter Correlation Analysis")
    
    # Load optimization results
    if 'results_df' in locals() and not results_df.empty:
        # Calculate correlation matrix between parameters and metrics
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        metric_cols = [col for col in results_df.columns if col.startswith('metric_')]
        
        # Create a copy of the data for correlation analysis
        corr_df = pd.DataFrame()
        
        # Extract parameter columns (only numeric)
        for param in param_cols:
            # Skip non-numeric parameters
            if pd.api.types.is_numeric_dtype(results_df[param]):
                corr_df[param.replace('param_', '')] = results_df[param]
        
        # Extract metric columns
        for metric in metric_cols:
            corr_df[metric.replace('metric_', '')] = results_df[metric]
        
        # Calculate correlation matrix
        correlation_df = corr_df.corr()
        
        # Get parameter and metric names
        param_names = [col for col in corr_df.columns if col in [p.replace('param_', '') for p in param_cols]]
        metric_names = [col for col in corr_df.columns if col in [m.replace('metric_', '') for m in metric_cols]]
        
        # Filter to show only parameter-metric correlations
        param_metric_corr = correlation_df.loc[param_names, metric_names]
        
        # Create tabs for different correlation views
        corr_tab1, corr_tab2, corr_tab3 = st.tabs(["Parameter-Metric Correlation", "Parameter Importance", "Parameter Interactions"])
        
        with corr_tab1:
            # Create standard heatmap
            fig = px.imshow(
                param_metric_corr,
                labels=dict(
                    x="Metrics",
                    y="Parameters",
                    color="Correlation"
                ),
                x=[col.replace('metric_', '') for col in metric_cols],
                y=[col.replace('param_', '') for col in param_cols],
                color_continuous_scale="RdBu",
                aspect="auto",
                range_color=[-1, 1]
            )
            
            # Add values to heatmap cells
            for i in range(len(param_names)):
                for j in range(len(metric_names)):
                    fig.add_annotation(
                        x=metric_names[j],
                        y=param_names[i],
                        text=f"{param_metric_corr.iloc[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(param_metric_corr.iloc[i, j]) > 0.5 else "black")
                    )
            
            # Update layout
            fig.update_layout(
                title="Parameter-Metric Correlation Matrix",
                width=800,
                height=600
            )
            
            # Display figure
            st.plotly_chart(fig)
        
        with corr_tab2:
            st.subheader("Parameter Importance Analysis")
            
            # Select target metric for importance analysis
            metric_options = [col for col in corr_df.columns if col in metric_names]
            if metric_options:
                selected_importance_metric = st.selectbox(
                    "Select target metric for importance analysis", 
                    metric_options,
                    index=metric_options.index('return_pct') if 'return_pct' in metric_options else 0,
                    key="importance_metric"
                )
                
                # Calculate absolute correlation as importance
                importance_values = []
                for param in param_names:
                    if param in correlation_df.index and selected_importance_metric in correlation_df.columns:
                        importance = abs(correlation_df.loc[param, selected_importance_metric])
                        importance_values.append((param, importance))
                
                # Sort by importance
                importance_values.sort(key=lambda x: x[1], reverse=True)
                
                # Plot importance bar chart
                importance_df = pd.DataFrame(importance_values, columns=['Parameter', 'Importance'])
                
                fig = px.bar(
                    importance_df,
                    x='Parameter',
                    y='Importance',
                    title=f"Parameter Importance for {selected_importance_metric}",
                    labels={'Importance': f'Absolute Correlation with {selected_importance_metric}'},
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                # Add values on top of bars
                for i, row in enumerate(importance_df.itertuples()):
                    fig.add_annotation(
                        x=row.Parameter,
                        y=row.Importance,
                        text=f"{row.Importance:.3f}",
                        showarrow=False,
                        yshift=10
                    )
                
                fig.update_layout(width=800, height=500)
                st.plotly_chart(fig)
                
                # Parameter impact explanation
                st.subheader("Parameter Impact Interpretation")
                
                # Create a more detailed explanation of each parameter's impact
                for param, importance in importance_values:
                    if param in correlation_df.index and selected_importance_metric in correlation_df.columns:
                        # Get the actual correlation (with sign)
                        corr_value = correlation_df.loc[param, selected_importance_metric]
                        direction = "increases" if corr_value > 0 else "decreases"
                        strength = "strong" if importance > 0.5 else "moderate" if importance > 0.2 else "weak"
                        
                        # Create explanation
                        st.markdown(f"**{param}** has a **{strength}** impact on {selected_importance_metric}. "
                                  f"Increasing this parameter likely {direction} {selected_importance_metric} "
                                  f"(correlation: {corr_value:.3f}).")
            else:
                st.warning("No numeric metrics available for importance analysis.")
        
        with corr_tab3:
            st.subheader("Parameter Interactions Analysis")
            
            # Select first parameter
            if len(param_names) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    param1 = st.selectbox(
                        "Select first parameter", 
                        param_names,
                        key="interaction_param1"
                    )
                
                with col2:
                    # Filter out the first parameter from options
                    param2_options = [p for p in param_names if p != param1]
                    param2 = st.selectbox(
                        "Select second parameter", 
                        param2_options,
                        key="interaction_param2"
                    )
                
                # Select target metric for analysis
                target_metric = st.selectbox(
                    "Select target metric for interaction analysis", 
                    metric_names,
                    index=metric_names.index('return_pct') if 'return_pct' in metric_names else 0,
                    key="interaction_metric"
                )
                
                # Ensure all selected values are in the dataset
                if (param1 in corr_df.columns and param2 in corr_df.columns and 
                    target_metric in corr_df.columns):
                    
                    # Create correlation between parameters
                    param_corr = correlation_df.loc[param1, param2]
                    
                    # Display correlation info
                    corr_color = "green" if abs(param_corr) < 0.3 else "orange" if abs(param_corr) < 0.7 else "red"
                    st.markdown(f"Correlation between **{param1}** and **{param2}**: "
                               f"<span style='color:{corr_color};font-weight:bold'>{param_corr:.3f}</span>", 
                               unsafe_allow_html=True)
                    
                    if abs(param_corr) > 0.7:
                        st.warning(f"High correlation detected! Parameters {param1} and {param2} are strongly correlated, "
                                  f"which may indicate redundancy in the optimization space.")
                    
                    # Create a copy of data for visualization
                    viz_df = corr_df.copy()
                    
                    # Create size column for 3D scatter plot (ensure positive values)
                    # Handle negative values if needed
                    if viz_df[target_metric].min() < 0:
                        # Make all size values positive by adding an offset
                        viz_df['size_value'] = viz_df[target_metric] + abs(viz_df[target_metric].min()) + 1
                    else:
                        # If all values are already positive, just use them directly
                        viz_df['size_value'] = viz_df[target_metric].copy()
                    
                    # Ensure a minimum size for all points
                    viz_df['size_value'] = viz_df['size_value'].clip(lower=5)
                    
                    # Create 3D scatter plot to visualize parameter interactions
                    fig = px.scatter_3d(
                        viz_df, 
                        x=param1, 
                        y=param2, 
                        z=target_metric,
                        color=target_metric,
                        size='size_value',
                        size_max=15,
                        opacity=0.7,
                        title=f"Parameter Interaction: {param1} vs {param2} on {target_metric}"
                    )
                    
                    # Customize layout
                    fig.update_layout(
                        scene = dict(
                            xaxis_title=param1,
                            yaxis_title=param2,
                            zaxis_title=target_metric
                        ),
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Create 2D heatmap for more clear visualization
                    st.subheader("2D Interaction Heatmap")
                    
                    # Create discrete bins for both parameters
                    param1_bins = min(10, len(corr_df[param1].unique()))  # Adjust bin count based on unique values
                    param2_bins = min(10, len(corr_df[param2].unique()))
                    
                    # Create a copy of the data with binned parameters
                    binned_df = corr_df.copy()
                    
                    # Check if we have enough unique values for binning
                    if param1_bins < 2 or param2_bins < 2:
                        st.warning(f"Not enough unique values for the selected parameters to create a meaningful heatmap.")
                        
                        # Show available values
                        st.subheader("Available Parameter Values")
                        
                        unique_param1 = sorted(binned_df[param1].unique())
                        unique_param2 = sorted(binned_df[param2].unique())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{param1}** unique values:")
                            for val in unique_param1:
                                st.write(f"- {val}")
                        
                        with col2:
                            st.write(f"**{param2}** unique values:")
                            for val in unique_param2:
                                st.write(f"- {val}")
                                
                        # Create scatter plot instead
                        st.subheader("Alternative Visualization: Scatter Plot")
                        st.write("Showing relationship between parameters using a scatter plot:")
                        
                        # Fix: Use absolute values for size to ensure positive values
                        # Also add a size_min parameter to ensure all points are visible
                        
                        # Create a positive version of the target metric for sizing points
                        # First, copy the target metric values
                        binned_df['original_values'] = binned_df[target_metric].copy()
                        
                        # Handle any negative values to ensure all size values are positive
                        if binned_df[target_metric].min() < 0:
                            # Shift all values up to make them positive
                            shift_value = abs(binned_df[target_metric].min()) + 1
                            binned_df['size_value'] = binned_df[target_metric] + shift_value
                        else:
                            # If all values are already positive, just use them directly
                            binned_df['size_value'] = binned_df[target_metric].copy()
                        
                        # Ensure a minimum size for all points regardless of value
                        binned_df['size_value'] = binned_df['size_value'].clip(lower=5)  # Min size of 5
                                
                        # Create scatter plot
                        scatter_fig = px.scatter(
                            binned_df, 
                            x=param1, 
                            y=param2, 
                            color=target_metric,
                            size='size_value',
                            size_max=20,
                            color_continuous_scale="Viridis",
                            title=f"Relationship between {param1}, {param2}, and {target_metric}",
                            labels={param1: param1, param2: param2, target_metric: target_metric}
                        )
                        
                        scatter_fig.update_layout(
                            width=800,
                            height=500,
                            coloraxis_colorbar_title=target_metric
                        )
                        
                        st.plotly_chart(scatter_fig)
                        st.stop()
                    
                    # Create bins for parameters safely with error handling
                    try:
                        # Try qcut but fall back to cut if we have repeated values
                        try:
                            binned_df[f'{param1}_bin'] = pd.qcut(
                                binned_df[param1], 
                                param1_bins, 
                                labels=list(range(param1_bins)), 
                                duplicates='drop'
                            )
                        except ValueError:
                            # If qcut fails, try cut instead
                            binned_df[f'{param1}_bin'] = pd.cut(
                                binned_df[param1], 
                                bins=param1_bins, 
                                labels=list(range(param1_bins-1))
                            )
                            
                        try:
                            binned_df[f'{param2}_bin'] = pd.qcut(
                                binned_df[param2], 
                                param2_bins, 
                                labels=list(range(param2_bins)), 
                                duplicates='drop'
                            )
                        except ValueError:
                            # If qcut fails, try cut instead
                            binned_df[f'{param2}_bin'] = pd.cut(
                                binned_df[param2], 
                                bins=param2_bins, 
                                labels=list(range(param2_bins-1))
                            )
                    except Exception as e:
                        st.error(f"Error creating parameter bins: {str(e)}")
                        st.stop()
                    
                    # Calculate mean of target metric for each bin combination
                    heatmap_data = binned_df.groupby([f'{param1}_bin', f'{param2}_bin'])[target_metric].mean().reset_index()
                    
                    # Create pivot table for heatmap
                    heatmap_pivot = heatmap_data.pivot(index=f'{param2}_bin', columns=f'{param1}_bin', values=target_metric)
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_pivot.values,
                        x=[f"{binned_df[param1].min():.2f} - {binned_df[param1].max():.2f}" for _ in range(len(heatmap_pivot.columns))],
                        y=[f"{binned_df[param2].min():.2f} - {binned_df[param2].max():.2f}" for _ in range(len(heatmap_pivot.index))],
                        colorscale='Viridis',
                        colorbar=dict(title=target_metric)
                    ))
                    
                    fig.update_layout(
                        title=f"Heatmap of {target_metric} by {param1} and {param2}",
                        xaxis_title=f"{param1} (increasing →)",
                        yaxis_title=f"{param2} (increasing →)",
                        width=800,
                        height=500
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Analysis of interactions
                    st.subheader("Interaction Analysis")
                    
                    # Calculate parameter interaction effect
                    # Compare the effect of param1 at different levels of param2
                    param2_low = binned_df[param2] < binned_df[param2].median()
                    param2_high = binned_df[param2] >= binned_df[param2].median()
                    
                    corr_at_low_param2 = binned_df[param2_low].corr().loc[param1, target_metric]
                    corr_at_high_param2 = binned_df[param2_high].corr().loc[param1, target_metric]
                    
                    # Check if effect changes
                    effect_change = corr_at_high_param2 - corr_at_low_param2
                    
                    st.markdown(f"**Effect of {param1} on {target_metric} when {param2} is LOW:** {corr_at_low_param2:.3f}")
                    st.markdown(f"**Effect of {param1} on {target_metric} when {param2} is HIGH:** {corr_at_high_param2:.3f}")
                    st.markdown(f"**Interaction strength:** {abs(effect_change):.3f}")
                    
                    if abs(effect_change) > 0.3:
                        st.markdown("**Significant interaction detected!** The effect of one parameter depends on the value of the other.")
                        if effect_change > 0:
                            st.markdown(f"- As {param2} increases, the positive effect of {param1} on {target_metric} gets stronger.")
                        else:
                            st.markdown(f"- As {param2} increases, the positive effect of {param1} on {target_metric} gets weaker.")
                    else:
                        st.markdown("**Limited interaction detected.** These parameters mostly act independently.")
                else:
                    st.error("Selected parameters or metric not found in the dataset.")
            else:
                st.warning("At least two numeric parameters are needed for interaction analysis.")
        
        # Key insights
        st.subheader("Key Correlation Insights")
        
        # Find strongest positive and negative correlations
        strongest_positive = pd.DataFrame(columns=['Parameter', 'Metric', 'Correlation'])
        strongest_negative = pd.DataFrame(columns=['Parameter', 'Metric', 'Correlation'])
        
        for param in param_cols:
            for metric in metric_cols:
                corr_value = param_metric_corr.loc[param, metric]
                
                param_name = param.replace('param_', '')
                metric_name = metric.replace('metric_', '')
                
                new_row = pd.DataFrame({
                    'Parameter': [param_name],
                    'Metric': [metric_name],
                    'Correlation': [corr_value]
                })
                
                if corr_value > 0:
                    strongest_positive = pd.concat([strongest_positive, new_row], ignore_index=True)
                else:
                    strongest_negative = pd.concat([strongest_negative, new_row], ignore_index=True)
        
        # Sort by absolute correlation
        strongest_positive = strongest_positive.sort_values('Correlation', ascending=False).head(5)
        strongest_negative = strongest_negative.sort_values('Correlation', ascending=True).head(5)
        
        # Display insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strongest Positive Correlations**")
            if not strongest_positive.empty:
                for _, row in strongest_positive.iterrows():
                    st.write(f"- **{row['Parameter']}** → **{row['Metric']}**: {row['Correlation']:.2f}")
            else:
                st.write("No positive correlations found.")
        
        with col2:
            st.markdown("**Strongest Negative Correlations**")
            if not strongest_negative.empty:
                for _, row in strongest_negative.iterrows():
                    st.write(f"- **{row['Parameter']}** → **{row['Metric']}**: {row['Correlation']:.2f}")
            else:
                st.write("No negative correlations found.")
        
        # Parameter recommendations based on correlations
        st.subheader("Parameter Recommendations")
        
        # Check return correlation
        return_col = 'metric_return_pct'
        if return_col in correlation_df.columns:
            # Get parameters with strongest correlation to return
            return_corr = correlation_df[return_col].loc[param_cols].sort_values(ascending=False)
            
            st.markdown("**Parameters Ranked by Impact on Returns:**")
            
            for param, corr in return_corr.items():
                param_name = param.replace('param_', '')
                impact = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.2 else "weak"
                
                st.write(f"- **{param_name}**: {strength} {impact} impact ({corr:.2f})")
        
        # Optimal parameter combination recommendation
        st.subheader("Optimal Parameter Combination")
        
        # Find best result by return
        if 'results' in result_data and result_data['results']:
            best_result = max(result_data['results'], key=lambda x: x['metrics']['return_pct'])
            
            st.markdown("**Recommended Parameter Values:**")
            
            for param, value in best_result['parameters'].items():
                st.write(f"- **{param}**: {value}")
            
            # Show performance
            st.markdown("**Expected Performance:**")
            for metric, value in best_result['metrics'].items():
                if metric in ['return_pct', 'max_drawdown_pct', 'win_rate']:
                    st.write(f"- **{metric}**: {value:.2f}%")
                else:
                    st.write(f"- **{metric}**: {value:.2f}")
    else:
        st.warning("No optimization results loaded. Please select results in the Parameter Impact Analysis tab.")

with tab3:
    st.header("Performance Distribution")
    
    # Load optimization results
    if 'results_df' in locals() and not results_df.empty:
        # Select metric for distribution analysis
        metric_options = [col for col in results_df.columns if col.startswith('metric_')]
        metric_labels = [col.replace('metric_', '') for col in metric_options]
        selected_dist_metric = st.selectbox(
            "Select metric for distribution analysis", 
            metric_labels, 
            index=metric_labels.index('return_pct') if 'return_pct' in metric_labels else 0,
            key="dist_metric"
        )
        selected_dist_metric_col = f"metric_{selected_dist_metric}"
        
        if selected_dist_metric_col in results_df.columns:
            # Create histogram
            fig = px.histogram(
                results_df, 
                x=selected_dist_metric_col,
                nbins=30,
                marginal="box",
                title=f"Distribution of {selected_dist_metric}",
                labels={selected_dist_metric_col: selected_dist_metric}
            )
            
            # Add mean and median lines
            mean_value = results_df[selected_dist_metric_col].mean()
            median_value = results_df[selected_dist_metric_col].median()
            
            fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_value:.2f}", annotation_position="top right")
            fig.add_vline(x=median_value, line_dash="dash", line_color="green",
                         annotation_text=f"Median: {median_value:.2f}", annotation_position="top left")
            
            # Update layout
            fig.update_layout(
                width=800,
                height=500
            )
            
            # Display figure
            st.plotly_chart(fig)
            
            # Distribution statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean", f"{mean_value:.2f}")
                st.metric("Standard Deviation", f"{results_df[selected_dist_metric_col].std():.2f}")
            
            with col2:
                st.metric("Median", f"{median_value:.2f}")
                st.metric("Interquartile Range", f"{results_df[selected_dist_metric_col].quantile(0.75) - results_df[selected_dist_metric_col].quantile(0.25):.2f}")
            
            with col3:
                st.metric("Min", f"{results_df[selected_dist_metric_col].min():.2f}")
                st.metric("Max", f"{results_df[selected_dist_metric_col].max():.2f}")
            
            # Percentile analysis
            st.subheader("Percentile Analysis")
            
            # Calculate percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            percentile_values = [results_df[selected_dist_metric_col].quantile(p/100) for p in percentiles]
            
            # Create table
            percentile_df = pd.DataFrame({
                'Percentile': [f"{p}%" for p in percentiles],
                selected_dist_metric: [f"{val:.2f}" for val in percentile_values]
            })
            
            st.table(percentile_df)
            
            # Top performing parameter combinations
            st.subheader("Top Performing Parameter Combinations")
            
            # Sort by selected metric
            if selected_dist_metric in ['return_pct', 'win_rate', 'profit_factor', 'sharpe_ratio']:
                # For metrics where higher is better
                top_combinations = results_df.sort_values(selected_dist_metric_col, ascending=False).head(10)
            else:
                # For metrics where lower is better
                top_combinations = results_df.sort_values(selected_dist_metric_col, ascending=True).head(10)
            
            # Display top combinations
            param_cols = [col for col in top_combinations.columns if col.startswith('param_')]
            metric_cols = [col for col in top_combinations.columns if col.startswith('metric_')]
            
            display_cols = [col.replace('param_', '') for col in param_cols] + [col.replace('metric_', '') for col in metric_cols]
            display_df = pd.DataFrame(index=range(len(top_combinations)))
            
            for col in param_cols:
                display_df[col.replace('param_', '')] = top_combinations[col].values
            
            for col in metric_cols:
                display_df[col.replace('metric_', '')] = top_combinations[col].values
            
            st.dataframe(display_df)
        else:
            st.error("Selected metric not found in the results.")
    else:
        st.warning("No optimization results loaded. Please select results in the Parameter Impact Analysis tab.")
with tab4:
    st.header("Meta-Analysis")
    
    # Load all optimization results for meta-analysis
    results_manager = ResultsManager()
    all_optimization_files = results_manager.list_optimization_results()
    
    if not all_optimization_files:
        st.warning("No optimization results found. Run some optimizations first.")
    else:
        # Process optimization files
        st.subheader("Optimization Test Overview")
        st.write("Analysis of all optimization tests across different datasets and parameter configurations.")
        
        # Load and process results
        try:
            processed_results, file_summaries, errors = MetaAnalysis.load_and_process_results(
                results_manager, all_optimization_files
            )
            
            if errors:
                st.error(f"Encountered {len(errors)} errors while loading results: {errors}")
            
            if processed_results and len(processed_results) > 0:
                # Create a combined DataFrame
                all_results_df = pd.concat([result['data'] for result in processed_results if 'data' in result])
                
                # Display test information
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tests", len(processed_results))
                    st.metric("Total Parameter Combinations", len(all_results_df))
                
                with col2:
                    unique_datasets = len(set([result['dataset_name'] for result in processed_results if 'dataset_name' in result]))
                    st.metric("Unique Datasets", unique_datasets)
                    st.metric("Date Range", 
                             f"{min([result['timestamp'] for result in processed_results if 'timestamp' in result])[:10]} to "
                             f"{max([result['timestamp'] for result in processed_results if 'timestamp' in result])[:10]}")
                
                # Show file summaries in an expandable section
                with st.expander("Test Summaries", expanded=False):
                    summaries_df = pd.DataFrame(file_summaries)
                    st.dataframe(summaries_df)
                
                # Find common parameters
                common_params = MetaAnalysis.find_common_parameters(all_results_df)
                
                if common_params:
                    # Parameter selection for analysis
                    selected_param = st.selectbox(
                        "Select parameter for analysis",
                        common_params,
                        format_func=lambda x: x.replace('param_', '')
                    )
                    
                    # Metric selection for analysis
                    metric_options = [col for col in all_results_df.columns if col.startswith('metric_')]
                    selected_metric = st.selectbox(
                        "Select metric for analysis",
                        metric_options,
                        format_func=lambda x: x.replace('metric_', '')
                    )
                    
                    # Filter data
                    filtered_df = all_results_df[~all_results_df[selected_param].isna()]
                    
                    # Parameter performance visualization
                    st.subheader("Parameter Performance Across Datasets")
                    param_fig = MetaAnalysis.parameter_performance_across_datasets(
                        filtered_df, selected_param, selected_metric
                    )
                    st.plotly_chart(param_fig, use_container_width=True)
                    
                    # Calculate parameter statistics
                    param_stats = MetaAnalysis.calculate_parameter_stats(
                        filtered_df, selected_param, selected_metric
                    )
                    
                    # Show parameter stats
                    st.subheader("Parameter Performance Statistics")
                    st.dataframe(param_stats)
                    
                    # Parameter performance chart
                    st.subheader("Parameter Value Performance")
                    perf_chart = MetaAnalysis.parameter_performance_chart(
                        param_stats, selected_param.replace('param_', ''), selected_metric.replace('metric_', '')
                    )
                    st.plotly_chart(perf_chart, use_container_width=True)
                    
                    # Find best parameters per dataset
                    st.subheader("Best Parameter Values by Dataset")
                    best_params = MetaAnalysis.find_best_parameters_per_dataset(
                        filtered_df, selected_param, selected_metric
                    )
                    
                    best_params_df = pd.DataFrame([
                        {"Dataset": dataset, "Best Value": value} 
                        for dataset, value in best_params.items()
                    ])
                    st.dataframe(best_params_df)
                    
                    # Parameter importance analysis
                    st.subheader("Parameter Importance Analysis")
                    
                    # Calculate parameter importance
                    importance_data = MetaAnalysis.calculate_parameter_importance(
                        all_results_df, common_params, metric_options
                    )
                    
                    if importance_data:
                        # Show correlation heatmap
                        st.write("Parameter-Metric Correlation Heatmap")
                        corr_fig = MetaAnalysis.create_correlation_heatmap(importance_data['correlation_df'])
                        st.plotly_chart(corr_fig, use_container_width=True)
                        
                        # Show importance chart
                        st.write("Parameter Importance Scores")
                        importance_fig = MetaAnalysis.create_importance_chart(importance_data['importance_df'])
                        st.plotly_chart(importance_fig, use_container_width=True)
                else:
                    st.warning("No common parameters found across optimization tests.")
            else:
                st.warning("No valid results found for meta-analysis.")
                
        except Exception as e:
            st.error(f"Error in meta-analysis: {str(e)}")
            st.warning("There may be incompatible optimization result files. Check the results directory.")
