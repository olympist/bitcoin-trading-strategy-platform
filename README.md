# Bitcoin Trading Strategy Backtester

A sophisticated Bitcoin trading strategy backtesting platform with advanced cloud deployment capabilities, enabling algorithmic traders to seamlessly transition from local development to cloud infrastructure.

## Core Technologies
- Python-based backtesting engine
- Streamlit interactive web interface
- Google Cloud Platform deployment automation
- Plotly advanced data visualization
- Monte Carlo simulation algorithms
- Flexible cloud VM configuration
- Automated deployment scripts

## Project Structure
- `app.py`: Main Streamlit application entry point
- `pages/`: Additional Streamlit application pages
  - `auto_optimization.py`: Automated parameter optimization
  - `dataset_comparison.py`: Compare optimization across datasets
  - `optimization_analysis.py`: Analyze optimization results
  - `optimization_progress.py`: Monitor optimization progress
  - `optimization_v2.py`: V2 strategy optimization 
  - `parameter_optimization.py`: Parameter optimization
  - `v7_strategy_dashboard.py`: Advanced V7 strategy with custom timing options
- `utils/`: Utility modules
  - `task_manager.py`: Manages optimization tasks
  - `optimization_strategies.py`: Different optimization approaches
  - `meta_analysis.py`: Cross-dataset analysis
  - `results_manager.py`: Save/load optimization results
- `strategy/`: Trading strategy implementations
- `backtesting/`: Backtesting engine
- `trading_bot/`: Live trading simulation
- `visualization/`: Data visualization utilities
- Deployment scripts: Cloud deployment utilities

## Getting Started in Replit

### 1. Create the Configuration File
First, make sure you have the required Streamlit configuration for Replit:

```sh
mkdir -p .streamlit
```

Create a configuration file at `.streamlit/config.toml` with:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### 2. Install Required Packages
Install necessary Python packages:

```sh
pip install streamlit pandas numpy plotly scikit-learn google-api-python-client google-auth google-cloud-compute google-cloud-storage
```

### 3. Start the Application
Run the Streamlit application:

```sh
streamlit run app.py
```

## Cloud Deployment Instructions

### Prerequisites
- Google Cloud Platform account
- Project with Compute Engine API enabled
- Service account credentials with appropriate permissions

### Deployment Options
1. **Minimal Deployment**: Only essential files
   ```sh
   python deploy_minimal.py
   ```

2. **Full Stack Deployment**: Backend + Frontend
   ```sh
   python deploy_frontend_and_backend.py
   ```

3. **Cloud Storage Deployment**: Using GCS as intermediary
   ```sh
   python cloud_storage_deploy.py
   ```

### Troubleshooting Deployment
- Check VM firewall rules (HTTP/HTTPS allowed)
- Verify application installation status via SSH
- Check service status and logs

Detailed deployment instructions are available in the `deployment_instructions.md` and `firewall_instructions.txt` files.

## Firewall Configuration
For the application to be accessible, make sure your VM has HTTP (port 80) and HTTPS (port 443) traffic allowed:

```sh
gcloud compute firewall-rules create allow-http --project PROJECT_ID --allow tcp:80 --source-ranges 0.0.0.0/0
gcloud compute firewall-rules create allow-https --project PROJECT_ID --allow tcp:443 --source-ranges 0.0.0.0/0
```

## VM Access Commands
Generate SSH commands for checking VM status:

```sh
python ssh_command_generator.py
```

## Features
- ✅ Advanced backtesting of Bitcoin trading strategies (V3-V7 implementations)
- ✅ V7 Strategy with custom start times and operating intervals
- ✅ Automated parameter optimization
- ✅ Monte Carlo simulation for risk assessment
- ✅ Interactive performance visualization
- ✅ Cross-dataset parameter robustness analysis
- ✅ One-click cloud deployment
- ✅ Real-time optimization progress tracking
- ✅ Meta-analysis of optimization results