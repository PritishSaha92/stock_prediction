# Reinforcement Learning Enhanced Forecasting System

A comprehensive stock price forecasting and trading system that combines advanced time series analysis, deep learning, and reinforcement learning.

## Features

- **Enhanced Feature Engineering**
  - Technical indicators (MACD, RSI, Bollinger Bands, etc.)
  - Eigen-portfolio construction using PCA
  - Sentiment analysis using BERT

- **Time Series Decomposition**
  - Hybrid Fourier-ARIMA decomposition
  - Trend, seasonality, and residual components

- **GAN-LSTM Architecture**
  - Generative Adversarial Network with LSTM
  - Accurate price forecasting

- **Reinforcement Learning Integration**
  - Rainbow DQN implementation
  - Enhanced trading environment
  - Risk management

- **Bayesian Optimization**
  - Hyperparameter tuning
  - Optimal trading strategy

- **Live Trading System**
  - Real-time data integration
  - Automated trading execution

## Installation

```bash
# Clone the repository
git clone https://github.com/PritishSaha92/stock_prediction.git
cd stockprediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The system can be run in four different modes:

### 1. Training Mode

Train the GAN-LSTM model and Rainbow DQN agent:

```bash
python main.py --mode train --symbol GS --epochs 100 --episodes 100
```

### 2. Optimization Mode

Optimize hyperparameters using Bayesian optimization:

```bash
python main.py --mode optimize --symbol GS --n_trials 50
```

### 3. Backtesting Mode

Backtest the trained model on historical data:

```bash
python main.py --mode backtest --symbol GS
```

### 4. Live Trading Mode

Run the system in live trading mode:

```bash
python main.py --mode live --symbols GS JPM MS BAC C --api_key YOUR_API_KEY
```

## Project Structure

- `data_preprocessing.py`: Data preprocessing and eigen-portfolio construction
- `feature_engineering.py`: Technical indicators, time series decomposition, and sentiment analysis
- `gan_lstm_model.py`: GAN-LSTM model for price forecasting
- `rl_environment.py`: Reinforcement learning environment for trading
- `rainbow_dqn_agent.py`: Rainbow DQN agent implementation
- `bayesian_optimization.py`: Bayesian optimization for hyperparameter tuning
- `live_trading_system.py`: Live trading system for real-time integration
- `main.py`: Main script to run the system

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Gym
- Optuna
- Transformers
- Matplotlib
- Statsmodels
- Requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Rainbow DQN implementation is based on the paper "Rainbow: Combining Improvements in Deep Reinforcement Learning" by Hessel et al.
- The GAN-LSTM architecture is inspired by various research papers on time series forecasting.

