import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import modules
from data_preprocessing import process_finance50, create_eigen_portfolios
from feature_engineering import calculate_technical_indicators, TimeSeriesDecomposer, SentimentAnalyzer, merge_features
from gan_lstm_model import GANLSTM, prepare_data
from rl_environment import EnhancedTradingEnv
from rainbow_dqn_agent import RainbowDQNAgent, RainbowDQNTrainer
from bayesian_optimization import BayesianOptimizer, train_with_best_params
from live_trading_system import LiveTradingSystem

def parse_args():
    """
    Parse command line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Reinforcement Learning Enhanced Forecasting System')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'optimize', 'backtest', 'live'],
                        help='Mode to run the system in')
    
    # Data arguments
    parser.add_argument('--symbol', type=str, default='GS',
                        help='Symbol to use for training/backtesting')
    parser.add_argument('--start_date', type=str, default='2009-01-01',
                        help='Start date for training/backtesting')
    parser.add_argument('--end_date', type=str, default='2017-12-31',
                        help='End date for training/backtesting')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for GAN-LSTM training')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for RL training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Sequence length for time series')
    
    # Optimization arguments
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials for Bayesian optimization')
    
    # Live trading arguments
    parser.add_argument('--symbols', type=str, nargs='+', default=['GS', 'JPM', 'MS', 'BAC', 'C'],
                        help='Symbols to trade in live mode')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for data provider')
    parser.add_argument('--update_interval', type=int, default=60,
                        help='Update interval in seconds for live trading')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save/load models')
    
    return parser.parse_args()

def setup_directories(args):
    """
    Set up directories for output
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)

def train_mode(args):
    """
    Run the system in training mode
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("Running in training mode...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(f'data/{args.symbol}.csv', parse_dates=['date'])
    
    # Load eigen-portfolios
    print("Loading eigen-portfolios...")
    try:
        eigen_df = pd.read_csv('data/eigen_portfolios.csv', parse_dates=['date'])
    except FileNotFoundError:
        print("Eigen-portfolios not found. Creating them...")
        fin50_df = pd.read_csv('data/finance50.csv', parse_dates=['date'])
        eigen_df = create_eigen_portfolios(fin50_df.drop('date', axis=1))
        eigen_df.to_csv('data/eigen_portfolios.csv', index=False)
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    df_with_indicators = calculate_technical_indicators(df)
    
    # Decompose time series
    print("Decomposing time series...")
    decomposer = TimeSeriesDecomposer()
    trend, seasonality, residuals = decomposer.decompose_series(df_with_indicators['close'])
    
    # Add decomposition to dataframe
    df_with_indicators['trend'] = trend
    df_with_indicators['seasonality'] = seasonality
    df_with_indicators['residual'] = residuals
    
    # Merge features
    print("Merging features...")
    merged_features = merge_features(df_with_indicators, eigen_df)
    
    # Save merged features
    merged_features.to_csv(f'data/{args.symbol}_features.csv', index=False)
    
    # Prepare data for GAN-LSTM
    print("Preparing data for GAN-LSTM...")
    feature_cols = ['close', 'MACD', 'RSI', 'PC1', 'trend']
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        merged_features, 'close', feature_cols, args.sequence_length
    )
    
    # Train GAN-LSTM
    print("Training GAN-LSTM...")
    gan_lstm = GANLSTM(input_shape=(args.sequence_length, len(feature_cols)))
    history = gan_lstm.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    # Make predictions
    print("Making predictions...")
    y_pred = gan_lstm.predict(X_test)
    
    # Inverse transform predictions
    y_test_inv = scaler.inverse_transform(np.hstack([y_test, np.zeros((len(y_test), len(feature_cols)))]))[:, 0]
    y_pred_inv = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), len(feature_cols)))]))[:, 0]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2))
    print(f"RMSE: {rmse}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.legend()
    plt.title('GAN-LSTM Predictions')
    plt.savefig(f'{args.output_dir}/gan_lstm_predictions.png')
    
    # Save model
    gan_lstm.save_models(path_prefix=f'{args.model_dir}/')
    
    # Create environment
    print("Creating RL environment...")
    env = EnhancedTradingEnv(merged_features, forecast_model=gan_lstm.generator)
    
    # Create agent
    print("Creating RL agent...")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        learning_rate=0.0001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=args.batch_size,
        n_step=3,
        dueling=True,
        noisy=True,
        double_q=True
    )
    
    # Create trainer
    print("Creating RL trainer...")
    trainer = RainbowDQNTrainer(
        env=env,
        agent=agent,
        episodes=args.episodes,
        target_update_freq=10,
        save_freq=25,
        log_freq=5,
        save_dir=args.model_dir
    )
    
    # Train agent
    print("Training RL agent...")
    history = trainer.train()
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'{args.output_dir}/episode_rewards.png')
    
    # Test agent
    print("Testing RL agent...")
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        state, reward, done, _ = env.step(action)
    
    # Render results
    env.render()
    
    # Save results
    env.save_results(f'{args.output_dir}/rainbow_dqn_results.csv')
    
    print("Training completed successfully!")

def optimize_mode(args):
    """
    Run the system in optimization mode
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("Running in optimization mode...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(f'data/{args.symbol}_features.csv', parse_dates=['date'])
    
    # Create environment
    print("Creating RL environment...")
    env = EnhancedTradingEnv(df)
    
    # Create optimizer
    print("Creating Bayesian optimizer...")
    optimizer = BayesianOptimizer(env, n_trials=args.n_trials)
    
    # Run optimization
    print("Running optimization...")
    best_params = optimizer.optimize()
    
    # Plot optimization history
    optimizer.plot_optimization_history(f'{args.output_dir}/optimization_history.png')
    
    # Plot parameter importances
    optimizer.plot_param_importances(f'{args.output_dir}/param_importances.png')
    
    # Save study
    optimizer.save_study(f'{args.output_dir}/optimization_study.csv')
    
    # Train with best parameters
    print("Training with best parameters...")
    agent, history = train_with_best_params(env, best_params, episodes=args.episodes)
    
    # Test agent
    print("Testing agent with best parameters...")
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        state, reward, done, _ = env.step(action)
    
    # Render results
    env.render()
    
    # Save results
    env.save_results(f'{args.output_dir}/best_agent_results.csv')
    
    print("Optimization completed successfully!")

def backtest_mode(args):
    """
    Run the system in backtesting mode
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("Running in backtesting mode...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(f'data/{args.symbol}_features.csv', parse_dates=['date'])
    
    # Load forecast model
    print("Loading forecast model...")
    forecast_model = None
    try:
        forecast_model = tf.keras.models.load_model(f'{args.model_dir}/generator.h5')
    except:
        print("Forecast model not found. Proceeding without it.")
    
    # Create environment
    print("Creating RL environment...")
    env = EnhancedTradingEnv(df, forecast_model=forecast_model)
    
    # Load agent
    print("Loading RL agent...")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size
    )
    agent.load(f'{args.model_dir}/rainbow_dqn_final.h5')
    
    # Run backtest
    print("Running backtest...")
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        state, reward, done, _ = env.step(action)
    
    # Render results
    env.render()
    
    # Save results
    env.save_results(f'{args.output_dir}/backtest_results.csv')
    
    print("Backtesting completed successfully!")

def live_mode(args):
    """
    Run the system in live trading mode
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("Running in live trading mode...")
    
    # Create trading system
    print("Creating live trading system...")
    trading_system = LiveTradingSystem(
        model_path=args.model_dir,
        symbols=args.symbols,
        api_key=args.api_key,
        lookback_days=args.sequence_length,
        update_interval=args.update_interval,
        risk_threshold=0.1,
        log_dir='logs'
    )
    
    # Run trading system
    print("Running live trading system...")
    trading_system.run()

def main():
    """
    Main function
    """
    # Parse arguments
    args = parse_args()
    
    # Set up directories
    setup_directories(args)
    
    # Run in specified mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'optimize':
        optimize_mode(args)
    elif args.mode == 'backtest':
        backtest_mode(args)
    elif args.mode == 'live':
        live_mode(args)
    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main() 