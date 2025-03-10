import pandas as pd
import numpy as np
import time
import os
import json
import datetime as dt
import requests
import tensorflow as tf
from feature_engineering import calculate_technical_indicators, TimeSeriesDecomposer
from gan_lstm_model import GANLSTM
from rainbow_dqn_agent import RainbowDQNAgent
from rl_environment import EnhancedTradingEnv

class LiveTradingSystem:
    """
    Live trading system for real-time integration
    """
    def __init__(
        self,
        model_path,
        symbols,
        api_key=None,
        lookback_days=30,
        update_interval=60,
        risk_threshold=0.1,
        log_dir='logs'
    ):
        """
        Initialize the live trading system
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        symbols : list
            List of symbols to trade
        api_key : str, optional
            API key for data provider
        lookback_days : int
            Number of days to look back for features
        update_interval : int
            Update interval in seconds
        risk_threshold : float
            Risk threshold for maximum loss
        log_dir : str
            Directory for logs
        """
        self.model_path = model_path
        self.symbols = symbols
        self.api_key = api_key
        self.lookback_days = lookback_days
        self.update_interval = update_interval
        self.risk_threshold = risk_threshold
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize models
        self.forecast_model = None
        self.agent = None
        
        # Initialize portfolio
        self.portfolio = {
            'cash': 10000,
            'positions': {},
            'history': []
        }
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """
        Load trained models
        """
        # Load forecast model
        try:
            self.forecast_model = tf.keras.models.load_model(f'{self.model_path}/generator.h5')
            print("Forecast model loaded successfully")
        except Exception as e:
            print(f"Error loading forecast model: {e}")
        
        # Load agent
        try:
            # Get state size from environment
            env = self._create_test_env()
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            
            # Create agent
            self.agent = RainbowDQNAgent(
                state_size=state_size,
                action_size=action_size
            )
            
            # Load weights
            self.agent.load(f'{self.model_path}/rainbow_dqn_final.h5')
            print("Agent loaded successfully")
        except Exception as e:
            print(f"Error loading agent: {e}")
    
    def _create_test_env(self):
        """
        Create a test environment
        
        Returns:
        --------
        EnhancedTradingEnv
            Test environment
        """
        # Create dummy dataframe
        df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'close': np.random.rand(100),
            'MACD': np.random.rand(100),
            'RSI': np.random.rand(100),
            'PC1': np.random.rand(100),
            'PC2': np.random.rand(100),
            'trend': np.random.rand(100),
            'seasonality': np.random.rand(100),
            'residual': np.random.rand(100)
        })
        
        # Create environment
        env = EnhancedTradingEnv(df, forecast_model=self.forecast_model)
        
        return env
    
    def fetch_live_data(self, symbol, days=30):
        """
        Fetch live data for a symbol
        
        Parameters:
        -----------
        symbol : str
            Symbol to fetch data for
        days : int
            Number of days to fetch
            
        Returns:
        --------
        pandas.DataFrame
            Fetched data
        """
        # Calculate start and end dates
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=days)
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        try:
            # Use Alpha Vantage API for demonstration
            # In a real system, you would use your broker's API
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=full'
            response = requests.get(url)
            data = response.json()
            
            # Extract time series data
            time_series = data.get('Time Series (Daily)', {})
            
            # Convert to dataframe
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Filter by date range
            df = df[(df.index >= start_str) & (df.index <= end_str)]
            
            # Reset index
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            
            return df
        
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            
            # Return dummy data for demonstration
            return pd.DataFrame({
                'date': pd.date_range(start=start_str, end=end_str),
                'open': np.random.rand(days),
                'high': np.random.rand(days),
                'low': np.random.rand(days),
                'close': np.random.rand(days),
                'volume': np.random.rand(days)
            })
    
    def preprocess_data(self, df):
        """
        Preprocess data for prediction
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw data
            
        Returns:
        --------
        pandas.DataFrame
            Processed data
        """
        # Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        # Decompose time series
        decomposer = TimeSeriesDecomposer()
        trend, seasonality, residuals = decomposer.decompose_series(df_with_indicators['close'])
        
        # Add decomposition to dataframe
        df_with_indicators['trend'] = trend
        df_with_indicators['seasonality'] = seasonality
        df_with_indicators['residual'] = residuals
        
        # Add dummy eigen-portfolio features
        df_with_indicators['PC1'] = np.random.rand(len(df_with_indicators))
        df_with_indicators['PC2'] = np.random.rand(len(df_with_indicators))
        
        return df_with_indicators
    
    def predict_action(self, df):
        """
        Predict action for a dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Processed data
            
        Returns:
        --------
        int
            Predicted action
        """
        # Create environment
        env = EnhancedTradingEnv(df, forecast_model=self.forecast_model)
        
        # Get observation
        observation = env.reset()
        
        # Predict action
        action = self.agent.act(observation, training=False)
        
        return action
    
    def execute_trade(self, symbol, action, price, quantity=None):
        """
        Execute a trade
        
        Parameters:
        -----------
        symbol : str
            Symbol to trade
        action : int
            Action to take (0=Hold, 1=Buy, 2=Sell, 3=Close)
        price : float
            Current price
        quantity : float, optional
            Quantity to trade
            
        Returns:
        --------
        dict
            Trade details
        """
        # Initialize trade details
        trade = {
            'symbol': symbol,
            'action': action,
            'price': price,
            'timestamp': dt.datetime.now().isoformat(),
            'status': 'executed'
        }
        
        # Execute action
        if action == 0:  # Hold
            trade['action_name'] = 'hold'
            trade['quantity'] = 0
            trade['value'] = 0
        
        elif action == 1:  # Buy
            trade['action_name'] = 'buy'
            
            # Calculate quantity if not provided
            if quantity is None:
                # Use 10% of available cash
                cash_to_use = self.portfolio['cash'] * 0.1
                quantity = cash_to_use / price
            
            # Update trade details
            trade['quantity'] = quantity
            trade['value'] = price * quantity
            
            # Update portfolio
            if symbol not in self.portfolio['positions']:
                self.portfolio['positions'][symbol] = 0
            
            self.portfolio['positions'][symbol] += quantity
            self.portfolio['cash'] -= trade['value']
        
        elif action == 2:  # Sell
            trade['action_name'] = 'sell'
            
            # Calculate quantity if not provided
            if quantity is None:
                # Sell all if we have the position
                quantity = self.portfolio['positions'].get(symbol, 0)
            
            # Update trade details
            trade['quantity'] = quantity
            trade['value'] = price * quantity
            
            # Update portfolio
            if symbol in self.portfolio['positions']:
                self.portfolio['positions'][symbol] -= quantity
                
                # Remove position if zero
                if self.portfolio['positions'][symbol] <= 0:
                    del self.portfolio['positions'][symbol]
            
            self.portfolio['cash'] += trade['value']
        
        elif action == 3:  # Close
            trade['action_name'] = 'close'
            
            # Close all positions for the symbol
            if symbol in self.portfolio['positions']:
                quantity = self.portfolio['positions'][symbol]
                trade['quantity'] = quantity
                trade['value'] = price * quantity
                
                # Update portfolio
                self.portfolio['cash'] += trade['value']
                del self.portfolio['positions'][symbol]
            else:
                trade['quantity'] = 0
                trade['value'] = 0
        
        # Add trade to history
        self.portfolio['history'].append(trade)
        
        # Log trade
        self._log_trade(trade)
        
        return trade
    
    def _log_trade(self, trade):
        """
        Log a trade
        
        Parameters:
        -----------
        trade : dict
            Trade details
        """
        # Create log file path
        log_file = f"{self.log_dir}/trades_{dt.datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add trade to logs
        logs.append(trade)
        
        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _log_portfolio(self):
        """
        Log portfolio status
        """
        # Create log file path
        log_file = f"{self.log_dir}/portfolio_{dt.datetime.now().strftime('%Y%m%d')}.json"
        
        # Save portfolio
        with open(log_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def run(self):
        """
        Run the trading system
        """
        print("Starting live trading system...")
        
        try:
            while True:
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        # Fetch live data
                        print(f"Fetching data for {symbol}...")
                        df = self.fetch_live_data(symbol, days=self.lookback_days)
                        
                        # Preprocess data
                        print(f"Preprocessing data for {symbol}...")
                        processed_df = self.preprocess_data(df)
                        
                        # Predict action
                        print(f"Predicting action for {symbol}...")
                        action = self.predict_action(processed_df)
                        
                        # Get current price
                        current_price = processed_df.iloc[-1]['close']
                        
                        # Execute trade
                        print(f"Executing trade for {symbol}...")
                        trade = self.execute_trade(symbol, action, current_price)
                        
                        # Print trade details
                        print(f"Trade executed: {trade}")
                        
                        # Check risk
                        self._check_risk()
                        
                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")
                
                # Log portfolio
                self._log_portfolio()
                
                # Wait for next update
                print(f"Waiting {self.update_interval} seconds for next update...")
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("Trading system stopped by user")
        
        except Exception as e:
            print(f"Trading system error: {e}")
        
        finally:
            # Log final portfolio
            self._log_portfolio()
            print("Trading system stopped")
    
    def _check_risk(self):
        """
        Check risk and take action if necessary
        """
        # Calculate portfolio value
        portfolio_value = self.portfolio['cash']
        for symbol, quantity in self.portfolio['positions'].items():
            # Fetch current price
            df = self.fetch_live_data(symbol, days=1)
            current_price = df.iloc[-1]['close']
            
            # Add position value
            portfolio_value += current_price * quantity
        
        # Calculate initial portfolio value (assuming 10000 if no history)
        if not self.portfolio['history']:
            initial_value = 10000
        else:
            # Find first trade
            first_trade = min(self.portfolio['history'], key=lambda x: x['timestamp'])
            initial_value = 10000  # Assuming initial value
        
        # Calculate loss
        loss = (portfolio_value - initial_value) / initial_value
        
        # Check if loss exceeds threshold
        if loss < -self.risk_threshold:
            print(f"Risk threshold exceeded: {loss:.2%}. Closing all positions.")
            
            # Close all positions
            for symbol in list(self.portfolio['positions'].keys()):
                # Fetch current price
                df = self.fetch_live_data(symbol, days=1)
                current_price = df.iloc[-1]['close']
                
                # Execute close trade
                self.execute_trade(symbol, 3, current_price)

if __name__ == "__main__":
    # Define symbols to trade
    symbols = ['GS', 'JPM', 'MS', 'BAC', 'C']
    
    # Create trading system
    trading_system = LiveTradingSystem(
        model_path='models',
        symbols=symbols,
        api_key='YOUR_API_KEY',  # Replace with your API key
        lookback_days=30,
        update_interval=60,  # 1 minute
        risk_threshold=0.1,
        log_dir='logs'
    )
    
    # Run trading system
    trading_system.run() 