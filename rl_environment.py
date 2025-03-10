import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt

class EnhancedTradingEnv(gym.Env):
    """
    Enhanced Trading Environment for Reinforcement Learning
    """
    def __init__(self, df, forecast_model=None, initial_balance=10000, transaction_cost=0.001, window_size=30):
        """
        Initialize the trading environment
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with features and prices
        forecast_model : object, optional
            Model for price forecasting
        initial_balance : float
            Initial account balance
        transaction_cost : float
            Transaction cost as a fraction of trade value
        window_size : int
            Size of the observation window
        """
        super(EnhancedTradingEnv, self).__init__()
        
        self.df = df
        self.forecast_model = forecast_model
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # Define action and observation spaces
        # Actions: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # Features: OHLCV, technical indicators, eigen-portfolios, sentiment, trend
        self.feature_columns = ['close', 'MACD', 'RSI', 'PC1', 'PC2', 'trend', 'seasonality', 'residual']
        
        # Observation space: features + account info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_columns) + 3,)  # +3 for balance, position, position_value
        )
        
        # Reset the environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment
        
        Returns:
        --------
        numpy.ndarray
            Initial observation
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.total_reward = 0
        self.history = {
            'balance': [self.balance],
            'position': [self.position],
            'position_value': [self.position_value],
            'total_value': [self.balance + self.position_value],
            'reward': [0]
        }
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation
        
        Returns:
        --------
        numpy.ndarray
            Current observation
        """
        # Get current features
        features = self.df.iloc[self.current_step][self.feature_columns].values
        
        # Add forecast if model is provided
        if self.forecast_model is not None:
            # Prepare input for forecast model
            X = self.df.iloc[self.current_step - self.window_size:self.current_step][self.feature_columns].values
            X = X.reshape(1, self.window_size, len(self.feature_columns))
            
            # Get forecast
            forecast = self.forecast_model.predict(X)[0][0]
            features = np.append(features, forecast)
        
        # Add account information
        account_info = np.array([
            self.balance,
            self.position,
            self.position_value
        ])
        
        return np.concatenate([features, account_info])
    
    def step(self, action):
        """
        Take a step in the environment
        
        Parameters:
        -----------
        action : int
            Action to take (0=Hold, 1=Buy, 2=Sell, 3=Close)
            
        Returns:
        --------
        tuple
            (observation, reward, done, info)
        """
        # Get current price
        current_price = self.df.iloc[self.current_step]['close']
        
        # Initialize reward
        reward = 0
        
        # Execute action
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy
            if self.position == 0:  # Only buy if no position
                # Calculate maximum shares to buy
                max_shares = self.balance / (current_price * (1 + self.transaction_cost))
                self.position = max_shares
                
                # Update balance and position value
                cost = current_price * self.position * (1 + self.transaction_cost)
                self.balance -= cost
                self.position_value = current_price * self.position
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': current_price,
                    'shares': self.position,
                    'cost': cost
                })
        
        elif action == 2:  # Sell
            if self.position == 0:  # Only sell if no position
                # Calculate maximum shares to sell short
                max_shares = self.balance / (current_price * (1 + self.transaction_cost))
                self.position = -max_shares
                
                # Update balance and position value
                proceeds = current_price * abs(self.position) * (1 - self.transaction_cost)
                self.balance += proceeds
                self.position_value = current_price * self.position
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'shares': abs(self.position),
                    'proceeds': proceeds
                })
        
        elif action == 3:  # Close
            if self.position > 0:  # Close long position
                # Calculate proceeds
                proceeds = current_price * self.position * (1 - self.transaction_cost)
                self.balance += proceeds
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'close_long',
                    'price': current_price,
                    'shares': self.position,
                    'proceeds': proceeds
                })
                
                # Reset position
                self.position = 0
                self.position_value = 0
            
            elif self.position < 0:  # Close short position
                # Calculate cost
                cost = current_price * abs(self.position) * (1 + self.transaction_cost)
                self.balance -= cost
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'close_short',
                    'price': current_price,
                    'shares': abs(self.position),
                    'cost': cost
                })
                
                # Reset position
                self.position = 0
                self.position_value = 0
        
        # Update position value
        if self.position != 0:
            self.position_value = current_price * self.position
        
        # Calculate reward (change in total portfolio value)
        total_value = self.balance + self.position_value
        prev_total_value = self.history['total_value'][-1]
        reward = (total_value - prev_total_value) / prev_total_value
        
        # Apply risk management
        if self._risk_checks(action):
            reward -= 0.1  # Penalty for triggering risk management
        
        # Update history
        self.history['balance'].append(self.balance)
        self.history['position'].append(self.position)
        self.history['position_value'].append(self.position_value)
        self.history['total_value'].append(total_value)
        self.history['reward'].append(reward)
        
        # Update total reward
        self.total_reward += reward
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Create info dictionary
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'total_value': total_value,
            'reward': reward,
            'total_reward': self.total_reward
        }
        
        return obs, reward, done, info
    
    def _risk_checks(self, action):
        """
        Perform risk checks and take action if necessary
        
        Parameters:
        -----------
        action : int
            Action to take
            
        Returns:
        --------
        bool
            Whether risk management was triggered
        """
        # Calculate maximum loss threshold
        max_loss = -0.1 * self.initial_balance
        
        # Check if portfolio value has dropped below threshold
        total_value = self.balance + self.position_value
        if total_value - self.initial_balance < max_loss:
            # Close all positions
            self.close_positions()
            return True
        
        return False
    
    def close_positions(self):
        """
        Close all positions
        """
        if self.position != 0:
            current_price = self.df.iloc[self.current_step]['close']
            
            if self.position > 0:  # Close long position
                proceeds = current_price * self.position * (1 - self.transaction_cost)
                self.balance += proceeds
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'close_long_risk',
                    'price': current_price,
                    'shares': self.position,
                    'proceeds': proceeds
                })
            
            elif self.position < 0:  # Close short position
                cost = current_price * abs(self.position) * (1 + self.transaction_cost)
                self.balance -= cost
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'close_short_risk',
                    'price': current_price,
                    'shares': abs(self.position),
                    'cost': cost
                })
            
            # Reset position
            self.position = 0
            self.position_value = 0
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Parameters:
        -----------
        mode : str
            Rendering mode
        """
        if mode == 'human':
            # Plot portfolio value
            plt.figure(figsize=(12, 6))
            plt.plot(self.history['total_value'])
            plt.title('Portfolio Value')
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.show()
            
            # Print summary
            print(f"Initial Balance: {self.initial_balance}")
            print(f"Final Balance: {self.balance}")
            print(f"Final Position Value: {self.position_value}")
            print(f"Final Total Value: {self.balance + self.position_value}")
            print(f"Return: {(self.balance + self.position_value - self.initial_balance) / self.initial_balance:.2%}")
            print(f"Number of Trades: {len(self.trades)}")
    
    def save_results(self, filename):
        """
        Save trading results
        
        Parameters:
        -----------
        filename : str
            Filename to save results
        """
        # Create results dataframe
        results = pd.DataFrame({
            'balance': self.history['balance'],
            'position': self.history['position'],
            'position_value': self.history['position_value'],
            'total_value': self.history['total_value'],
            'reward': self.history['reward']
        })
        
        # Save results
        results.to_csv(filename, index=False)
        
        # Save trades
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filename.replace('.csv', '_trades.csv'), index=False)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/gs_features.csv', parse_dates=['date'])
    
    # Create environment
    env = EnhancedTradingEnv(df)
    
    # Test environment with random actions
    obs = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    
    # Render results
    env.render()
    
    # Save results
    env.save_results('results/random_agent_results.csv') 