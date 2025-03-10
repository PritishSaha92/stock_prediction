import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from rl_environment import EnhancedTradingEnv
from rainbow_dqn_agent import RainbowDQNAgent, RainbowDQNTrainer

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning
    """
    def __init__(
        self,
        env,
        n_trials=50,
        study_name='rainbow_dqn_optimization',
        storage=None,
        direction='maximize',
        n_jobs=1
    ):
        """
        Initialize the Bayesian optimizer
        
        Parameters:
        -----------
        env : gym.Env
            Training environment
        n_trials : int
            Number of trials
        study_name : str
            Name of the study
        storage : str, optional
            Storage URL for the study
        direction : str
            Direction of optimization ('maximize' or 'minimize')
        n_jobs : int
            Number of parallel jobs
        """
        self.env = env
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.n_jobs = n_jobs
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True
        )
    
    def objective(self, trial):
        """
        Objective function for optimization
        
        Parameters:
        -----------
        trial : optuna.Trial
            Trial object
            
        Returns:
        --------
        float
            Objective value
        """
        # Define hyperparameters to optimize
        gamma = trial.suggest_float('gamma', 0.8, 0.999)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        epsilon_decay = trial.suggest_float('epsilon_decay', 0.9, 0.999)
        memory_size = trial.suggest_categorical('memory_size', [1000, 5000, 10000, 20000])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        n_step = trial.suggest_int('n_step', 1, 5)
        
        # Create agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        agent = RainbowDQNAgent(
            state_size=state_size,
            action_size=action_size,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            batch_size=batch_size,
            n_step=n_step,
            dueling=True,
            noisy=True,
            double_q=True
        )
        
        # Create trainer
        trainer = RainbowDQNTrainer(
            env=self.env,
            agent=agent,
            episodes=20,  # Use fewer episodes for optimization
            target_update_freq=5,
            save_freq=100,  # Don't save during optimization
            log_freq=10,
            save_dir='models/optimization'
        )
        
        # Train agent
        history = trainer.train()
        
        # Calculate objective value (average reward over last 5 episodes)
        avg_reward = np.mean(history['episode_rewards'][-5:])
        
        return avg_reward
    
    def optimize(self):
        """
        Run optimization
        
        Returns:
        --------
        dict
            Best parameters
        """
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = self.study.best_params
        
        # Print best parameters
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Print best value
        print(f"Best value: {self.study.best_value}")
        
        return best_params
    
    def plot_optimization_history(self, save_path=None):
        """
        Plot optimization history
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(self.study)
        
        # Save plot if path is provided
        if save_path is not None:
            fig.write_image(save_path)
        
        return fig
    
    def plot_param_importances(self, save_path=None):
        """
        Plot parameter importances
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        # Plot parameter importances
        fig = optuna.visualization.plot_param_importances(self.study)
        
        # Save plot if path is provided
        if save_path is not None:
            fig.write_image(save_path)
        
        return fig
    
    def save_study(self, filepath):
        """
        Save study to file
        
        Parameters:
        -----------
        filepath : str
            Path to save the study
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save study
        pd.DataFrame({
            'number': [trial.number for trial in self.study.trials],
            'value': [trial.value for trial in self.study.trials],
            **{param: [trial.params.get(param, None) for trial in self.study.trials] 
               for param in self.study.best_params.keys()}
        }).to_csv(filepath, index=False)

def train_with_best_params(env, best_params, episodes=100):
    """
    Train agent with best parameters
    
    Parameters:
    -----------
    env : gym.Env
        Training environment
    best_params : dict
        Best parameters
    episodes : int
        Number of episodes
        
    Returns:
    --------
    tuple
        (agent, history)
    """
    # Create agent with best parameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=best_params.get('gamma', 0.99),
        learning_rate=best_params.get('learning_rate', 0.0001),
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=best_params.get('epsilon_decay', 0.995),
        memory_size=best_params.get('memory_size', 10000),
        batch_size=best_params.get('batch_size', 64),
        n_step=best_params.get('n_step', 3),
        dueling=True,
        noisy=True,
        double_q=True
    )
    
    # Create trainer
    trainer = RainbowDQNTrainer(
        env=env,
        agent=agent,
        episodes=episodes,
        target_update_freq=10,
        save_freq=25,
        log_freq=5,
        save_dir='models/best'
    )
    
    # Train agent
    history = trainer.train()
    
    return agent, history

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/gs_features.csv', parse_dates=['date'])
    
    # Create environment
    env = EnhancedTradingEnv(df)
    
    # Create optimizer
    optimizer = BayesianOptimizer(env, n_trials=50)
    
    # Run optimization
    best_params = optimizer.optimize()
    
    # Plot optimization history
    optimizer.plot_optimization_history('results/optimization_history.png')
    
    # Plot parameter importances
    optimizer.plot_param_importances('results/param_importances.png')
    
    # Save study
    optimizer.save_study('results/optimization_study.csv')
    
    # Train with best parameters
    agent, history = train_with_best_params(env, best_params, episodes=100)
    
    # Test agent
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        state, reward, done, _ = env.step(action)
    
    # Render results
    env.render()
    
    # Save results
    env.save_results('results/best_agent_results.csv') 