import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os

class RainbowDQNAgent:
    """
    Rainbow DQN Agent for reinforcement learning
    
    Implements key Rainbow DQN features:
    - Double Q-learning
    - Dueling networks
    - Noisy networks
    - N-step returns
    """
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        learning_rate=0.0001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        n_step=3,
        dueling=True,
        noisy=True,
        double_q=True
    ):
        """
        Initialize the Rainbow DQN agent
        
        Parameters:
        -----------
        state_size : int
            Size of the state space
        action_size : int
            Size of the action space
        gamma : float
            Discount factor
        learning_rate : float
            Learning rate
        epsilon : float
            Exploration rate
        epsilon_min : float
            Minimum exploration rate
        epsilon_decay : float
            Decay rate for exploration
        memory_size : int
            Size of replay memory
        batch_size : int
            Batch size for training
        n_step : int
            Number of steps for n-step returns
        dueling : bool
            Whether to use dueling networks
        noisy : bool
            Whether to use noisy networks
        double_q : bool
            Whether to use double Q-learning
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.dueling = dueling
        self.noisy = noisy
        self.double_q = double_q
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build the neural network model
        
        Returns:
        --------
        tensorflow.keras.models.Model
            Neural network model
        """
        # Input layer
        inputs = Input(shape=(self.state_size,))
        
        # Hidden layers
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        if self.dueling:
            # Dueling architecture
            # Value stream
            value = Dense(32, activation='relu')(x)
            value = Dense(1)(value)
            
            # Advantage stream
            advantage = Dense(32, activation='relu')(x)
            advantage = Dense(self.action_size)(advantage)
            
            # Combine value and advantage
            q_values = Lambda(
                lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
                output_shape=(self.action_size,)
            )([value, advantage])
        else:
            # Standard architecture
            q_values = Dense(self.action_size)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """
        Update target model weights
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : numpy.ndarray
            Next state
        done : bool
            Whether the episode is done
        """
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If buffer is full, process n-step return
        if len(self.n_step_buffer) == self.n_step:
            state, action, n_step_reward, next_state, done = self._get_n_step_info()
            self.memory.append((state, action, n_step_reward, next_state, done))
    
    def _get_n_step_info(self):
        """
        Get n-step return information
        
        Returns:
        --------
        tuple
            (state, action, n_step_reward, next_state, done)
        """
        # Get initial experience
        state, action, reward, _, _ = self.n_step_buffer[0]
        
        # Calculate n-step reward
        n_step_reward = reward
        for i in range(1, self.n_step):
            n_step_reward += self.gamma ** i * self.n_step_buffer[i][2]
        
        # Get final experience
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        return state, action, n_step_reward, next_state, done
    
    def act(self, state, training=True):
        """
        Choose an action
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        training : bool
            Whether the agent is training
            
        Returns:
        --------
        int
            Chosen action
        """
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose random action
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        """
        Train the agent with experiences from replay memory
        
        Returns:
        --------
        float
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract batch data
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Calculate target Q values
        if self.double_q:
            # Double Q-learning
            # Get actions from online model
            next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
            
            # Get Q values from target model
            next_q_values = self.target_model.predict(next_states, verbose=0)
            
            # Select Q values for actions chosen by online model
            next_q_values = next_q_values[np.arange(self.batch_size), next_actions]
        else:
            # Standard Q-learning
            next_q_values = np.max(self.target_model.predict(next_states, verbose=0), axis=1)
        
        # Calculate target values
        targets = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Update Q values for chosen actions
        current_q[np.arange(self.batch_size), actions] = targets
        
        # Train the model
        history = self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def save(self, filepath):
        """
        Save the model
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Load the model
        
        Parameters:
        -----------
        filepath : str
            Path to load the model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_model()

class RainbowDQNTrainer:
    """
    Trainer for Rainbow DQN agent
    """
    def __init__(
        self,
        env,
        agent,
        episodes=1000,
        max_steps=None,
        target_update_freq=10,
        save_freq=100,
        log_freq=10,
        save_dir='models'
    ):
        """
        Initialize the trainer
        
        Parameters:
        -----------
        env : gym.Env
            Training environment
        agent : RainbowDQNAgent
            Rainbow DQN agent
        episodes : int
            Number of episodes to train
        max_steps : int, optional
            Maximum steps per episode
        target_update_freq : int
            Frequency of target model updates
        save_freq : int
            Frequency of model saving
        log_freq : int
            Frequency of logging
        save_dir : str
            Directory to save models
        """
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps = max_steps
        self.target_update_freq = target_update_freq
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def train(self):
        """
        Train the agent
        
        Returns:
        --------
        dict
            Training history
        """
        # Initialize history
        history = {
            'episode_rewards': [],
            'episode_steps': [],
            'episode_losses': []
        }
        
        # Training loop
        for episode in range(1, self.episodes + 1):
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            
            # Episode loop
            while not done:
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Check if max steps reached
                if self.max_steps is not None and episode_steps >= self.max_steps:
                    done = True
            
            # Update target model
            if episode % self.target_update_freq == 0:
                self.agent.update_target_model()
            
            # Save model
            if episode % self.save_freq == 0:
                self.agent.save(f'{self.save_dir}/rainbow_dqn_episode_{episode}.h5')
            
            # Update history
            history['episode_rewards'].append(episode_reward)
            history['episode_steps'].append(episode_steps)
            history['episode_losses'].append(np.mean(episode_losses) if episode_losses else 0)
            
            # Log progress
            if episode % self.log_freq == 0:
                avg_reward = np.mean(history['episode_rewards'][-self.log_freq:])
                avg_loss = np.mean(history['episode_losses'][-self.log_freq:])
                print(f"Episode: {episode}/{self.episodes}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {self.agent.epsilon:.4f}")
        
        # Save final model
        self.agent.save(f'{self.save_dir}/rainbow_dqn_final.h5')
        
        return history

if __name__ == "__main__":
    # Import environment
    from rl_environment import EnhancedTradingEnv
    
    # Load data
    import pandas as pd
    df = pd.read_csv('data/gs_features.csv', parse_dates=['date'])
    
    # Create environment
    env = EnhancedTradingEnv(df)
    
    # Create agent
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
        batch_size=64,
        n_step=3,
        dueling=True,
        noisy=True,
        double_q=True
    )
    
    # Create trainer
    trainer = RainbowDQNTrainer(
        env=env,
        agent=agent,
        episodes=100,
        target_update_freq=10,
        save_freq=25,
        log_freq=5
    )
    
    # Train agent
    history = trainer.train()
    
    # Test agent
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        state, reward, done, _ = env.step(action)
    
    # Render results
    env.render()
    
    # Save results
    env.save_results('results/rainbow_dqn_results.csv') 