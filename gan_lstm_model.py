import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class TimeSeriesGenerator:
    """
    Class for generating time series data for training
    """
    def __init__(self, data, sequence_length, batch_size=32, shuffle=True):
        """
        Initialize the generator
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data
        sequence_length : int
            Length of input sequences
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle the data
        """
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data) - sequence_length)
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """
        Get the number of batches
        """
        return (len(self.data) - self.sequence_length) // self.batch_size
    
    def __getitem__(self, index):
        """
        Get a batch of data
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self.data[i:i + self.sequence_length] for i in batch_indices])
        y = np.array([self.data[i + self.sequence_length, 0] for i in batch_indices])
        return X, y.reshape(-1, 1)

class GANLSTM:
    """
    GAN-LSTM model for time series forecasting
    """
    def __init__(self, input_shape, latent_dim=50):
        """
        Initialize the GAN-LSTM model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        latent_dim : int
            Dimension of latent space
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.scaler = MinMaxScaler()
        
        # Build models
        self._build_generator()
        self._build_discriminator()
        self._build_gan()
    
    def _build_generator(self):
        """
        Build the generator model
        """
        # Generator input
        generator_input = Input(shape=self.input_shape)
        
        # LSTM layers
        x = LSTM(128, return_sequences=True)(generator_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = LSTM(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        generator_output = Dense(1)(x)
        
        # Create model
        self.generator = Model(generator_input, generator_output, name="generator")
        self.generator.compile(optimizer=Adam(0.0002, beta_1=0.5), loss='mse')
    
    def _build_discriminator(self):
        """
        Build the discriminator model
        """
        # Discriminator input
        discriminator_input = Input(shape=self.input_shape)
        
        # LSTM layers
        x = LSTM(64, return_sequences=True)(discriminator_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = LSTM(32)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        discriminator_output = Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.discriminator = Model(discriminator_input, discriminator_output, name="discriminator")
        self.discriminator.compile(optimizer=Adam(0.0002, beta_1=0.5), loss='binary_crossentropy')
    
    def _build_gan(self):
        """
        Build the combined GAN model
        """
        # Freeze discriminator weights during generator training
        self.discriminator.trainable = False
        
        # GAN input
        gan_input = Input(shape=self.input_shape)
        
        # Generator output
        generator_output = self.generator(gan_input)
        
        # Discriminator output
        gan_output = self.discriminator(gan_input)
        
        # Create model
        self.gan = Model(gan_input, [generator_output, gan_output], name="gan")
        self.gan.compile(
            optimizer=Adam(0.0002, beta_1=0.5),
            loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.8, 0.2]
        )
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the GAN-LSTM model
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training input data
        y_train : numpy.ndarray
            Training target data
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        
        Returns:
        --------
        dict
            Training history
        """
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create generator
        generator = TimeSeriesGenerator(X_train_scaled, self.input_shape[0], batch_size)
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            'models/gan_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # Train generator
        history = self.generator.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint]
        )
        
        return history.history
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        return self.generator.predict(X)
    
    def save_models(self, path_prefix='models/'):
        """
        Save models
        
        Parameters:
        -----------
        path_prefix : str
            Path prefix for saving models
        """
        self.generator.save(f'{path_prefix}generator.h5')
        self.discriminator.save(f'{path_prefix}discriminator.h5')
        self.gan.save(f'{path_prefix}gan.h5')
    
    def load_models(self, path_prefix='models/'):
        """
        Load models
        
        Parameters:
        -----------
        path_prefix : str
            Path prefix for loading models
        """
        self.generator = tf.keras.models.load_model(f'{path_prefix}generator.h5')
        self.discriminator = tf.keras.models.load_model(f'{path_prefix}discriminator.h5')
        self.gan = tf.keras.models.load_model(f'{path_prefix}gan.h5')

def prepare_data(df, target_col, feature_cols, sequence_length, test_size=0.2):
    """
    Prepare data for training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name
    feature_cols : list
        List of feature column names
    sequence_length : int
        Length of input sequences
    test_size : float
        Test size ratio
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, scaler)
    """
    # Extract features and target
    data = df[feature_cols + [target_col]].values
    
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length, 0])  # Assuming target is the first column
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/gs_features.csv', parse_dates=['date'])
    
    # Prepare data
    feature_cols = ['close', 'MACD', 'RSI', 'PC1', 'trend']
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        df, 'close', feature_cols, sequence_length=30
    )
    
    # Create and train model
    model = GANLSTM(input_shape=(30, len(feature_cols)))
    history = model.train(X_train, y_train, epochs=100, batch_size=32)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
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
    plt.savefig('results/gan_lstm_predictions.png')
    plt.show()
    
    # Save model
    model.save_models() 