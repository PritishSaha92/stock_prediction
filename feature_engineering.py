import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from statsmodels.tsa.arima.model import ARIMA
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class SentimentAnalyzer:
    """
    Class for sentiment analysis using BERT
    """
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize the BERT model and tokenizer
        
        Parameters:
        -----------
        model_name : str
            Name of the pre-trained BERT model
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = TFBertModel.from_pretrained(model_name)
    
    def get_bert_embeddings(self, news_headlines, max_length=64):
        """
        Get BERT embeddings for news headlines
        
        Parameters:
        -----------
        news_headlines : list
            List of news headlines
        max_length : int
            Maximum length of tokens
            
        Returns:
        --------
        numpy.ndarray
            BERT embeddings
        """
        inputs = self.tokenizer(
            news_headlines, 
            return_tensors='tf', 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        outputs = self.bert_model(inputs)
        return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()

class TimeSeriesDecomposer:
    """
    Class for time series decomposition using Fourier-ARIMA
    """
    def __init__(self, arima_order=(5, 1, 0)):
        """
        Initialize the decomposer
        
        Parameters:
        -----------
        arima_order : tuple
            Order of the ARIMA model (p, d, q)
        """
        self.arima_order = arima_order
    
    def decompose_series(self, series):
        """
        Decompose time series into trend, seasonality, and residuals
        
        Parameters:
        -----------
        series : pandas.Series
            Time series to decompose
            
        Returns:
        --------
        tuple
            (trend, seasonality, residuals)
        """
        # Fourier Transform for seasonality
        fft_vals = fft(series.values)
        freq = fftfreq(len(series))
        
        # ARIMA for trend
        arima = ARIMA(series, order=self.arima_order).fit()
        trend = arima.predict()
        
        # Residuals
        seasonality = np.abs(fft_vals.real)[:len(series)]
        residuals = series - trend - seasonality
        
        return trend, seasonality, residuals
    
    def plot_decomposition(self, series, trend, seasonality, residuals):
        """
        Plot the decomposition
        
        Parameters:
        -----------
        series : pandas.Series
            Original time series
        trend : numpy.ndarray
            Trend component
        seasonality : numpy.ndarray
            Seasonality component
        residuals : numpy.ndarray
            Residual component
        """
        plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(series)
        plt.title('Original Time Series')
        
        plt.subplot(4, 1, 2)
        plt.plot(trend)
        plt.title('Trend Component')
        
        plt.subplot(4, 1, 3)
        plt.plot(seasonality)
        plt.title('Seasonality Component')
        
        plt.subplot(4, 1, 4)
        plt.plot(residuals)
        plt.title('Residual Component')
        
        plt.tight_layout()
        plt.show()

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with OHLCV columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with technical indicators
    """
    # Make a copy of the dataframe
    result = df.copy()
    
    # Moving Averages
    result['MA5'] = result['close'].rolling(window=5).mean()
    result['MA20'] = result['close'].rolling(window=20).mean()
    result['MA50'] = result['close'].rolling(window=50).mean()
    
    # MACD
    result['EMA12'] = result['close'].ewm(span=12, adjust=False).mean()
    result['EMA26'] = result['close'].ewm(span=26, adjust=False).mean()
    result['MACD'] = result['EMA12'] - result['EMA26']
    result['Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = result['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    result['BB_middle'] = result['close'].rolling(window=20).mean()
    result['BB_std'] = result['close'].rolling(window=20).std()
    result['BB_upper'] = result['BB_middle'] + 2 * result['BB_std']
    result['BB_lower'] = result['BB_middle'] - 2 * result['BB_std']
    
    # Drop NaN values
    result = result.dropna()
    
    return result

def merge_features(stock_df, eigen_df, sentiment_df=None):
    """
    Merge stock data with eigen-portfolios and sentiment data
    
    Parameters:
    -----------
    stock_df : pandas.DataFrame
        Stock data with technical indicators
    eigen_df : pandas.DataFrame
        Eigen-portfolio data
    sentiment_df : pandas.DataFrame, optional
        Sentiment data
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe with all features
    """
    # Merge stock data with eigen-portfolios
    merged_df = pd.merge(stock_df, eigen_df, on='date', how='inner')
    
    # Merge with sentiment data if provided
    if sentiment_df is not None:
        merged_df = pd.merge(merged_df, sentiment_df, on='date', how='inner')
    
    return merged_df

if __name__ == "__main__":
    # Load stock data
    gs_df = pd.read_csv('data/GS.csv', parse_dates=['date'])
    
    # Load eigen-portfolios
    eigen_df = pd.read_csv('data/eigen_portfolios.csv', parse_dates=['date'])
    
    # Calculate technical indicators
    gs_with_indicators = calculate_technical_indicators(gs_df)
    
    # Decompose time series
    decomposer = TimeSeriesDecomposer()
    trend, seasonality, residuals = decomposer.decompose_series(gs_with_indicators['close'])
    
    # Add decomposition to dataframe
    gs_with_indicators['trend'] = trend
    gs_with_indicators['seasonality'] = seasonality
    gs_with_indicators['residual'] = residuals
    
    # Merge features
    merged_features = merge_features(gs_with_indicators, eigen_df)
    
    # Save merged features
    merged_features.to_csv('data/gs_features.csv', index=False)
    
    print("Feature engineering completed successfully!") 