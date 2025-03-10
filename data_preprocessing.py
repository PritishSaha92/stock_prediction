import datetime as DT
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def extract_ticker(in_df, ticker, start_date, end_date):
    """
    Extract data for a specific ticker within a date range and save to CSV
    
    Parameters:
    -----------
    in_df : DataFrame
        Input dataframe containing all stock data
    ticker : str
        Stock ticker symbol
    start_date : datetime
        Start date for extraction
    end_date : datetime
        End date for extraction
        
    Returns:
    --------
    DataFrame
        Extracted data for the specified ticker
    """
    t_df = in_df[in_df['ticker']==ticker]
    out_df = t_df[(t_df['date']>= start_date) & (t_df['date']<= end_date)].copy()
    out_df.drop('ticker', axis=1, inplace=True)
    base_path = 'data/'
    filename = base_path+ticker+'.csv'
    out_df.to_csv(filename, index=False)
    return out_df

def create_eigen_portfolios(df, n_components=5):
    """
    Create eigen-portfolios using PCA on stock returns
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with stock prices
    n_components : int
        Number of principal components to extract
        
    Returns:
    --------
    DataFrame
        Principal components of stock returns
    """
    # Calculate returns
    returns = df.pct_change().dropna()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(returns)
    
    # Create dataframe with principal components
    eigen_df = pd.DataFrame(
        principal_components, 
        index=returns.index,
        columns=[f'PC{i}' for i in range(1, n_components+1)]
    )
    
    # Add date column back
    eigen_df['date'] = returns.index
    
    return eigen_df

def process_finance50(all_df, start_date, end_date):
    """
    Process all tickers in finance50.txt and create a combined dataframe
    
    Parameters:
    -----------
    all_df : DataFrame
        Input dataframe containing all stock data
    start_date : datetime
        Start date for extraction
    end_date : datetime
        End date for extraction
        
    Returns:
    --------
    DataFrame
        Combined dataframe with close prices for all tickers
    """
    # Read finance50 tickers
    fin50 = np.genfromtxt('finance50.txt', dtype='str')
    
    # Process each ticker
    for i in range(0, len(fin50)):
        stock = fin50[i]
        stock_df = extract_ticker(all_df, stock, start_date, end_date)
        close_df = stock_df[['date', 'close']].copy()
        close_df.rename(columns={'date': 'date', 'close': stock}, inplace=True)
        
        if i == 0:
            fin_df = close_df.copy()
        else:
            fin_df = fin_df.merge(close_df, how='left', on='date', copy=True)
    
    # Save combined dataframe
    fin_df.to_csv('data/finance50.csv', index=False)
    
    return fin_df

if __name__ == "__main__":
    # Load historical stock prices
    all_df = pd.read_csv('data/historical_stock_prices.csv', header=0, parse_dates=[-1])
    
    # Set date range
    start_date = DT.datetime(2009, 1, 1)
    end_date = DT.datetime(2017, 12, 31)
    
    # Process finance50 stocks
    fin_df = process_finance50(all_df, start_date, end_date)
    
    # Create eigen-portfolios
    eigen_df = create_eigen_portfolios(fin_df.drop('date', axis=1))
    
    # Save eigen-portfolios
    eigen_df.to_csv('data/eigen_portfolios.csv', index=False)
    
    print("Data preprocessing completed successfully!") 