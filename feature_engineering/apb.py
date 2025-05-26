import pandas as pd
import numpy as np
import os, duckdb

def calculate_twap(date, data_type = 'five_minute'):
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    if os.path.exists(file_path):
        sql = f"""
        SELECT security_code as SecuCode, trading_day as TradingDay, AVG((open_price+close_price)/2) as avg_price
        FROM read_parquet('{file_path}')
        GROUP BY SecuCode, TradingDay
        """
        data = duckdb.query(sql).df()
        data.dropna(inplace=True)
        return data

def day_freq_apb(group):

    # Sort by TradingDay
    group = group.sort_values('TradingDay')
    
    # Calculate 20-day rolling window values for arithmetic mean
    arithmetic_mean = group['vwap'].rolling(window=20, min_periods=1).mean()
    
    # Calculate turnover-volume weighted mean using vectorized operations
    # First calculate rolling sum of (vwap * volume)
    weighted_sum = (group['vwap'] * group['TurnoverVolume']).rolling(window=20, min_periods=1).sum()
    # Then calculate rolling sum of volume
    volume_sum = group['TurnoverVolume'].rolling(window=20, min_periods=1).sum()
    # Finally calculate weighted mean
    weighted_mean = weighted_sum / volume_sum
    
    # Calculate the ratio: arithmetic mean / weighted mean
    ratio = arithmetic_mean / weighted_mean
    
    # Handle any inf/nan values
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    
    return ratio

def apb(data):
    df = data
    df['TradingDay'] = pd.to_datetime(df['TradingDay'])
    df['vwap'] = df['TurnoverValue']/df['TurnoverVolume']

    results = []
    trading_days = df['TradingDay'].unique()
    # Process each stock on each day
    for date in trading_days:
        try:
            # Fetch intraday transaction data for this stock on this day
            twap = calculate_twap(date)
            if twap is not None and not twap.empty:
                results.append(twap)
        except Exception as e:
            print(f"Error processing on {date}: {e}")
            continue

    if results:
        df_results = pd.concat(results, ignore_index=True)
        df = pd.merge(df, df_results, on=['TradingDay', 'SecuCode'], how='left')
    
    #停牌的时候,分钟数据中没有对应数据,avg_price会merge出nan
    df['apb'] = (df['avg_price']/df['vwap']).replace([np.inf,-np.inf],np.nan)
    
    # Apply day_freq_apb to each stock group - using vectorized operations for better performance
    print("Calculating daily frequency APB...")
    df['day_freq_apb'] = df.groupby('SecuCode').apply(day_freq_apb).reset_index(level=0, drop=True)
    
    # Return both APB values
    return df[['TradingDay', 'InnerCode', 'apb', 'day_freq_apb']]
