from rich import print as rprint 
# abhi chor rhe h rich se error highlighting seekhna h 

import os
import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go

import os
import pandas as pd

# Step 1: Read all .csv files in the specified directory
def read_csv_files(directory):
    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            # Check if the file contains required columns
            with open(filepath, 'r') as file:
                header = file.readline().strip().lower().split(',')
                if 'open' in header and 'high' in header and 'low' in header and 'close' in header:
                    df = pd.read_csv(filepath)
                    data_frames.append(df)
                else:
                    print("Not a valid CandleStick data file:", filepath)
                    continue
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

# Step 2: Clean the DataFrame
def cleaner(df):
    # Find and rename datetime column to 'timestamp'
    datetime_columns = ['timestamp', 'time', 'date&time','date','Date','Time']  # Add other synonyms as needed
    for column in df.columns:
        if column.lower() in datetime_columns:
            df.rename(columns={column: 'timestamp'}, inplace=True)
            break

        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()

        # Ensure 'timestamp' column is converted to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort DataFrame by 'timestamp' before removing duplicates
        df.sort_values(by='timestamp', inplace=True)

        # Remove duplicated timestamps, keeping only the latest instance
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    
    rprint(df)        #Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    rprint(df)
    return df
 

#make sure how to run calculate_indicators() for all candles 
# Step 3: Calculate technical indicators
def calculate_indicators(df):
    # sma 1 
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_15'] = df['close'].rolling(window=15).mean()
    df['SMA_45'] = df['close'].rolling(window=45).mean()
    df['SMA_90'] = df['close'].rolling(window=90).mean()
    df['SMA_3h'] = df['close'].rolling(window=180).mean()
    # RSI 2
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)

    # ema 3
    
    df['EMA_5'] = talib.EMA(df['close'], timeperiod=10)
    df['EMA_15'] = talib.EMA(df['close'], timeperiod=15)
    df['EMA_45'] = talib.EMA(df['close'], timeperiod=45)
    df['EMA_90'] = talib.EMA(df['close'], timeperiod=90)
    df['EMA_3h'] = talib.EMA(df['close'], timeperiod=180)
    
    #MACD 4
    macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_line'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    
    #BB 5
    upper_band, middle_band, lower_band = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_upper'] = upper_band
    df['BB_middle'] = middle_band
    df['BB_lower'] = lower_band

    #Stochastic Oscillator 6
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCH_slowk'] = slowk
    df['STOCH_slowd'] = slowd
    
    #Average True Range ATR 7
    df['ATR_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    #Volume Weighted Average Price VWAP 8
    df['VWAP'] = talib.WMA((df['high'] + df['low'] + df['close']) / 3, timeperiod=14)

    #Ichimoku Cloud 9
    tenkan_sen = talib.EMA((df['high'] + df['low']) / 2, timeperiod=9)
    kijun_sen = talib.EMA((df['high'] + df['low']) / 2, timeperiod=26)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = talib.EMA((df['high'] + df['low']) / 2, timeperiod=52)
    df['Ichimoku_tenkan_sen'] = tenkan_sen
    df['Ichimoku_kijun_sen'] = kijun_sen
    df['Ichimoku_senkou_span_a'] = senkou_span_a
    df['Ichimoku_senkou_span_b'] = senkou_span_b.shift(26)

    #MACD Histogram 10
    macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_hist'] = hist


    return df


'''

    SMA Analysis: 6 analyzers (SMA_good_1 to SMA_good_6)
    EMA Analysis: 2 analyzers (EMA_crossover, EMA_trend_strength)
    MACD Analysis: 5 analyzers (MACD_crossover, MACD_hist_trend, MACD_line_slope, MACD_hist_divergence, MACD_hist_reversal)
    Bollinger Bands Analysis: 4 analyzers (BB_breakout, BB_squeeze, BB_reversal, BB_trend_strength)
    Stochastic Oscillator Analysis: 5 analyzers (Stochastic_oversold, Stochastic_overbought, Stochastic_crossover, Stochastic_trend_reversal, Stochastic_trend_strength)
    ATR Analysis: 4 analyzers (ATR_breakout, ATR_trend_strength, ATR_trend_reversal, ATR_trend_strength)
    VWAP Analysis: 4 analyzers (VWAP_deviation, VWAP_trend, VWAP_breakout, VWAP_trend_reversal)
    Ichimoku Cloud Analysis: 5 analyzers (Cloud_breakout, Cloud_twist, Kumo_breakout, Tenkan_Kijun_crossover)
    MACD Histogram Analysis: 7 analyzers (MACD_hist_divergence, MACD_hist_trend, MACD_hist_reversal, MACD_hist_trend_strength, MACD_hist_slope, MACD_hist_signal_divergence)

Total analyzers: 6+2+5+4+5+4+4+5+7=426+2+5+4+5+4+4+5+7=42 analyzers

'''


# Step 4: Implement good/bad analyzers for indicators
def analyze_indicators(df):
    # Analyzing SMA
    df['SMA_good_1'] = np.where(df['SMA_5'] > df['SMA_15'], 1, 0)
    df['SMA_good_2'] = np.where(df['SMA_15']>df['SMA_45'] , 1, 0)
    df['SMA_good_3'] = np.where(df['SMA_45']>df['SMA_90'],1,0)
    df['SMA_good_4'] = np.where(df['SMA_45']>df['SMA_3h'],1,0)
    df['SMA_good_5'] = np.where(df['SMA_45']>df['SMA_3h'],1,0)
    df['SMA_good_6'] = np.where(df['SMA_90']>df['SMA_3h'],1,0)



    # Analysing EMA 
    # EMA crossover: EMA_5 crossing above EMA_15
    df['EMA_crossover'] = np.where(df['EMA_5'] > df['EMA_15'], 1, 0)
    # EMA trend strength: EMA_15 increasing consistently over the last 5 periods
    df['EMA_trend_strength'] = np.where(df['EMA_15'].diff(5).gt(0).all(), 1, 0)



    # Analysing MACD
    # MACD signal line crossover: MACD line crossing above signal line
    df['MACD_crossover'] = np.where(df['MACD_line'] > df['MACD_signal'], 1, 0)

    # MACD histogram trend: MACD histogram increasing consistently over the last 3 periods
    df['MACD_hist_trend'] = np.where(df['MACD_hist'].diff(3).gt(0).all(), 1, 0)

    # MACD line slope: MACD line's slope indicating momentum
    df['MACD_line_slope'] = np.where(df['MACD_line'].diff(3).gt(0), 1,
                                  np.where(df['MACD_line'].diff(3).lt(0), -1, 0))

    # MACD histogram divergence: MACD histogram diverging from MACD line
    df['MACD_hist_divergence'] = np.where(df['MACD_hist'].diff(3).gt(df['MACD_line'].diff(3)), 1,
                                       np.where(df['MACD_hist'].diff(3).lt(df['MACD_line'].diff(3)), -1, 0))




    # Analysing BBs
    # Bollinger Bands breakout: Price breaking above upper Bollinger Band
    df['BB_breakout'] = np.where(df['close'] > df['BB_upper'], 1, 0)

    # Bollinger Bands squeeze: Narrowing of Bollinger Bands (standard deviation decreasing)
    df['BB_squeeze'] = np.where(df['BB_upper'] - df['BB_lower'] < df['BB_upper'].rolling(window=20).std(), 1, 0)

    # Bollinger Bands reversal: Price reversing from upper or lower Bollinger Band
    df['BB_reversal'] = np.where((df['close'] < df['BB_lower']) | (df['close'] > df['BB_upper']), 1, 0)

    # Bollinger Bands trend strength: Bands widening or narrowing consistently
    df['BB_trend_strength'] = np.where(df['BB_upper'].diff(5).gt(0).all() | df['BB_lower'].diff(5).lt(0).all(), 1, 0)


    # Analysing Stochastic Oscillator 
    # Stochastic oversold/overbought conditions: %K line crossing above/below thresholds
    df['Stochastic_oversold'] = np.where(df['STOCH_slowk'] < 20, 1, 0)
    df['Stochastic_overbought'] = np.where(df['STOCH_slowk'] > 80, 1, 0)

    # Stochastic crossover: %K line crossing above %D line
    df['Stochastic_crossover'] = np.where(df['STOCH_slowk'] > df['STOCH_slowd'], 1, 0)
    
    # Stochastic trend reversal: %K or %D line changing direction after a prolonged trend
    df['Stochastic_trend_reversal'] = np.where(df['STOCH_slowk'].diff(3).gt(0) & df['STOCH_slowk'].diff(10).lt(0), 1, 0)

    # Stochastic trend strength: %K line increasing consistently over the last 5 periods
    df['Stochastic_trend_strength'] = np.where(df['STOCH_slowk'].diff(5).gt(0).all(), 1, 0)


    # Analysing ATR 
    # ATR volatility breakout: ATR value increasing significantly
    df['ATR_breakout'] = np.where(df['ATR_14'].diff(3).gt(0), 1, 0)

    # ATR trend strength: ATR increasing consistently over the last 5 periods
    df['ATR_trend_strength'] = np.where(df['ATR_14'].diff(5).gt(0).all(), 1, 0)
    # ATR trend reversal: ATR changing direction after a prolonged trend
    df['ATR_trend_reversal'] = np.where(df['ATR_14'].diff(3).gt(0) & df['ATR_14'].diff(10).lt(0), 1, 0)


    # Analysing VWAP 
    # VWAP deviation: Price deviating significantly from VWAP
    df['VWAP_deviation'] = np.where(df['close'] > df['VWAP'] + df['VWAP'].rolling(window=10).std(), 1, 0)

    # VWAP trend analysis: VWAP line increasing consistently over the last 5 periods
    df['VWAP_trend'] = np.where(df['VWAP'].diff(5).gt(0).all(), 1, 0)

    # VWAP breakout: Price breaking above or below VWAP significantly
    df['VWAP_breakout'] = np.where((df['close'] > df['VWAP'] + df['VWAP'].rolling(window=10).std()) |
                                (df['close'] < df['VWAP'] - df['VWAP'].rolling(window=10).std()), 1, 0)

    # VWAP trend reversal: VWAP changing direction after a prolonged trend
    df['VWAP_trend_reversal'] = np.where(df['VWAP'].diff(3).gt(0) & df['VWAP'].diff(10).lt(0), 1, 0)



    # Ichimoku Cloud Analysis 
    # Cloud breakout: Price breaking above cloud
    df['Cloud_breakout'] = np.where(df['close'] > df['Ichimoku_senkou_span_a'], 1, 0)
    
    # Cloud twist: Senkou Span A crossing above/below Senkou Span B
    df['Cloud_twist'] = np.where(df['Ichimoku_senkou_span_a'] > df['Ichimoku_senkou_span_b'], 1, 0)

    # Kumo breakout: Price breaking above/below Kumo cloud
    df['Kumo_breakout'] = np.where((df['close'] > df['Ichimoku_senkou_span_a']) & 
                                (df['close'] > df['Ichimoku_senkou_span_b']), 1,
                                np.where((df['close'] < df['Ichimoku_senkou_span_a']) & 
                                         (df['close'] < df['Ichimoku_senkou_span_b']), -1, 0))


    # Tenkan-sen and Kijun-sen crossover: Tenkan-sen line crossing above Kijun-sen line
    df['Tenkan_Kijun_crossover'] = np.where(df['Ichimoku_tenkan_sen'] > df['Ichimoku_kijun_sen'], 1, 0)


    # MACD Histogram Analysis
    # MACD Histogram divergence: MACD Histogram diverging from price movement
    df['MACD_hist_divergence'] = np.where(df['MACD_hist'].diff(3).gt(0), 1, 0)

    # MACD Histogram trend analysis: MACD Histogram increasing consistently over the last 5 periods
    df['MACD_hist_trend'] = np.where(df['MACD_hist'].diff(5).gt(0).all(), 1, 0)
    
    # MACD Histogram trend reversal: MACD Histogram changing direction after a prolonged trend
    df['MACD_hist_reversal'] = np.where(df['MACD_hist'].diff(3).gt(0) & df['MACD_hist'].diff(10).lt(0), 1, 0)

    # MACD Histogram trend strength: MACD Histogram increasing consistently over the last 5 periods
    df['MACD_hist_trend_strength'] = np.where(df['MACD_hist'].diff(5).gt(0).all(), 1, 0)

    # MACD Histogram slope: Slope of MACD Histogram indicating momentum
    df['MACD_hist_slope'] = np.where(df['MACD_hist'].diff(3).gt(0), 1,
                                  np.where(df['MACD_hist'].diff(3).lt(0), -1, 0))

    # MACD Histogram divergence from signal line: MACD Histogram diverging from MACD signal line
    df['MACD_hist_signal_divergence'] = np.where(df['MACD_hist'].diff(3).gt(df['MACD_signal'].diff(3)), 1,
                                              np.where(df['MACD_hist'].diff(3).lt(df['MACD_signal'].diff(3)), -1, 0))



    # Analysing RSI     
    # RSI oversold/overbought conditions: RSI crossing above/below thresholds
    df['RSI_oversold'] = np.where(df['RSI_14'] < 30, 1, 0)
    df['RSI_overbought'] = np.where(df['RSI_14'] > 70, 1,0)
    
    # RSI trend analysis: RSI increasing consistently over the last 5 periods
    df['RSI_trend'] = np.where(df['RSI_14'].diff(5).gt(0).all(), 1, 0)

    # RSI divergence: RSI diverging from price movement
    df['RSI_divergence'] = np.where(df['RSI_14'].diff(3).gt(0), 1, 0)
    
    # RSI overbought/oversold zones: RSI staying in overbought/oversold zones for a duration
    df['RSI_oversold_duration'] = df['RSI_14'].apply(lambda x: 1 if x < 30 else -1 if x > 70 else 0)





    # Analyze other indicators similarly
    return df
   
'''
# Step 6: Save buy/sell signals
def save_signals(df):
    # Save buy/sell signals with datetime and LTP
    buy_signals = df[df['signal'] == 'Buy'][['timestamp', 'LTP']]
    sell_signals = df[df['signal'] == 'Sell'][['timestamp', 'LTP']]
    
    # Save signals in object format or any desired format
    
    return buy_signals, sell_signals
'''

# Step 5: Calculate arithmetic mean and generate buy/sell signals
def generate_signals(df):
    # Calculate arithmetic mean of indicator values
    df['arithmetic_mean'] = (df['SMA_good_1'] + df['SMA_good_2'] + df['SMA_good_3'] + df['SMA_good_4'] +
                             df['SMA_good_5'] + df['SMA_good_6'] +
                             df['EMA_crossover'] + df['EMA_trend_strength'] +
                             df['MACD_crossover'] + df['MACD_hist_trend'] + df['MACD_line_slope'] + df['MACD_hist_divergence'] +
                             df['BB_breakout'] + df['BB_squeeze'] + df['BB_reversal'] + df['BB_trend_strength'] +
                             df['Stochastic_oversold'] + df['Stochastic_overbought'] + df['Stochastic_crossover'] +
                             df['Stochastic_trend_reversal'] + df['Stochastic_trend_strength'] +
                             df['ATR_breakout'] + df['ATR_trend_strength'] + df['ATR_trend_reversal'] +
                             df['VWAP_deviation'] + df['VWAP_trend'] + df['VWAP_breakout'] + df['VWAP_trend_reversal'] +
                             df['Cloud_breakout'] + df['Cloud_twist'] + df['Kumo_breakout'] + df['Tenkan_Kijun_crossover'] +
                             df['MACD_hist_divergence'] + df['MACD_hist_trend'] + df['MACD_hist_reversal'] + df['MACD_hist_trend_strength'] +
                             df['RSI_oversold'] + df['RSI_overbought'] + df['RSI_trend'] + df['RSI_divergence'] + df['RSI_oversold_duration']
                             ) / 42
    
    # Convert arithmetic mean into percentage
    df['arithmetic_mean_percentage'] = df['arithmetic_mean'] * 100
    
    # Generate buy signals when arithmetic mean > 70 and sell signals when < 35
    df.loc[df['arithmetic_mean_percentage'] > 30, 'signal'] = 'Buy'
    df.loc[df['arithmetic_mean_percentage'] < 30, 'signal'] = 'Sell'
    
    # Filter buy signals for plotting
    buy_signals = df[df['signal'] == 'Buy']
#    sell_signals = df[df['signal'] == 'Sell']    
    # Call plot_graph() function with buy_signals
    plot_graph(df, buy_signals)
    
    return df


# Step 7: Plot the graph using Plotly

def plot_graph(df, buy_signals):
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Candlestick'))

    # Create a list to store the y-values for buy signals
    buy_prices = []

    # Iterate through buy signals and store corresponding close prices
    for index in buy_signals.index:
        buy_prices.append(df.loc[index, 'close'])

    # Add buy signals to the graph using stored y-values
    fig.add_trace(go.Scatter(x=buy_signals['timestamp'],
                             y=buy_prices,
                             mode='markers',
                             marker={'color': 'green', 'symbol': 'triangle-up', 'size': 19},
                             name='Buy Signals'))

    # Add checkbox for 'Show Signals' option
    fig.update_layout(
        title='Financial Data Analysis with Buy/Sell Signals',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label='Show Signals',
                        method='update',
                        args=[{'visible': [True, True]}, {'title': 'Financial Data Analysis with Buy/Sell Signals'}]
                    ),
                    dict(
                        label='Hide Signals',
                        method='update',
                        args=[{'visible': [True, False]}, {'title': 'Financial Data Analysis without Signals'}]
                    )
                ],
                direction='down',
                showactive=True,
                x=1.05,
                y=0.5
            )
        ]
    )

    fig.show()


# Main program
if __name__ == "__main__":
    # Specify directory containing .csv files
    data_directory = './'
    
    # Read data from .csv files
    combined_data = read_csv_files(data_directory)
    
    # Remove duplicates
    cleaned_data = cleaner(combined_data)
    
    # Calculate technical indicators
    data_with_indicators = calculate_indicators(cleaned_data)
    
    # Analyze indicators
    data_analyzed = analyze_indicators(data_with_indicators)
    
    # Generate buy/sell signals
    signals_data = generate_signals(data_analyzed)
    
    # Save buy/sell signals
#    buy_signals, sell_signals = save_signals(signals_data)
    

