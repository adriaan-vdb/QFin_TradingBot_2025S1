import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Prompt logic
path_to_data = "HistData/UEC.csv" if input("1 for UEC, otherwise SOBER: ") == "1" else "HistData/SOBER_expanded.csv"

stma = 50
ltma = 200
tradeTimes = []

import matplotlib.pyplot as plt
import numpy as np

def graphData(df):

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot spread on primary y-axis
    ax1.plot(df['Timestamp'], df['spread'], label="Spread", color='blue')
    ax1.set_ylabel('Spread', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['Timestamp'], df['mid'], label="Mid Price", linestyle='--', color='orange')
    ax2.set_ylabel('Mid Price', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Title and layout
    plt.title('Bid-Ask Spread and Mid Price')
    fig.tight_layout()
    plt.show()

    # Mid Price and Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['mid'], label="Mid Price", color='blue')
    plt.plot(df['Timestamp'], df['stma'], label=f"{stma}-period SMA", color='green')
    plt.plot(df['Timestamp'], df['ltma'], label="LTMA", color='red')
    plt.plot(df['Timestamp'], df['ema'], label="EMA", linestyle='--', color='purple')
    plt.plot(df['Timestamp'], df['cumulativeAVG'], label="Cumulative Avg", linestyle='--', color='orange')
    plt.title('Mid Price & Moving Averages')
    plt.legend()
    plt.show()

    # Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['mid'], label="Mid Price", color='blue')
    plt.plot(df['Timestamp'], df['BB_upper'], label='Upper Bollinger Band', linestyle='--')
    plt.plot(df['Timestamp'], df['BB_mid'], label="Bollinger Mid", linestyle='-.')
    plt.plot(df['Timestamp'], df['BB_lower'], label="Lower Band")
    plt.fill_between(df['Timestamp'], df['BB_upper'], df['BB_lower'], color='gray', alpha=0.2)
    plt.title('Bollinger Bands')
    plt.legend()
    plt.show()

    # RSI
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['RSI'], label="RSI")
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('Relative Strength Index (RSI)')
    plt.legend()
    plt.show()

    # MACD
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['MACD'], label="MACD")
    plt.plot(df['Timestamp'], df['Signal'], label="Signal Line")
    plt.bar(df['Timestamp'], df['MACD'] - df['Signal'], color='gray', alpha=0.3, label="MACD Histogram")
    plt.title('MACD Indicator')
    plt.legend()
    plt.show()

    # --- ATR w/ Mid Price overlay ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['Timestamp'], df['ATR'], label="ATR", color='blue')
    ax1.set_ylabel('ATR', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(df['Timestamp'], df['mid'], label="Mid Price", linestyle='--', color='orange')
    ax2.set_ylabel('Mid Price', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('ATR & Mid Price Overlay')
    ln1, lb1 = ax1.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax2.legend(ln1+ln2, lb1+lb2, loc='best')
    plt.show()

    # --- Z-Score w/ Mid Price overlay ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['Timestamp'], df['zscore'], label="Z-Score", color='blue')
    ax1.axhline(2, color='red', linestyle='--', label='Z=2')
    ax1.axhline(-2, color='green', linestyle='--', label='Z=-2')
    ax1.set_ylabel('Z-Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(df['Timestamp'], df['mid'], label="Mid Price", color='orange')
    ax2.set_ylabel('Mid Price', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Z-Score & Mid Price Overlay')
    ln1, lb1 = ax1.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax2.legend(ln1+ln2, lb1+lb2, loc='best')
    plt.show()

    # Donchian Channels
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['mid'], label="Mid Price", color='blue')
    plt.plot(df['Timestamp'], df['donchian_high'], label="Donchian High", linestyle='--', color='red')
    plt.plot(df['Timestamp'], df['donchian_low'], label="Donchian Low", linestyle='--', color='green')
    plt.fill_between(df['Timestamp'], df['donchian_high'], df['donchian_low'], color='gray', alpha=0.2)
    plt.title('Donchian Channels')
    plt.legend()
    plt.show()

    # Keltner Channels
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['mid'], label="Mid Price", color='blue')
    plt.plot(df['Timestamp'], df['keltner_mid'], label="Keltner Mid (EMA)", linestyle='-.', color='purple')
    plt.plot(df['Timestamp'], df['keltner_upper'], label="Keltner Upper", linestyle='--', color='red')
    plt.plot(df['Timestamp'], df['keltner_lower'], label="Keltner Lower", linestyle='--', color='green')
    plt.fill_between(df['Timestamp'], df['keltner_upper'], df['keltner_lower'], color='gray', alpha=0.2)
    plt.title('Keltner Channels')
    plt.legend()
    plt.show()


# Additional indicator implementations:

def calculate_RSI(series, period=14):
    delta = series.diff(1)
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + gain / loss))
    return rsi

def calculate_MACD(series, EMA_fast_period=12, EMA_slow_period=26, signal_period=9):
    EMA_fast = series.ewm(span=EMA_fast_period, adjust=False).mean()
    EMA_slow = series.ewm(span=EMA_slow_period, adjust=False).mean()
    MACD_line = EMA_fast - EMA_slow
    Signal_line = MACD_line.ewm(span=signal_period, adjust=False).mean()
    return MACD_line, Signal_line

def calculate_ATR(df, period=14):
    high_low = df['Asks'] - df['Bids']
    high_close = np.abs(df['Asks'] - df['mid'].shift())
    low_close = np.abs(df['Bids'] - df['mid'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def switchCheck(stmaL, ltmaL, stmaC, ltmaC):
    if (stmaL < ltmaL) and (stmaC > ltmaC):
        return 1
    elif (stmaL > ltmaL) and (stmaC < ltmaC):
        return -1
    return 0

def createZscore(series, window=20):
    mean_ = series.rolling(window).mean()
    std_  = series.rolling(window).std()
    z = (series - mean_) / std_
    return z

def createDonchian(series, window=20):
    high = series.rolling(window).max()
    low  = series.rolling(window).min()
    return high, low

def createKeltner(series, atr, ema_period=20, multiplier=2):
    mid = series.ewm(span=ema_period, adjust=False).mean()
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    return mid, upper, lower

def createNewColumns(df):
    df['spread'] = df['Asks'] - df['Bids']
    df['mid'] = (df['Asks'] + df['Bids']) / 2
    df['stma'] = df['mid'].rolling(stma).mean()
    df['ltma'] = df['mid'].rolling(ltma).mean()
    df['cumulativeAVG'] = df['mid'].expanding().mean()

    df['ema'] = df['mid'].ewm(span=stma, adjust=False).mean()

    # Bollinger
    df['BB_mid'] = df['mid'].rolling(20).mean()
    df['BB_std'] = df['mid'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']

    # RSI
    df['RSI'] = calculate_RSI(df['mid'])

    # MACD
    macd_line, sig_line = calculate_MACD(df['mid'])
    df['MACD']   = macd_line
    df['Signal'] = sig_line

    # ATR
    df['ATR'] = calculate_ATR(df, 14)

    # Pearson correlations
    df['pearsonStma'] = df['mid'].rolling(stma).corr(df['stma'])
    df['pearsonLtma'] = df['mid'].rolling(ltma).corr(df['ltma'])

    # Crossovers
    df['stmaShift'] = df['stma'].shift(1)
    df['ltmaShift'] = df['ltma'].shift(1)
    df['crossOvers'] = df.apply(lambda x: switchCheck(x['stmaShift'], x['ltmaShift'], x['stma'], x['ltma']), axis=1)
    df.drop(['stmaShift', 'ltmaShift'], axis=1, inplace=True)

    # Z-Score
    df['zscore'] = createZscore(df['mid'], window=20)

    # Donchian Channels
    df['donchian_high'], df['donchian_low'] = createDonchian(df['mid'], window=20)

    # Keltner Channels
    k_mid, k_upper, k_lower = createKeltner(df['mid'], df['ATR'], ema_period=20, multiplier=2)
    df['keltner_mid']   = k_mid
    df['keltner_upper'] = k_upper
    df['keltner_lower'] = k_lower

    return df

def csvToDf():
    df = pd.read_csv(path_to_data)
    df.rename(columns={"Unnamed: 0": "Timestamp"}, inplace=True)
    return df

def main():
    df = csvToDf()
    df = createNewColumns(df)
    print(df.head())
    graphData(df)

main()
