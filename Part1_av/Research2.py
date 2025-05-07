import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

path_to_data = "HistData/UEC.csv" if input() == "1" else "HistData/SOBER.csv"
# path_to_data = r"HistData/UEC.csv"
stma = 50  # Short term moving average
ltma = 200  # Long term moving average
 
tradeTimes = []
 
def graphData(df):  # This function graphs the data
    fig, ax1 = plt.subplots()
    ax1.plot(df['Timestamp'], df['mid'],  label="Mid Price")
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.plot(df['Timestamp'], df['pearsonLtma'], label="pearsonLtma")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
 
    #plt.plot(df['Timestamp'], df['stma'], color='r', label="STMA")
    #plt.plot(df['Timestamp'], df['ltma'], color='g', label="LTMA")

    ax1.plot(df['Timestamp'], df['cumulativeAVG'], label="Cumulative Average", color='orange', linestyle='--')

   
    buys = False
    if buys:
        for i in tradeTimes:
            if i[1] == 'b': colors = 'b'
            else: colors = 'y'
            plt.axvline(x=i[0], color=colors)
 
    plt.show()
 
def rolling_pearson(x, y):
    if np.isnan(x).any() or np.isnan(y).any():  # Check for NaNs
        return np.nan
    return pearsonr(x, y)[0]  # Pearson correlation coefficient
 
def determinePearsons(movingAverage, x, y): # Returns the actual pearson correlation
    x_values = pd.Series(x)
    y_values = pd.Series(y)
    return x_values.rolling(movingAverage).corr(y_values)  # Native Pandas method for rolling correlation
 
def switchCheck(stmaL, ltmaL, stmaC, ltmaC):  # this function determines if there was a switch in which average had the highest value
        if (stmaL < ltmaL) and (stmaC > ltmaC):   # return 0 = no cross, 1 = stma moving above, -1 = stma moving below
            return 1
        elif (stmaL > ltmaL) and (stmaC < ltmaC):
            return -1
        return 0
 
def createNewColumns(df):  # This function adds new columns to the dataframe
 
    df['spread'] = df['Asks'] - df['Bids']  # spread
    df['mid'] = (df['Asks'] + df['Bids']) / 2  # average of asks and bids
    df['stma'] = df['mid'].rolling(stma).mean()
    df['ltma'] = df['mid'].rolling(ltma).mean()

    # print(df.iterrows())

    df['cumulativeAVG'] = df['mid'].expanding().mean()
    print()
    print(f"-----> {df.iloc[::100]}")


    print('Calculating Pearson correlations...')
    df['pearsonStma'] = determinePearsons(stma, df['Timestamp'], df['stma'])
    df['pearsonLtma'] = determinePearsons(ltma, df['Timestamp'], df['ltma'])
    print('Done.')
 
    print('Starting cross over calculations...')
    df["stmaShift"] = df['stma'].shift(1)   # temp columns to make apply funciton faster
    df["ltmaShift"] = df['ltma'].shift(1)
    df['crossOvers'] = df.apply(lambda x: switchCheck(x['stmaShift'], x['ltmaShift'], x['stma'], x['ltma']), axis=1)
    df = df.drop('stmaShift',axis='columns')
    df = df.drop('ltmaShift',axis='columns')  # deleting temp columns
    print('Cross over calculations complete...')
   
    return df
 
def csvToDf():  # This function turns the CSV file into a pandas DataFrame
    df = pd.read_csv(path_to_data)
    df.rename(columns={"Unnamed: 0": "Timestamp"}, inplace=True)  # Fix column name
    return df
 
def trade(row, currentlyHolding, holdPrice, minPearsonToBuy):
    tradeType = None
    if not currentlyHolding:   # will buy if stma pearson is positive and stma moves below ltma
        if (row['crossOvers'] == 1) and (row['pearsonStma'] >=minPearsonToBuy):
            tradeType = 'buy'
            print(f'Buying - {row["Asks"]}')
            currentlyHolding = True
            tradeTimes.append([row['Timestamp'],'b'])
    else:
        if (row['crossOvers'] == -1):
            tradeType = 'sell'
            print(f'Selling - {row["Bids"]}')
            currentlyHolding = False
            tradeTimes.append([row['Timestamp'],'s'])
    return tradeType, currentlyHolding
 
def runTrades(df):
    currentlyHolding = False
    holdPrice = None
    pnl = 0
    minPearsonToBuy = -1
    numberToBuy = 100
    for index,row in df.iterrows():
        tradeType, currentlyHolding = trade(row, currentlyHolding, holdPrice,minPearsonToBuy)
        if tradeType == 'buy':
            pnl -= 100*row['Asks']
        elif tradeType == 'sell':
            pnl += 100*row['Bids']
    if currentlyHolding:
        pnl += 100*row['Bids']
        print(f'Selling - {row["Bids"]}')
        tradeTimes.append([row["Timestamp"],'s'])
    return pnl
 
 
 
def main():
    df = csvToDf()
    df = createNewColumns(df)
    #print(df.head())
    pnl = runTrades(df)    # EDIT THIS BIT
    print(f'PnL = ${pnl}')
    print(df['spread'].describe())
    graphData(df)
 
main()