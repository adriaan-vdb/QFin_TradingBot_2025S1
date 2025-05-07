from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
enter_th   = 1.4531
exit_th    = 0.3003
lookback   = 167.3007
"""

class TradingAlgorithm:

    def __init__(self):
        self.positions: Dict[str, int] = {} # self.positions is a dictionary that will keep track of your position in each product 
        ticker1 = "VP"
        ticker2 = "ORE"
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.previousResults = {ticker1:{'Timestamp':[], 'Bid':[], 'Ask':[]}, ticker2:{'Timestamp':[], 'Bid':[], 'Ask':[]}}
        # Below are some more optional data points if we want them
        self.mid = {ticker1:[], ticker2:[]}
        self.spread = {ticker1:[], ticker2:[]}
        self.new_order = {'':0}
        self.stmaList = {ticker1:[], ticker2:[]}
        self.ltmaList = {ticker1:[], ticker2:[]}
        self.stmaListN = {ticker1:[], ticker2:[]}
        self.ltmaListN = {ticker1:[], ticker2:[]}
        self.normList = {ticker1:[], ticker2:[]}
        self.mean = {ticker1:0, ticker2:0}
        self.std = {ticker1:0, ticker2:0}
        self.currentlyHolding = {ticker1:0, ticker2:0}
        self.trade = False
        self.normLength = 170

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        # To buy ABC for quantity x -> order_data["ABC"] = x (This will buy from the current best ask)
        # To sell ABC for quantity x -> order_data["ABC"] = -x (This will sell to the current best bid)
        self.addToStorage(current_data)   # adding these results to previous dictionary
        self.generateOtherData(current_data) # This funciton just makes some other potentially useful data for use to access
        self.new_order = {'':0}
        if self.tickerStratergy(): order_data.update(self.new_order)
        return order_data
    
    def tickerStratergy(self):
        self.trade = False
        # Only going forward if there is enough data
        if len(self.mid[self.ticker1]) > self.normLength and len(self.normList[self.ticker1]) > self.normLength:
            # Compute z-score of the spread between the normalized prices
            norm1 = self.normList[self.ticker1][-1]
            norm2 = self.normList[self.ticker2][-1]
            spread = norm1 - norm2
            spread_history = [a - b for a, b in zip(self.normList[self.ticker1][-self.normLength:], self.normList[self.ticker2][-self.normLength:])]
            mean_spread = np.mean(spread_history)
            std_spread = np.std(spread_history)

            if std_spread == 0: return False  # Prevent divd by zero

            zscore = (spread - mean_spread) / std_spread

            """
            enter_th   = 1.4531
            exit_th    = 0.3003
            """

            enter_threshold = 1.45  # best is 1.4 - 0.3 
            exit_threshold = 0.3  # 
            position_size = 100

            # Sell Logic
            if abs(zscore) < exit_threshold:
                if self.currentlyHolding[self.ticker1] > 0:
                    self.sell(self.ticker1, position_size)
                elif self.currentlyHolding[self.ticker1] < 0:
                    self.buy(self.ticker1, position_size)
                if self.currentlyHolding[self.ticker2] > 0:
                    self.sell(self.ticker2, position_size)
                elif self.currentlyHolding[self.ticker2] < 0:
                    self.buy(self.ticker2, position_size)

            # Buy Logic 
            elif zscore > enter_threshold:
                if self.currentlyHolding[self.ticker1] <= 0 and self.currentlyHolding[self.ticker2] >= 0:
                    self.sell(self.ticker1, position_size)
                    self.buy(self.ticker2, position_size)

            elif zscore < -enter_threshold:
                if self.currentlyHolding[self.ticker1] >= 0 and self.currentlyHolding[self.ticker2] <= 0:
                    self.buy(self.ticker1, position_size)
                    self.sell(self.ticker2, position_size)

        return self.trade


    def buy(self,ticker,amount=100):
        self.new_order.update({ticker:100})
        self.currentlyHolding[ticker] += 100
        self.trade = True

    def sell(self,ticker,amount=100):
        self.new_order.update({ticker:-100})
        self.currentlyHolding[ticker] -=100
        self.trade = True

    def addToStorage(self,current_data):   # This function adds all the previous data to a list
        for ticker in current_data:
            for column in self.previousResults[ticker]:
                self.previousResults[ticker][column].append(current_data[ticker][column])

    def generateOtherData(self,current_data):
        for i in current_data:
            self.spread[i].append(current_data[i]['Ask']-current_data[i]['Bid']) # gets the spread and adds it to dict
            self.mid[i].append((current_data[i]['Ask']+current_data[i]['Bid'])/2)  # gets the mid price and adds it to dict
            stmaLength = 25  # was 25
            ltmaLength = 200 # was 200
            if len(self.mid[i]) >= ltmaLength:
                self.stmaList[i].append(np.mean(self.mid[i][-stmaLength:]))
                self.ltmaList[i].append(np.mean(self.mid[i][-ltmaLength:]))
                self.stmaListN[i].append(np.mean(self.normList[i][-stmaLength:]))
                self.ltmaListN[i].append(np.mean(self.normList[i][-ltmaLength:]))
            if len(self.mid[i]) > self.normLength:
                self.normList[i].append(self.mid[i][-1]/self.mid[i][0])
                self.mean[i] = np.mean(self.normList[i][:-self.normLength])
                self.std[i] = np.std(self.normList[i][:-self.normLength])



# Leave this stuff as it is
team_algorithm = TradingAlgorithm()

def returnObject():
    return team_algorithm

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data)
