from typing import Dict
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
 
class TradingAlgorithm:
 
    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.ticker1 = "SOBER_expanded"
        self.ticker2 = "UEC_expanded"
       
        self.previousResults = {
            self.ticker1: {'Timestamp': [], 'Bid': [], 'Ask': []},
            self.ticker2: {'Timestamp': [], 'Bid': [], 'Ask': []}
        }
       
        self.mid = {self.ticker1: [], self.ticker2: []}
        self.spread = {self.ticker1: [], self.ticker2: []}
        self.positions = {self.ticker1: 0, self.ticker2: 0}
        self.jumps = {self.ticker1: [], self.ticker2: []}
        self.newOrder = {}
        self.buyData = {self.ticker1: {}, self.ticker2: {}}
        self.gradientInfo = {self.ticker1: {}, self.ticker2: {}}
       
        self.jumpDetected = False
        self.jumpPercent = 0.98
        self.purchasePercent = 1.5
        self.orderPrice = 0
        self.jumpCounter = 0
        self.holdCounter = 0
        self.currentlyHolding = False
        self.FirstRun = True
        self.orderVolume = 200
        self.gradient_window = deque(maxlen=100)

        self.pnl_over_time = []
        self.cash = 0  # Track cumulative cash directly here

 
    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        self.addToStorage(current_data)
        self.generateOtherData(current_data)
        if self.executeStrategy():
            order_data = self.newOrder
            self.newOrder = {}
            order_type = 'buy' if int(order_data[self.ticker1]) > 0 else 'sell'

            qty = int(order_data[self.ticker1])
            if qty > 0:
                self.cash -= current_data[self.ticker1]['Ask'] * qty * (1 + 0.002)  # Include fees
            elif qty < 0:
                self.cash += current_data[self.ticker1]['Bid'] * -qty * (1 - 0.002)

            self.buyData[self.ticker1][self.mid[self.ticker1][-1]] = [order_type, current_data[self.ticker1]['Timestamp']]
            #print(f'ORDER MADE: {order_data}')
        # Mark-to-market PnL calculation
        position = self.positions[self.ticker1]
        mid_price = self.mid[self.ticker1][-1]
        estimated_value = position * mid_price
        self.pnl_over_time.append(self.cash + estimated_value)

        return order_data
        
    def calculate_average_gradient(self):
        if len(self.gradient_window) < 100:
            return None  # Not enough points to calculate
       
        gradients = []
        x_100, y_100 = self.gradient_window[-1]  # 100th point (latest point)
       
        for i in range(99):
            x_i, y_i = self.gradient_window[i]
            gradient = (y_100 - y_i) / (x_100 - x_i) if x_100 != x_i else 0
            gradients.append(gradient)
       
        return np.mean(gradients)
 
    def add_point_and_get_gradient(self, x, y, ticker):
        self.gradient_window.append((x, y))
        avg_gradient = self.calculate_average_gradient()
        if avg_gradient is not None:
            self.gradientInfo[ticker][x] = avg_gradient
            #sprint(f'Avg Gradient = {avg_gradient}')
       
    # def executeStrategy(self):  
    #     """
    #     Stratergy:
    #     1. Sell 100 at the start
    #     2. Buy 200 if price is about to spike UP
    #     3. Sell 200 as soon as the price spikes UP
    #     4. Make sure to sell near end
    #     """
 
    #     if len(self.mid[self.ticker1]) < 100:
    #         if self.FirstRun:
    #             self.newOrder = {self.ticker1: -100}
    #             self.FirstRun = False
    #             return True
    #         return False  # Avoid phantom trades at the start
 
    #     jumpValue = self.jumps[self.ticker1][-1]
    #     if jumpValue != 0 and not self.jumpDetected:
    #         self.jumpDetected = True
    #         self.orderPrice = self.mid[self.ticker1][-1]
    #         self.jumpCounter = 0  # Start tracking time since jump
 
    #     # Buy condition: drop below threshold OR 20 time steps since jump
    #     if self.jumpDetected and not self.currentlyHolding:
    #         self.jumpCounter += 1
    #         if float(self.mid[self.ticker1][-1]) / float(self.orderPrice) <= self.jumpPercent or self.jumpCounter >= 20:
    #             self.newOrder = {self.ticker1: self.orderVolume}
    #             self.currentlyHolding = True
    #             self.orderPrice = self.previousResults[self.ticker1]['Ask'][-1]  # Buy at Ask price
    #             self.holdCounter = 0
    #             return True
       
    #     # Sell condition: price exceeds threshold OR 50 time steps since buy
    #     if self.currentlyHolding:
    #         self.holdCounter += 1
    #         if float(self.mid[self.ticker1][-1]) / self.orderPrice >= self.purchasePercent: #or self.holdCounter >= 50
    #             self.newOrder = {self.ticker1: -self.orderVolume}
    #             self.currentlyHolding = False
    #             self.jumpDetected = False
    #             self.orderPrice = self.previousResults[self.ticker1]['Bid'][-1]  # Sell at Bid price
    #             return True
    #     return False
    
    def executeStrategy(self):
        """
        Strategy:
        1. Sell 100 at the start
        2. Buy 200 if price drops after jump
        3. Sell 200 just after a steep upward spike (based on gradient)
        4. Make sure to sell near end
        """
        if len(self.mid[self.ticker1]) < 100:
            if self.FirstRun:
                self.newOrder = {self.ticker1: -100}
                self.FirstRun = False
                return True
            return False

        jumpValue = self.jumps[self.ticker1][-1]
        if jumpValue != 0 and not self.jumpDetected:
            self.jumpDetected = True
            self.orderPrice = self.mid[self.ticker1][-1]
            self.jumpCounter = 0

        # Buy condition: drop below threshold OR 20 time steps since jump
        if self.jumpDetected and not self.currentlyHolding:
            self.jumpCounter += 1
            if float(self.mid[self.ticker1][-1]) / float(self.orderPrice) <= self.jumpPercent or self.jumpCounter >= 20:
                self.newOrder = {self.ticker1: self.orderVolume}
                self.currentlyHolding = True
                self.orderPrice = self.previousResults[self.ticker1]['Ask'][-1]  # Buy at Ask
                self.holdCounter = 0
                return True

        # Sell condition: detect top of steep upward spike via gradient
        if self.currentlyHolding:
            self.holdCounter += 1

            current_timestamp = self.previousResults[self.ticker1]['Timestamp'][-1]
            gradient = self.gradientInfo[self.ticker1].get(current_timestamp, 0)

            if gradient > 0.4:  # Threshold can be tuned
                self.newOrder = {self.ticker1: -self.orderVolume}
                self.currentlyHolding = False
                self.jumpDetected = False
                self.orderPrice = self.previousResults[self.ticker1]['Bid'][-1]  # Sell at Bid
                return True

        return False

   
    def addToStorage(self, current_data):
        for ticker in current_data:
            for column in self.previousResults[ticker]:
                self.previousResults[ticker][column].append(current_data[ticker][column])
 
    def jumpCalculate(self, ticker):
        if len(self.mid[ticker]) < 100:
            return 0  # Not enough data
 
        differences = np.abs(np.diff(self.mid[ticker][-100:]))  
        avgDifference = np.mean(differences)
        stdDifference = np.std(differences)
        lastDifference = np.abs(self.mid[ticker][-1] - self.mid[ticker][-2])
        multiplier = 6
 
        if lastDifference > avgDifference + multiplier * stdDifference:
            direction = np.sign(self.mid[ticker][-1] - self.mid[ticker][-2])
            return 1 if direction > 0 else -1
 
        return 0
 
    def generateOtherData(self, current_data):
        for i in current_data:
            self.spread[i].append(current_data[i]['Ask'] - current_data[i]['Bid'])  # Spread calculation
            self.mid[i].append((current_data[i]['Ask'] + current_data[i]['Bid']) / 2)  # Mid price calculation
            jumpData = self.jumpCalculate(i)
            self.jumps[i].append(jumpData)
 
            if i == self.ticker1:
                x = current_data[i]['Timestamp']
                y = self.mid[i][-1]
                self.add_point_and_get_gradient(x, y, i)
 
team_algorithm = TradingAlgorithm()
 
def returnObject():
    return team_algorithm
 
def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data)


def plotPnL(self):
    plt.figure(figsize=(12, 6))
    plt.plot(self.pnl_over_time, label="PnL Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("PnL ($)")
    plt.title("PnL Evolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graphData(object):
    print('Starting Graphing')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(object.previousResults['SOBER_expanded']['Timestamp'], object.mid['SOBER_expanded'])
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.plot(object.previousResults['SOBER_expanded']['Timestamp'], object.jumps['SOBER_expanded'], color='red')

   
    sellList = [(object.buyData['SOBER_expanded'][i][1], i) for i in object.buyData['SOBER_expanded'] if object.buyData['SOBER_expanded'][i][0] == 'sell']
    buyList = [(object.buyData['SOBER_expanded'][i][1], i) for i in object.buyData['SOBER_expanded'] if object.buyData['SOBER_expanded'][i][0] == 'buy']
   
    for buy, sell in zip(buyList, sellList):
        buy_price = buy[1]
        sell_price = sell[1]
        profit_percent = ((sell_price - buy_price) / buy_price) * 100
        color = 'r' if profit_percent <= 2 else 'g'
        ax1.axvline(x=buy[0], color=color, lw=5, alpha=0.15)
        ax1.axvline(x=sell[0], color=color, lw=5, alpha=0.15)
   
    ax1.scatter([i[0] for i in sellList], [i[1] for i in sellList], color='r', marker='x')
    ax1.scatter([i[0] for i in buyList], [i[1] for i in buyList], color='g', marker='o')
    ax1.legend()
    plt.show()
    plotPnL(object)


print('Running...')