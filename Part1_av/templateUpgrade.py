from typing import Dict


class TradingAlgorithm:

    def __init__(self):
        self.positions: Dict[str, int] = {} # self.positions is a dictionary that will keep track of your position in each product 
        ticker1 = "SOBER"
        ticker2 = "UEC"
        self.previousResults = {ticker1:{'Timestamp':[], 'Bid':[], 'Ask':[]}, ticker2:{'Timestamp':[], 'Bid':[], 'Ask':[]}}
        # Below are some more optional data points if we want them
        self.mid = {ticker1:[], ticker2:[]}
        self.spread = {ticker1:[], ticker2:[]}

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        # print(current_data)
        # To buy ABC for quantity x -> order_data["ABC"] = x (This will buy from the current best ask)
        # To sell ABC for quantity x -> order_data["ABC"] = -x (This will sell to the current best bid)
        self.addToStorage(current_data)   # adding these results to previous dictionary
        self.generateOtherData(current_data) # This funciton just makes some other potentially useful data for use to access
        return order_data
    
    def addToStorage(self,current_data):   # This function adds all the previous data to a list
        for ticker in current_data:
            for column in self.previousResults[ticker]:
                self.previousResults[ticker][column].append(current_data[ticker][column])

    def generateOtherData(self,current_data):
        for i in current_data:
            self.spread[i].append(current_data[i]['Ask']-current_data[i]['Bid']) # gets the spread and adds it to dict
            self.mid[i].append((current_data[i]['Ask']+current_data[i]['Bid'])/2)  # gets the mid price and adds it to dict


# Leave this stuff as it is
team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data)


