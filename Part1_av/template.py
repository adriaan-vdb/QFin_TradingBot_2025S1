from typing import Dict
import Research3

class TradingAlgorithm:

    def __init__(self):
        self.positions: Dict[str, int] = {} # self.positions is a dictionary that will keep track of your position in each product 
                                            # E.g. {"ABC": 2, "XYZ": -5, ...}
                                            # This will get automatically updated after each call to getOrders()

        #
        # TODO: Initialise any other variables that you want to use to keep track of things between getOrders() calls here
        #


    # This method will be called every timestamp with information about the new best bid and best ask for each product
    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        # current_data is a dictionary that holds the current timestamp, best bid and best ask for each product
        # E.g. {"ABC": {"Timestamp": 134, "Bid": 34, "Ask" 38}, "XYZ": {"Timestamp": 134, "Bid": 1034, "Ask": 1038}, ...}
        
        # order_data is a dictionary that holds the quantity you will order for each product in this current timestamp
        # Intially the quantity for each product is set to 0 (i.e. no buy or sell orders will be sent if order_data is returned as it is)
        # To buy ABC for quantity x -> order_data["ABC"] = x (This will buy from the current best ask)
        # To sell ABC for quantity x -> order_data["ABC"] = -x (This will sell to the current best bid)
        
        #
        # TODO: Process new data and populate order_data here

        # Example of a simple strategy:
            # When the short term moving average is greater than the long term moving average, sell
            # When the short term moving average is less than the long term moving average, buy


        return order_data
    
    # You may create more methods to help process data but ensure they are inside this class
    


# Leave this stuff as it is
team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data)
