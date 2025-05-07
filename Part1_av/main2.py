from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ShortOnlyStrategy:
    def __init__(self, product="SOBER_expanded", max_position: int = 100):
        self.product = product
        self.max_position = max_position
        self.fees = 0.002

        self.positions = {}
        self.cash = 0.0

        self.pnl_history = []
        self.timestamp_history = []
        self.position_history = []
        self.mid_history = []

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        if self.product not in current_data:
            return order_data

        bid = current_data[self.product]['Bid']
        ask = current_data[self.product]['Ask']
        timestamp = current_data[self.product]['Timestamp']
        mid_price = (bid + ask) / 2

        position = self.positions.get(self.product, 0)
        qty = 0

        if position > -self.max_position:
            qty = -10  # Always short in chunks of 10 until max_position
            if position + qty < -self.max_position:
                qty = -self.max_position - position

            self.cash += abs(qty) * bid * (1 - self.fees)
            self.positions[self.product] = position + qty

        # Record for plotting
        self._mark_to_market(mid_price, timestamp)
        order_data[self.product] = qty
        return order_data

    def _mark_to_market(self, mid_price, timestamp):
        position = self.positions.get(self.product, 0)
        equity = self.cash + position * mid_price
        self.pnl_history.append(equity)
        self.timestamp_history.append(timestamp)
        self.position_history.append(position)
        self.mid_history.append(mid_price)

    def plotPerformance(self):
        final_pnl = self.pnl_history[-1] if self.pnl_history else 0.0
        print(f"Final PnL: {final_pnl:.2f}")

        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp_history, self.pnl_history, label='PnL')
        plt.plot(self.timestamp_history, self.mid_history, label='Mid Price', linestyle='--')
        plt.title("PnL & Mid Price (Short-Only Strategy)")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp_history, self.position_history, label='Position')
        plt.plot(self.timestamp_history, self.mid_history, label='Mid Price', linestyle='--')
        plt.title("Position Over Time")
        plt.legend()
        plt.show()

short_only_algo = ShortOnlyStrategy()

def getOrders(current_data, positions):
    short_only_algo.positions = positions
    return short_only_algo.getOrders(current_data, {p: 0 for p in current_data})

team_algorithm = short_only_algo