from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque

def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    return drawdowns.max()

def compute_sortino_ratio(returns: np.ndarray, rf_rate=0.0):
    if len(returns) < 2:
        return 0.0
    neg_rets = returns[returns < 0]
    if len(neg_rets) == 0:
        return float('inf')
    dr = np.std(neg_rets, ddof=1)
    mean_excess = np.mean(returns) - rf_rate
    return mean_excess / dr if dr != 0 else float('inf') if mean_excess > 0 else 0.0

class TradingAlgorithm:
    def __init__(
        self,
        product="SOBER_expanded",
        window_size: int = 50,
        num_std_dev: float = 2.0,
        max_position: int = 100
    ):
        self.product = product
        self.window_size = window_size
        self.num_std_dev = num_std_dev
        self.max_position = max_position
        self.fees = 0.002

        self.positions = {}
        self.prices = []
        self.cash = 0.0
        self.pnl_history = []
        self.timestamp_history = []
        self.position_history = []
        self.mid_history = []
        self.signals = {self.product: []}
        self.trade_history = []
        self.active_trade = None

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        if self.product not in current_data:
            return order_data

        bid = current_data[self.product]['Bid']
        ask = current_data[self.product]['Ask']
        timestamp = current_data[self.product]['Timestamp']

        mid_price = (bid + ask) / 2
        self.prices.append(mid_price)

        if len(self.prices) < self.window_size:
            self._mark_to_market(mid_price, timestamp)
            return order_data

        rolling_mean = np.mean(self.prices[-self.window_size:])
        rolling_std = np.std(self.prices[-self.window_size:], ddof=1)

        upper_band = rolling_mean + self.num_std_dev * rolling_std
        lower_band = rolling_mean - self.num_std_dev * rolling_std

        position = self.positions.get(self.product, 0)
        qty = 0

        if mid_price < lower_band and position < self.max_position:
            qty = min(self.max_position - position, 10)
            self._record_trade(qty, ask, 'buy', timestamp)
        elif mid_price > upper_band and position > -self.max_position:
            qty = max(-self.max_position - position, -10)
            self._record_trade(qty, bid, 'sell', timestamp)
        elif abs(mid_price - rolling_mean) < 0.05 * mid_price and position != 0:
            qty = -position
            self._record_trade(qty, bid if qty < 0 else ask, 'close', timestamp)

        order_data[self.product] = qty
        self._mark_to_market(mid_price, timestamp)
        return order_data

    def _mark_to_market(self, mid_price, timestamp):
        position = self.positions.get(self.product, 0)
        self.pnl_history.append(self.cash + position * mid_price)
        self.timestamp_history.append(timestamp)
        self.position_history.append(position)
        self.mid_history.append(mid_price)

    def _record_trade(self, qty, price, label, timestamp):
        old_pos = self.positions.get(self.product, 0)
        new_pos = old_pos + qty

        if qty > 0:
            self.cash -= price * qty * (1 + self.fees)
        else:
            self.cash += abs(qty) * price * (1 - self.fees)

        self.positions[self.product] = new_pos
        self.signals[self.product].append((timestamp, label))

        if old_pos == 0 and new_pos != 0:
            self.active_trade = {'open_time': timestamp, 'open_price': price, 'size': new_pos}
        elif new_pos == 0 and old_pos != 0 and self.active_trade:
            pnl = (price - self.active_trade['open_price']) * self.active_trade['size']
            self.trade_history.append({
                'open_time': self.active_trade['open_time'],
                'close_time': timestamp,
                'pnl': pnl,
                'side': 'long' if self.active_trade['size'] > 0 else 'short'
            })
            self.active_trade = None

    def plotPerformance(self):
        # Calculate metrics
        final_pnl = self.pnl_history[-1] if self.pnl_history else 0.0
        stats_str = self._compute_metrics(final_pnl)
        print(stats_str)

        # (A) PnL + Mid
        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp_history, self.pnl_history, label='PnL')
        plt.plot(self.timestamp_history, self.mid_history, label='Mid Price', linestyle='--')
        plt.title("PnL & Mid Price")
        plt.legend()
        plt.show()

        # (B) Position + Mid
        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp_history, self.position_history, label='Position')
        plt.plot(self.timestamp_history, self.mid_history, label='Mid Price', linestyle='--')
        plt.title("Position Over Time")
        plt.legend()
        plt.show()

        # (C) Signals + Shading
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.timestamp_history, self.mid_history, label='Mid Price', linestyle='--')
        for trade in self.trade_history:
            color = 'green' if trade['pnl'] > 0 else 'red'
            ax.axvspan(trade['open_time'], trade['close_time'], color=color, alpha=0.2)
            x_mid = (trade['open_time'] + trade['close_time']) / 2

            # FIX: safely find the closest timestamp index
            closest_idx = np.abs(np.array(self.timestamp_history) - x_mid).argmin()
            y_mid = self.mid_history[closest_idx]

            ax.text(x_mid, y_mid, f"{trade['pnl']:+.2f}", ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.5))
        plt.title("Trades Overlaid on Mid Price")
        plt.legend()
        plt.show()


    def _compute_metrics(self, final_pnl):
        n_trades = len(self.trade_history)
        trade_pnls = [t['pnl'] for t in self.trade_history]
        win_rate = sum(1 for p in trade_pnls if p > 0) / n_trades * 100 if n_trades else 0.0
        max_dd = compute_max_drawdown(np.array(self.pnl_history))
        rets = np.diff(self.pnl_history)
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(50000) if len(rets) > 1 and np.std(rets) > 0 else 0.0
        sortino = compute_sortino_ratio(np.array(rets))
        return (
            f"Final PnL: {final_pnl:.2f}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Max Drawdown: {max_dd:.2f}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
            f"Sortino Ratio: {sortino:.2f}\n"
            f"Trades: {n_trades}"
        )

team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    return team_algorithm.getOrders(current_data, {p: 0 for p in current_data})

