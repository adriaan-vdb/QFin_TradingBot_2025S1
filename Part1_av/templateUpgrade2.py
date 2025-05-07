from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

class TradingAlgorithm:
    """
    Strategy Logic (UEC only):
        - We detect 'high spread' events. 
        - If we're IN_POSITION when that happens, we flatten immediately.
        - Then we transition to WAITING_DIRECTION, wait a small number of ticks (direction_wait_ticks),
          and decide to go long/short up to max_position if mid price has gone up/down
          compared to mid_at_high_spread.
        - We remain in that position until the next high spread event.
    """

    def __init__(
        self,
        high_spread_threshold: float = 1.35,
        direction_wait_ticks: int = 50,
        max_position: int = 100,
        stma_window: int = 70
    ):
        """
        :param high_spread_threshold: Spread above this triggers a special event.
        :param direction_wait_ticks: How many ticks to wait after high spread to confirm direction.
        :param max_position: Maximum position to enter (buy or sell).
        :param stma_window: Rolling average window if you want more advanced checks (unused here).
        """
        self.product = "UEC_expanded"
        self.high_spread_threshold = high_spread_threshold
        self.direction_wait_ticks = direction_wait_ticks
        self.max_position = max_position
        self.stma_window = stma_window

        # Positions. 
        # The backtester passes these in each tick, but we store them here for internal logic & plotting.
        self.positions: Dict[str, int] = {}

        # We'll keep historical data ONLY for UEC
        self.previousResults = {
            self.product: {'Timestamp': [], 'Bid': [], 'Ask': []}
        }

        # Store time-series for mid price & spread (for plotting, logic if needed)
        self.mid = {self.product: []}
        self.spread = {self.product: []}

        # --- STRATEGY STATE MACHINE ---
        # We can be FLAT, WAITING_DIRECTION, or IN_POSITION
        self.state = "FLAT"
        # We'll store the tick/time of the last high-spread event & the mid price then
        self.last_high_spread_tick = None
        self.mid_at_high_spread = None

        # We'll track "current_tick" from the backtester
        self.current_tick = 0

        # --- For approximate PnL tracking & plotting ---
        # We'll store "cash" for real-time PnL as if we are marking to market each tick.
        self.cash = 0.0
        self.fees = 0.002  # same as backtest

        # We'll build arrays for plotting
        self.pnl_history = []         # approximate total PnL each tick
        self.timestamp_history = []   # store the i (timestamp) each tick
        self.position_history = []    # store net position each tick
        self.mid_history = []         # store mid price each tick

        # IMPORTANT: This is now a DICTIONARY keyed by self.product
        self.order_signals = {self.product: []}  # (tick, 'buy'/'sell'/'close')

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        """
        The backtester calls this each tick. 
        current_data might contain UEC, SOBER, etc. We only look at UEC.
        order_data is a dict of {product: 0}, which we fill with buy/sell quantities.
        """
        # If the product we care about is not in current_data, do nothing
        if self.product not in current_data:
            return order_data

        # Grab the current UEC data
        bid = current_data[self.product]['Bid']
        ask = current_data[self.product]['Ask']
        self.current_tick = current_data[self.product]['Timestamp']

        # Add to internal storage
        self.addToStorage(bid, ask)
        self.generateOtherData(bid, ask)

        # Default order is 0 (no trade)
        order_data[self.product] = 0

        # Strategy logic
        spread_now = ask - bid
        mid_now = (ask + bid) / 2

        # (A) If we see a high spread, flatten if in position, then WAITING_DIRECTION
        if spread_now > self.high_spread_threshold:
            if self.state == "IN_POSITION":
                # Flatten
                qty_to_close = -self.positions.get(self.product, 0)
                if qty_to_close != 0:
                    order_data[self.product] = qty_to_close
                    # We'll record a "close" signal (though from PnL perspective, buy vs. sell depends on sign)
                    # If it's negative, it's a "sell" from backtest's perspective, but let's call it "close" for clarity.
                    self._recordTrade(qty_to_close, (ask if qty_to_close > 0 else bid), 'close')

            # Switch to WAITING_DIRECTION
            self.state = "WAITING_DIRECTION"
            self.last_high_spread_tick = self.current_tick
            self.mid_at_high_spread = mid_now

        # (B) If we're WAITING_DIRECTION, check if enough ticks have passed since high spread
        if self.state == "WAITING_DIRECTION" and (self.last_high_spread_tick is not None):
            ticks_elapsed = self.current_tick - self.last_high_spread_tick
            if ticks_elapsed >= self.direction_wait_ticks:
                # Compare current mid vs. mid_at_high_spread
                if mid_now > self.mid_at_high_spread:
                    # Buy up to max_position
                    target_pos = self.max_position
                    if self.positions.get(self.product, 0) < target_pos:
                        qty = target_pos - self.positions[self.product]
                        order_data[self.product] = qty
                        self._recordTrade(qty, ask, 'buy')
                else:
                    # Sell down to -max_position
                    target_pos = -self.max_position
                    if self.positions.get(self.product, 0) > target_pos:
                        qty = target_pos - self.positions[self.product]
                        self._recordTrade(qty, bid, 'sell')
                        order_data[self.product] = qty

                self.state = "IN_POSITION"

        # (C) If IN_POSITION and not high spread => do nothing until next high spread
        return order_data

    def addToStorage(self, bid: float, ask: float):
        """
        For each tick, store the timestamp, bid, ask, and do approximate mark-to-market 
        to track PnL for plotting.
        """
        self.previousResults[self.product]['Timestamp'].append(self.current_tick)
        self.previousResults[self.product]['Bid'].append(bid)
        self.previousResults[self.product]['Ask'].append(ask)

        # Mark to market
        mid_p = (bid + ask) / 2
        current_pos = self.positions.get(self.product, 0)
        total_equity = (current_pos * mid_p) + self.cash

        self.pnl_history.append(total_equity)
        self.timestamp_history.append(self.current_tick)
        self.position_history.append(current_pos)

    def generateOtherData(self, bid: float, ask: float):
        """Keep track of mid price & spread for logic or plotting."""
        mid_price = (bid + ask) / 2
        sprd = ask - bid
        self.mid[self.product].append(mid_price)
        self.spread[self.product].append(sprd)
        self.mid_history.append(mid_price)

    def _recordTrade(self, quantity: int, trade_price: float, signal_label: str):
        """
        Update our approximate 'cash' for real-time PnL and record the trade signal.
        :param quantity: + means buy, - means sell
        :param trade_price: the price we "transact" at
        :param signal_label: 'buy', 'sell', or 'close' for plotting
        """
        # 1) Record the signal so we can plot it
        self.order_signals[self.product].append((self.current_tick, signal_label))

        # 2) Update approximate cash
        if quantity > 0:
            # buy => reduce cash
            cost = trade_price * quantity * (1 + self.fees)
            self.cash -= cost
        else:
            # sell => increase cash
            proceeds = trade_price * abs(quantity) * (1 - self.fees)
            self.cash += proceeds

    # -------------------------------------------------
    #             PERFORMANCE PLOTTING
    # -------------------------------------------------
    def plotPerformance(self):
        """
        Generates multiple performance plots:
          1) PnL (equity) over time
          2) Drawdown
          3) Position over time
          4) Combined plot showing Spread (primary axis) and Mid Price (secondary axis)
             with Buy/Sell/Close signals overlaid.
        """
        import pandas as pd

        # 1) Build a DataFrame with approximate PnL over time
        df_perf = pd.DataFrame({
            'Timestamp': self.timestamp_history,
            'PnL': self.pnl_history
        }).set_index('Timestamp')

        # ---------------------------
        # (A) PnL Over Time
        # ---------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(df_perf.index, df_perf['PnL'], label='Equity Curve')
        plt.title('PnL / Equity Curve Over Time (UEC)')
        plt.xlabel('Tick')
        plt.ylabel('PnL')
        plt.legend()
        plt.show()

        # ---------------------------
        # (B) Drawdown
        # ---------------------------
        df_perf['cummax'] = df_perf['PnL'].cummax()
        df_perf['drawdown'] = df_perf['PnL'] - df_perf['cummax']

        plt.figure(figsize=(10, 4))
        plt.plot(df_perf.index, df_perf['drawdown'], label='Drawdown', color='red')
        plt.title('Drawdown Over Time (UEC)')
        plt.xlabel('Tick')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.show()

        # ---------------------------
        # (C) Position Over Time
        # ---------------------------
        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp_history, self.position_history, label='UEC Position')
        plt.title('UEC Position Over Time')
        plt.xlabel('Tick')
        plt.ylabel('Position')
        plt.legend()
        plt.show()

        # ---------------------------------------------------------
        # (D) Spread & Mid Price + Buy/Sell/Close signals (two y-axes)
        # ---------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # 1) Plot Spread on the primary y-axis
        ax1.plot(self.timestamp_history, self.spread[self.product], label='Spread')
        ax1.set_xlabel('Tick')
        ax1.set_ylabel('Spread')
        ax1.tick_params(axis='y')

        # 2) Plot Mid Price on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(self.timestamp_history, self.mid_history, label='Mid Price', linestyle='--')
        ax2.set_ylabel('Mid Price')
        ax2.tick_params(axis='y')

        # 3) Scatter trade signals on the mid-price axis
        buy_ticks = [ts for (ts, sig) in self.order_signals[self.product] if sig == 'buy']
        sell_ticks = [ts for (ts, sig) in self.order_signals[self.product] if sig == 'sell']
        close_ticks = [ts for (ts, sig) in self.order_signals[self.product] if sig == 'close']

        # Convert those ticks to mid-price for plotting
        buy_prices = [
            self.mid_history[i] for i, t in enumerate(self.timestamp_history)
            if t in buy_ticks
        ]
        sell_prices = [
            self.mid_history[i] for i, t in enumerate(self.timestamp_history)
            if t in sell_ticks
        ]
        close_prices = [
            self.mid_history[i] for i, t in enumerate(self.timestamp_history)
            if t in close_ticks
        ]

        ax2.scatter(buy_ticks, buy_prices,   marker='^', label='Buy',   s=100)
        ax2.scatter(sell_ticks, sell_prices, marker='v', label='Sell',  s=100)
        ax2.scatter(close_ticks, close_prices, marker='o', label='Close', s=100)

        # 4) Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

        plt.title('Spread & Mid Price w/ Trade Signals (UEC)')
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------
# Instantiate for the backtester
team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    """
    Called by the backtest engine every tick.
    :param current_data: dictionary w/ 'UEC' (and maybe other products).
    :param positions: dict of current positions (e.g. {'UEC': 0}, etc.)
    :return: a dict of orders, e.g. {'UEC': 10}
    """
    # Sync positions with local
    team_algorithm.positions = positions

    # The backtester gives {product:0} for each product in current_data.
    # We'll only modify 'UEC' based on logic in getOrders().
    order_data = {product: 0 for product in current_data}

    return team_algorithm.getOrders(current_data, order_data)
