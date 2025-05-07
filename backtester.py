import pandas as pd
import template as template
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA_LOCATION = "Set3/DataFolder"  # Set this to the location of your data folder

# List of stock names (without .csv)
products = ["VP", "ORE"]

# Dictionary to store the price series dataframes for each stock
price_series = {product: pd.read_csv(f"{DATA_LOCATION}/{product}.csv") for product in products}

# Initialize positions and cash
positions = {product: 0 for product in products}
cash = {product: 0 for product in products}

# Lists for tracking
cumulative_pnl = []
trade_events = []

# Constants
position_limit = 100
fees = 0.002
n_timestamps = len(price_series[products[0]])
totalOrders = 0

# Backtest loop
for i in range(n_timestamps):
    current_data = {
        product: {
            "Timestamp": i,
            "Bid": price_series[product].iloc[i]["Bids"],
            "Ask": price_series[product].iloc[i]["Asks"]
        } for product in products
    }

    order = template.getOrders(deepcopy(current_data), deepcopy(positions))

    for product in order:
        quant = int(order[product])
        if quant == 0:
            continue

        # Buy
        if quant > 0:
            if positions[product] + quant > position_limit:
                quant = 0
            else:
                cash[product] -= current_data[product]["Ask"] * quant * (1 + fees)
                print(f'Selling: -{current_data[product]["Ask"] * quant * (1 + fees)}')
                trade_events.append(i)
                totalOrders += 1

        # Sell
        elif quant < 0:
            if positions[product] + quant < -position_limit:
                quant = 0
            else:
                cash[product] += current_data[product]["Bid"] * -quant * (1 - fees)
                print(f'Buying: {current_data[product]["Bid"] * -quant * (1 - fees)}')
                trade_events.append(i)
                totalOrders += 1

        positions[product] += quant

    # Cumulative PnL tracking (cash + mark-to-market value of current holdings)
    mtm_value = sum(
        positions[product] * (
            current_data[product]["Bid"] if positions[product] < 0 else current_data[product]["Ask"]
        ) for product in products
    )
    total_pnl = sum(cash[product] for product in products) + mtm_value
    cumulative_pnl.append(total_pnl)


# Close open positions
for product in products:
    final_bid = price_series[product].iloc[-1]["Bids"]
    final_ask = price_series[product].iloc[-1]["Asks"]

    if positions[product] > 0:
        cash[product] += final_bid * positions[product] * (1 - fees)
    elif positions[product] < 0:
        cash[product] -= final_ask * -positions[product] * (1 + fees)

# Final PnL
final_cash = sum(cash[product] for product in products)
print(f"Total Orders = {totalOrders}")
print(f"Total PnL = {final_cash:.2f}")

# --- Graph 1: PnL Curve + T-stat ---

returns = np.diff(cumulative_pnl)
t_stat, p_val = stats.ttest_1samp(returns, 0)
print(f"T-statistic of strategy returns: {t_stat:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(cumulative_pnl, label="Cumulative PnL", linewidth=2)
plt.title("Cumulative PnL Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Cumulative PnL ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)

object = template.returnObject()

# === Third Graph: Mid Prices + LTMA on Same Scale with Trade Lines ===
plt.figure(figsize=(10, 6))

# Plot mid prices and LTMA for both tickers
for product in products:
    mid_price = object.normList[product]
    ltma = object.ltmaListN[product]
    
    plt.plot(mid_price, label=f'{product} Mid Price')
    plt.plot(ltma, linestyle='--', label=f'{product} LTMA')

# Plot vertical lines at every trade (buy or sell)
for event_time in trade_events:
    plt.axvline(x=event_time, color='yellow', linestyle='--', alpha=0.6)

plt.title('Mid Prices and LTMA of Both Tickers (Same Scale) with Trade Lines')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
