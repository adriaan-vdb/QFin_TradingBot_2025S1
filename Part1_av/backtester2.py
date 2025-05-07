import pandas as pd
import main_uec as template_uec   # Strategy for UEC_expanded
import ethan as template_sober  # Strategy for SOBER_expanded
from copy import deepcopy

DATA_LOCATION = "./HistData"  # Set this to the location of your data folder

# Dictionary mapping products to their corresponding strategy modules
product_strategy_map = {
    "UEC_expanded": template_uec,
    "SOBER_expanded": template_sober
}

products = list(product_strategy_map.keys())

# Initialize data containers
price_series = {}
positions = {}
cash = {}

# Load data and initialize positions/cash
for product in products:
    price_series[product] = pd.read_csv(f"{DATA_LOCATION}/{product}.csv")
    positions[product] = 0
    cash[product] = 0

position_limit = 100
fees = 0.002
n_timestamps = len(price_series[products[0]])

# Backtest loop
for i in range(n_timestamps):
    current_data = {}

    for product in products:
        current_data[product] = {
            "Timestamp": i,
            "Bid": price_series[product].iloc[i]["Bids"],
            "Ask": price_series[product].iloc[i]["Asks"]
        }

    # Call each strategy individually
    for product in products:
        strategy = product_strategy_map[product]
        single_product_data = {product: deepcopy(current_data[product])}
        single_product_position = {product: deepcopy(positions[product])}

        order = strategy.getOrders(single_product_data, single_product_position)

        if product in order:
            quant = int(order[product])

            if quant == 0:
                continue

            if quant > 0:  # Buy
                if positions[product] + quant > position_limit:
                    quant = 0
                cash[product] -= current_data[product]["Ask"] * quant * (1 + fees)

            elif quant < 0:  # Sell
                if positions[product] + quant < -position_limit:
                    quant = 0
                cash[product] += current_data[product]["Bid"] * -quant * (1 - fees)

            positions[product] += quant

# Final position closure and PnL calculation
cash_sum = 0
for product in products:
    print(f"{product} unclosed: PnL = {cash[product]}, Position = {positions[product]}")

    if positions[product] > 0:
        cash[product] += price_series[product].iloc[-1]["Bids"] * positions[product] * (1 - fees)
    elif positions[product] < 0:
        cash[product] -= price_series[product].iloc[-1]["Asks"] * -positions[product] * (1 + fees)

    cash_sum += cash[product]
    print(f"{product} closed: PnL = {cash[product]}")

print(f"Total PnL = {cash_sum}")

# Plot individual strategy performances if available
if hasattr(template_uec.team_algorithm, "plotPerformance"):
    print("UEC Strategy Performance:")
    template_uec.team_algorithm.plotPerformance()

print("SOBER Strategy Performance:")
template_sober.graphData(template_sober.returnObject())

