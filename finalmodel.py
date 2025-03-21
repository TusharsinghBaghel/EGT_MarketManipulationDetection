import pandas as pd
import numpy as np
import matplotlib.dates as mdates


def assign_strategy(row, marketindex):
    strat_string = ""
    if row["ModalPrice"] < priceThreshold[marketindex] and row["ArrivalQuintals"] < arrivalThreshold[marketindex]:
        strat_string = "lPrice_lArrival"
    elif row["ModalPrice"] < priceThreshold[marketindex] and row["ArrivalQuintals"] >= arrivalThreshold[marketindex]:
        strat_string = "lPrice_hArrival"
    elif row["ModalPrice"] >= priceThreshold[marketindex] and row["ArrivalQuintals"] < arrivalThreshold[marketindex]:
        strat_string = "hPrice_lArrival"
    elif row["ModalPrice"] >= priceThreshold[marketindex] and row["ArrivalQuintals"] >= arrivalThreshold[marketindex]:
        strat_string = "hPrice_hArrival"
    return strat_string
def seasonal_factor(month): #more seasonal factor means less production cost and more demand
    # Use a periodic function to approximate the seasonal factor
    # For example, using a sine function to simulate seasonal variation
    # The function is adjusted to have a period of 12 months and values between 0.85 and 1.15
    return 1 +  np.sin(( 4*np.pi * (month - 1)) / 12)

def production_cost(row):
    #this is per quintal cost
    # approximate cost of production per quintal, + inflation per year and consider the seasonal nature of cost of production
    base_cost = 1000  # base cost of production per quintal
    inflation_rate = 0.05  # 5% inflation per year
    
    year_diff = row['Reported Date'].year - 2000  # assuming base year is 2000
    inflation_adjusted_cost = base_cost * ((1 + inflation_rate) ** year_diff)
    month = row['Reported Date'].month
    seasonal_adjusted_cost = inflation_adjusted_cost / seasonal_factor(month)
    
    return seasonal_adjusted_cost

def demand_factor(row):
    # demand depends on the total arrival of all markets in that interval
    # seasonal factors are used to adjust the demand
    two_months_ago = row['Reported Date'] - pd.DateOffset(months=2)
    total_arrival = df[(df['Reported Date'] >= two_months_ago) & (df['Reported Date'] < row['Reported Date'])]['ArrivalQuintals'].sum()
    month = row['Reported Date'].month
    demand = seasonal_factor(month) / (total_arrival + 1)  ##in for scrutiny  
    return demand

def utility_function(row):
    cost = production_cost(row)
    demand = demand_factor(row)
    arrival = row['ArrivalQuintals']
    price = row['ModalPrice']
    return (price - cost) * arrival * demand

###############################################
# Data Preprocessing and renaming columns
###############################################
df = pd.read_csv("agmarknet2.csv")
df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d-%b-%y')
markets = df['Market Name'].unique().tolist()
for col in ["Arrivals (Tonnes)", "Modal Price (Rs./Quintal)"]:
    df.loc[:, col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

df["Arrivals Quintals"] = df["Arrivals (Tonnes)"] * 10
#print all columns
#print(df.columns)
#remove arrival in tonnes
df.drop(columns=["Arrivals (Tonnes)"], inplace=True)
df['year'] = df['Reported Date'].dt.year
#rename columns
df.rename(columns={"Arrivals Quintals": "ArrivalQuintals"}, inplace=True)
df.rename(columns={"Modal Price (Rs./Quintal)": "ModalPrice"}, inplace=True)
df.rename(columns={"Min Price (Rs./Quintal)": "MinPrice"}, inplace=True)
df.rename(columns={"Max Price (Rs./Quintal)": "MaxPrice"}, inplace=True)
df.rename(columns={"Commodity": "Commodity"}, inplace=True)
df.rename(columns={"State Name": "State"}, inplace=True)
df.rename(columns={"District Name": "District"}, inplace=True)
df.rename(columns={"Market Name": "Market"}, inplace=True)
#print(df.head)

df_market = []
for market in markets:
    df_market.append(df[df["Market"] == market])

################################################################################
# Add exterior columns like strategy, production cost, demand factor, and profit
################################################################################

priceThreshold = []
arrivalThreshold = []

for i in range(0, len(df_market)):
    priceThreshold.append(np.percentile(df_market[i]["ModalPrice"], 50))
    arrivalThreshold.append(np.percentile(df_market[i]["ArrivalQuintals"], 25))

for i in range(0, len(df_market)):
    df_market[i]['Strategy'] = df_market[i].apply(assign_strategy, axis=1, marketindex=i)
    df_market[i]['ProductionCost'] = df_market[i].apply(production_cost, axis=1)
    df_market[i]['DemandFactor'] = df_market[i].apply(demand_factor, axis=1)    
    df_market[i]['profit'] = df_market[i].apply(utility_function, axis=1)


#print count of different strategies for all markets loop
for i in range(0, len(df_market)):
    print(df_market[i]["Strategy"].value_counts())


# Plot arrival rate, seasonal factor, demand factor, production cost, and modal price for 1 year
import matplotlib.pyplot as plt

# Prompt user to select a market index
print("Select a market index from the following list:")
for idx, market in enumerate(markets):
    print(f"{idx}: {market}")

market_index = int(input("Enter the market index: "))

# Ensure the selected index is valid
if market_index < 0 or market_index >= len(markets):
    raise ValueError("Invalid market index selected.")

df_market_sorted = df_market[market_index].sort_values(by='Reported Date')
year = int(input("Enter the year for which you want to plot the data: "))
df_market_year = df_market_sorted[df_market_sorted['Reported Date'].dt.year == year]

# Group by week and sum the arrival quintals and average the modal price
df_market_year['Week'] = df_market_year['Reported Date'].dt.to_period('W').apply(lambda r: r.start_time)
df_market_weekly = df_market_year.groupby('Week').agg({
    'ArrivalQuintals': 'sum',
    'Reported Date': 'first',
    'DemandFactor': 'first',
    'ProductionCost': 'first',
    'ModalPrice': 'mean'
}).reset_index()

fig, ax1 = plt.subplots()

ax1.set_xlabel('Date')
ax1.set_ylabel('Arrival Quintals', color='tab:blue')
ax1.plot(df_market_weekly['Week'], df_market_weekly['ArrivalQuintals'], label='Arrival Quintals', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Demand Factor', color='tab:orange')  # we already handled the x-label with ax1
ax2.plot(df_market_year['Reported Date'], df_market_year['DemandFactor'], label='Demand Factor', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Plot seasonal factor on a secondary y-axis
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Seasonal Factor', color='tab:green')
ax3.plot(df_market_year['Reported Date'], df_market_year['Reported Date'].dt.month.apply(seasonal_factor), label='Seasonal Factor', color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')

# Plot production cost on a secondary y-axis
ax4 = ax1.twinx()
ax4.spines['right'].set_position(('outward', 120))
ax4.set_ylabel('Production Cost', color='tab:red')
ax4.plot(df_market_year['Reported Date'], df_market_year['ProductionCost'], label='Production Cost', color='tab:red')
ax4.tick_params(axis='y', labelcolor='tab:red')

# Plot modal price on a secondary y-axis
ax5 = ax1.twinx()
ax5.spines['right'].set_position(('outward', 180))
ax5.set_ylabel('Modal Price', color='tab:purple')
ax5.plot(df_market_weekly['Week'], df_market_weekly['ModalPrice'], label='Modal Price', color='tab:purple')
ax5.tick_params(axis='y', labelcolor='tab:purple')

#plot the utility function
ax6 = ax1.twinx()
ax6.spines['right'].set_position(('outward', 240))  
ax6.set_ylabel('Utility Function', color='tab:gray')
ax6.plot(df_market_year['Reported Date'], df_market_year['profit'], label='Utility Function', color='tab:gray')
ax6.tick_params(axis='y', labelcolor='tab:gray')


fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()


####################################################################
#Group markets weakly and calculate frequencies plot of strategies
####################################################################

# Group each market's data by weekly, create 4 new columns giving the frequency of each strategy that week, modal price is mean, and arrival quintals is sum, and the demand factor is mean, and the production cost is mean, and the utility function is sum
weekly_summaries = []

for i in range(0, len(df_market)):
    df_market[i]['Week'] = df_market[i]['Reported Date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_summary = df_market[i].groupby('Week').agg({
        'ArrivalQuintals': 'sum',
        'ModalPrice': 'mean',
        'DemandFactor': 'mean',
        'ProductionCost': 'mean',
        'profit': 'sum',
        'Strategy': lambda x: x.value_counts(normalize=True).to_dict()
    }).reset_index()

    # Create separate columns for each strategy frequency
    strategy_columns = ['lPrice_lArrival', 'lPrice_hArrival', 'hPrice_lArrival', 'hPrice_hArrival']
    for col in strategy_columns:
        weekly_summary[col] = weekly_summary['Strategy'].apply(lambda x: x.get(col, 0))

    # Drop the original 'Strategy' column as it's no longer needed
    weekly_summary.drop(columns=['Strategy'], inplace=True)

    weekly_summaries.append(weekly_summary)

# Print the weekly summaries for each market
for i, summary in enumerate(weekly_summaries):
    print(f"Weekly summary for market {markets[i]}:")
    print(summary.head())
    #print no of rows
    print(summary.shape)

# plot the strategies of markets in separate subplots
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

strategy_columns = ['lPrice_lArrival', 'lPrice_hArrival', 'hPrice_lArrival', 'hPrice_hArrival']
titles = ['Low Price Low Arrival', 'Low Price High Arrival', 'High Price Low Arrival', 'High Price High Arrival']

for idx, strategy in enumerate(strategy_columns):
    for i, summary in enumerate(weekly_summaries):
        axs[idx].plot(summary['Week'], summary[strategy], label=f'{markets[i]}')
    axs[idx].set_ylabel('Frequency')
    axs[idx].set_title(titles[idx])
    axs[idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

axs[-1].set_xlabel('Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ###############################################
# # 6. Run Replicator Dynamics (Weekly, Lower Gamma)
# ###############################################

def dynamic_gamma(prev_profits, smoothing_factor=0.1, base_gamma=0.0001, max_gamma=0.01, min_gamma=0.00001):
    """
    Computes a dynamic gamma (learning rate) based on profit fluctuations.

    Parameters:
    - prev_profits (list): A list of previous weekly profits.
    - smoothing_factor (float): Weight for exponential moving average (EMA).
    - base_gamma (float): Minimum baseline gamma.
    - max_gamma (float): Maximum allowable gamma.
    - min_gamma (float): Minimum gamma to prevent too slow learning.

    Returns:
    - float: Adjusted gamma value.
    """
    if len(prev_profits) < 2:
        return base_gamma  # Default gamma for initial weeks

    # Compute weekly profit differences
    profit_changes = np.abs(np.diff(prev_profits))

    # Exponential Moving Average (EMA) of profit changes (smoothens noise)
    volatility = smoothing_factor * profit_changes[-1] + (1 - smoothing_factor) * np.mean(profit_changes)

    # Normalize volatility to range [0,1] using min-max scaling
    max_volatility = np.max(profit_changes) if len(profit_changes) > 0 else 1
    norm_volatility = min(1, volatility / (max_volatility + 1e-6))  # Prevent divide by zero

    # Scale gamma dynamically between min_gamma and max_gamma
    gamma = min_gamma + (max_gamma - min_gamma) * norm_volatility

    return gamma

def replicator_dynamics(frequencies, payoffs, gamma=1):
    """
    Perform one step of replicator dynamics.
    
    Parameters:
    frequencies (np.array): Current strategy frequencies.
    payoffs (np.array): Payoff for each strategy.
    gamma (float): Learning rate.
    
    Returns:
    np.array: Updated strategy frequencies.
    """
    avg_payoff = np.dot(frequencies, payoffs)
    new_frequencies = frequencies * (1 + gamma * (payoffs - avg_payoff))
    return new_frequencies / new_frequencies.sum()

def predict_strategies(start_date, end_date, gamma=0.1):
    """
    Predict strategies using replicator dynamics from start_date to end_date.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    gamma (float): Learning rate.
    
    Returns:
    dict: Predicted strategy frequencies for each market.
    """
    predictions = {market: [] for market in markets}
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    
    for i, summary in enumerate(weekly_summaries):
        strategies = ['lPrice_lArrival', 'lPrice_hArrival', 'hPrice_lArrival', 'hPrice_hArrival']
        frequencies = summary[strategies].values[-1]  # Start with the last known frequencies
        payoffs = summary['profit'].values
        past_profits = summary['profit'].values[-5:].tolist()
        for date in date_range:
            gamma = dynamic_gamma(past_profits)
            frequencies = replicator_dynamics(frequencies, payoffs[-1], gamma)  # Use the last known payoff
            predictions[markets[i]].append((date, frequencies.copy()))
    
    return predictions

# Example usage with weekly summaries
gamma = 0.0001
start_date = '2021-01-01'
end_date = '2025-12-31'
predicted_strategies = predict_strategies(start_date, end_date, gamma)

# Print the predicted strategies for each market
for market in markets:
    print(f"Predicted strategies for market {market}:")
    for date, freqs in predicted_strategies[market]:
        print(f"Date: {date}, Frequencies: {freqs}")

# Plot the predicted strategy frequency against the actual strategy frequency for each market separately
for i, market in enumerate(markets):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    for idx, strategy in enumerate(strategy_columns):
        # Plot actual frequencies
        axs[idx].plot(weekly_summaries[i]['Week'], weekly_summaries[i][strategy], label=f'Actual {market}', linestyle='-')
        
        # Plot predicted frequencies
        predicted_freqs = [freqs[idx] for date, freqs in predicted_strategies[market]]
        predicted_dates = [date for date, freqs in predicted_strategies[market]]
        axs[idx].plot(predicted_dates, predicted_freqs, label=f'Predicted {market}', linestyle='--')
        
        axs[idx].set_ylabel('Frequency')
        axs[idx].set_title(titles[idx])
        axs[idx].legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    axs[-1].set_xlabel('Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()