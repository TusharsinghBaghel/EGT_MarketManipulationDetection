import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go

# Strategy assignment function
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

# Seasonal factor function
def seasonal_factor(month):
    # Adjusted to have a period of 12 months and values between ~0.85 and ~1.15
    return 1 + 0.2 * np.sin((2 * np.pi * (month - 1)) / 12)

# Production cost function
def production_cost(row):
    base_cost = 1000  # base cost per quintal
    inflation_rate = 0.05  # 5% inflation per year
    year_diff = row['Reported Date'].year - 2000  # base year 2000
    inflation_adjusted_cost = base_cost * ((1 + inflation_rate) ** year_diff)
    month = row['Reported Date'].month
    return inflation_adjusted_cost * seasonal_factor(month)

# Wholesale price function (returns average price as scalar)
selling_df = pd.read_csv("selling.csv")

def wholesale_price(row):
    state = row['State']
    year = row['Reported Date'].year
    month = row['Reported Date'].month
    
    # Mapping month number to month name
    month_mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_name = month_mapping[month]
    
    state_prices = selling_df[selling_df['State'] == state]
    state_prices['Year'] = state_prices['Year'].astype(int)  # Convert year to integer
    price_value = state_prices[(state_prices['Year'] == year) & (state_prices['Month'] == month_name)]['Price'].mean()
    print(price_value)
    return price_value

# Demand factor function
def demand_factor(row):
    two_months_ago = row['Reported Date'] - pd.DateOffset(months=1)
    total_arrival = 0.5 * df[(df['Reported Date'] >= two_months_ago) & (df['Reported Date'] < row['Reported Date'])]['ArrivalQuintals'].sum()
    month = row['Reported Date'].month
    demand = seasonal_factor(month) / (total_arrival + 1)
    return demand

# Utility function (profit)
def utility_function(row):
    demand = demand_factor(row)
    arrival = row['ArrivalQuintals']
    buying_price = row['ModalPrice']
    avg_wholesale_price = wholesale_price(row)
    return (avg_wholesale_price - buying_price) * arrival * demand

###############################################
# Data Preprocessing and renaming columns
###############################################
df = pd.read_csv("agmarknet2.csv", parse_dates=['Reported Date'])
df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d-%b-%y')
markets = df['Market Name'].unique().tolist()

for col in ["Arrivals (Tonnes)", "Modal Price (Rs./Quintal)"]:
    df.loc[:, col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

df["Arrivals Quintals"] = df["Arrivals (Tonnes)"] * 10
df.drop(columns=["Arrivals (Tonnes)"], inplace=True)
df['year'] = df['Reported Date'].dt.year

# Rename columns
df.rename(columns={"Arrivals Quintals": "ArrivalQuintals"}, inplace=True)
df.rename(columns={"Modal Price (Rs./Quintal)": "ModalPrice"}, inplace=True)
df.rename(columns={"Min Price (Rs./Quintal)": "MinPrice"}, inplace=True)
df.rename(columns={"Max Price (Rs./Quintal)": "MaxPrice"}, inplace=True)
df.rename(columns={"Commodity": "Commodity"}, inplace=True)
df.rename(columns={"State Name": "State"}, inplace=True)
df.rename(columns={"District Name": "District"}, inplace=True)
df.rename(columns={"Market Name": "Market"}, inplace=True)

# Split dataframe by market
df_market = []
for market in markets:
    df_market.append(df[df["Market"] == market])

################################################################################
# Add exterior columns: strategy, production cost, demand factor, and profit
################################################################################

priceThreshold = []
arrivalThreshold = []

for i in range(len(df_market)):
    priceThreshold.append(np.percentile(df_market[i]["ModalPrice"], 67))
    arrivalThreshold.append(np.percentile(df_market[i]["ArrivalQuintals"], 67))

for i in range(len(df_market)):
    df_market[i]['Strategy'] = df_market[i].apply(assign_strategy, axis=1, marketindex=i)
    df_market[i]['ProductionCost'] = df_market[i].apply(production_cost, axis=1)
    df_market[i]['wholeSalePrice'] = df_market[i].apply(wholesale_price, axis=1)
    df_market[i]['DemandFactor'] = df_market[i].apply(demand_factor, axis=1)    
    df_market[i]['profit'] = df_market[i].apply(utility_function, axis=1)

# Print counts of different strategies for each market
for i in range(len(df_market)):
    print(df_market[i]["Strategy"].value_counts())

####################################################################
# Plotting the data for a specific market and year
####################################################################

# Prompt user to select a market index
print("Select a market index from the following list:")
for idx, market in enumerate(markets):
    print(f"{idx}: {market}")

market_index = int(input("Enter the market index: "))
if market_index < 0 or market_index >= len(markets):
    raise ValueError("Invalid market index selected.")

df_market_sorted = df_market[market_index].sort_values(by='Reported Date')
year = int(input("Enter the year for which you want to plot the data: "))
df_market_year = df_market_sorted[df_market_sorted['Reported Date'].dt.year == year]

# Group by week and aggregate data
df_market_year['Week'] = df_market_year['Reported Date'].dt.to_period('W').apply(lambda r: r.start_time)
df_market_weekly = df_market_year.groupby('Week').agg({
    'ArrivalQuintals': 'sum',
    'Reported Date': 'first',
    'DemandFactor': 'first',
    'wholeSalePrice': 'first',
    'ModalPrice': 'mean'
}).reset_index()

# Convert 'Week' to datetime for Plotly
df_market_weekly['Week'] = pd.to_datetime(df_market_weekly['Week'])
#print some values of df_market_weekly
print(df_market_weekly.head())
# Create Plotly figure with secondary y-axis
fig = make_subplots(rows=3, cols=2, subplot_titles=("Arrival Quintals", "Demand Factor", "Seasonal Factor", "Wholesale Price", "Modal Price", "Utility Function"))

# Add traces with separate y-axes
fig.add_trace(
    go.Scatter(x=df_market_weekly['Week'], y=df_market_weekly['ArrivalQuintals'], name='Arrival Quintals', line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_market_weekly['Week'], y=df_market_weekly['DemandFactor'], name='Demand Factor', line=dict(color='orange')),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=df_market_weekly['Week'], y=df_market_weekly['Week'].dt.month.apply(seasonal_factor), name='Seasonal Factor', line=dict(color='green')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=df_market_weekly['Week'], y=df_market_weekly['wholeSalePrice'], name='Wholesale Price', line=dict(color='red')),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(x=df_market_weekly['Week'], y=df_market_weekly['ModalPrice'], name='Modal Price', line=dict(color='purple')),
    row=3, col=1
)

# Plot profit (utility function) using the original daily dates
fig.add_trace(
    go.Scatter(x=df_market_year['Reported Date'], y=df_market_year['profit'], name='Utility Function', line=dict(color='gray')),
    row=3, col=2
)

# Update layout
fig.update_layout(
    title_text="Market Data Analysis",
    height=1000,
    width=1200
)

fig.show()

####################################################################
# Group markets weekly and calculate frequencies plot of strategies
####################################################################

weekly_summaries = []

for i in range(len(df_market)):
    df_market[i]['Week'] = df_market[i]['Reported Date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_summary = df_market[i].groupby('Week').agg({
        'ArrivalQuintals': 'sum',
        'ModalPrice': 'mean',
        'DemandFactor': 'mean',
        'ProductionCost': 'mean',
        'profit': 'sum',
        'Strategy': lambda x: x.value_counts(normalize=True).to_dict()
    }).reset_index()

    strategy_columns = ['lPrice_lArrival', 'lPrice_hArrival', 'hPrice_lArrival', 'hPrice_hArrival']
    for col in strategy_columns:
        weekly_summary[col] = weekly_summary['Strategy'].apply(lambda x: x.get(col, 0))
    weekly_summary.drop(columns=['Strategy'], inplace=True)
    weekly_summaries.append(weekly_summary)

titles = ['Low Price Low Arrival', 'Low Price High Arrival', 'High Price Low Arrival', 'High Price High Arrival']
fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=titles)

for idx, strategy in enumerate(strategy_columns):
    for i, summary in enumerate(weekly_summaries):
        fig2.add_trace(
            go.Scatter(x=summary['Week'], y=summary[strategy], name=f'{markets[i]} - {titles[idx]}'),
            row=idx+1, col=1
        )

fig2.update_layout(height=1200, width=800, title_text="Strategy Frequencies Over Time")
fig2.update_xaxes(title_text="Week", row=4, col=1)
fig2.update_yaxes(title_text="Frequency")
fig2.show()
