import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
# Removed unused import
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
# Wholesale price function (returns average price as scalar)
selling_df = pd.read_csv("selling.csv")
k = 130

# Strategy assignment function
def assign_strategy(row):
    strat_string = ""
    avg_arrival = np.mean([np.mean(market['ArrivalQuintals']) for market in df_market])
    if row['ArrivalQuintals'] < avg_arrival*0.8:
        strat_string += 'lArrival_'
    elif row['ArrivalQuintals'] > avg_arrival*1.2:
        strat_string += 'hArrival_'
    else:
        strat_string += 'mArrival_'

    avg_sellingprice =  wholesale_price(row)    
    if row['sellingprice'] < avg_sellingprice:
        strat_string += 'lPrice'
    else:
        strat_string += 'hPrice'
        
    return strat_string

def h(demand):
    #if demand = 1.15 -> 1.2
    #if demand = 1 -> 0.8
    return 2.67*demand - 1.87

def g(arrival):
    #bw 0.8 and 1.2 
    #monotonous
    min_arrival = min(df['ArrivalQuintals'])
    max_arrival = max(df['ArrivalQuintals'])
    return 0.8 + 0.4*(arrival - min_arrival)/(max_arrival - min_arrival)

def SellingPrice(row):
    demand_factor = row['DemandFactor']
    arrival = row['ArrivalQuintals']
    buying_price = row['ModalPrice']
    return buying_price + k*h(demand_factor)*g(arrival)

def depletion_factor(row):
    demand_factor = row['DemandFactor']
    scale = -1
    avg_arrival = np.mean([np.mean(market['ArrivalQuintals']) for market in df_market])
    if row['ArrivalQuintals'] < avg_arrival*0.8:
        scale = 0.8
    elif row['ArrivalQuintals'] > avg_arrival*1.2:
        scale = 0.9
    else:
        scale = 1
    return 0.82*scale*demand_factor

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
    # Hardcoded demand factor values for each month
    demand_factors = {
        1: 1.0,   # January
        2: 1.05,  # February
        3: 1.1,   # March
        4: 1.15,  # April 
        5: 1.15,  # May
        6: 1.02,  # June
        7: 1.0,   # July
        8: 1.03,  # August
        9: 1.01,  # September
        10: 1.05, # October
        11: 1.04, # November
        12: 1.05  # December
    }
    month = row['Reported Date'].month
    return demand_factors[month]



# Utility function (profit)
def utility_function(row):
    selling_price = SellingPrice(row)
    buying_price = row['ModalPrice']
    return (selling_price - buying_price)*row['ArrivalQuintals']*depletion_factor(row)

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


for i in range(len(df_market)):
    demand_factors = df_market[i].apply(demand_factor, axis=1)
    df_market[i]['DemandFactor'] = demand_factors
    df_market[i]['sellingprice'] = df_market[i].apply(SellingPrice, axis=1)  # Ensure sellingprice is calculated first
    df_market[i]['Strategy'] = df_market[i].apply(assign_strategy, axis=1)
    df_market[i]['sellingprice'] = df_market[i].apply(SellingPrice, axis=1)
    df_market[i]['wholeSalePrice'] = df_market[i].apply(wholesale_price, axis=1)
    
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
    'ModalPrice': 'mean',
    'sellingprice': 'mean',
    'profit': 'sum'

}).reset_index()

# Convert 'Week' to datetime for Plotly
df_market_weekly['Week'] = pd.to_datetime(df_market_weekly['Week'])
#print some values of df_market_weekly
print(df_market_weekly.head())
# Create Plotly figure with secondary y-axis
fig = make_subplots(rows=3, cols=2, subplot_titles=("Arrival Quintals", "Demand Factor", "sellingprice", "Wholesale Price", "Modal Price", "Utility Function"))

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
    go.Scatter(x=df_market_weekly['Week'], y=df_market_weekly['sellingprice'], name='Selling Price', line=dict(color='green')),
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
        'sellingprice': 'mean',
        'profit': 'sum',
        'Strategy': lambda x: x.value_counts(normalize=True).to_dict()
    }).reset_index()

    strategy_columns = ['lArrival_lPrice', 'lArrival_hPrice', 'mArrival_lPrice', 'mArrival_hPrice', 'hArrival_lPrice', 'hArrival_hPrice']
    for col in strategy_columns:
        weekly_summary[col] = weekly_summary['Strategy'].apply(lambda x: x.get(col, 0))
    weekly_summary.drop(columns=['Strategy'], inplace=True)
    weekly_summaries.append(weekly_summary)

titles = ['Low Arrival Low Price', 'Low Arrival High Price', 'Medium Arrival Low Price', 'Medium Arrival High Price', 'High Arrival Low Price', 'High Arrival High Price']
fig2 = make_subplots(rows=6, cols=1, shared_xaxes=True, subplot_titles=titles)

for idx, strategy in enumerate(strategy_columns):
    for i, summary in enumerate(weekly_summaries):
        fig2.add_trace(
            go.Scatter(x=summary['Week'], y=summary[strategy], name=f'{markets[i]} - {titles[idx]}'),
            row=idx+1, col=1
        )

fig2.update_layout(height=1500, width=800, title_text="Strategy Frequencies Over Time")
fig2.update_xaxes(title_text="Week", row=6, col=1)
fig2.update_yaxes(title_text="Frequency")
fig2.show()

# Plot the average of the wholesale price and selling price for all the months of the year for Madhya Pradesh.
state_name = "Madhya Pradesh"

# Filter data for Madhya Pradesh from df_market
df_madhyapradesh = pd.concat([market for market in df_market if market['State'].iloc[0] == state_name])

# Group by year and month, then calculate the average wholesale price and selling price
df_madhyapradesh['Year'] = df_madhyapradesh['Reported Date'].dt.year
df_madhyapradesh['Month'] = df_madhyapradesh['Reported Date'].dt.month
monthly_avg_prices = df_madhyapradesh.groupby(['Year', 'Month']).agg({
    'wholeSalePrice': 'first',
    'sellingprice': 'mean'
}).reset_index()

# Plot the data
fig3 = go.Figure()

for year in monthly_avg_prices['Year'].unique():
    yearly_data = monthly_avg_prices[monthly_avg_prices['Year'] == year]
    fig3.add_trace(
        go.Scatter(x=yearly_data['Month'], y=yearly_data['wholeSalePrice'], 
                   mode='lines+markers', name=f'Wholesale Price {year}', line=dict(dash='solid'))
    )
    fig3.add_trace(
        go.Scatter(x=yearly_data['Month'], y=yearly_data['sellingprice'], 
                   mode='lines+markers', name=f'Selling Price {year}', line=dict(dash='dot'))
    )

# Update layout
fig3.update_layout(
    title=f"Average Monthly Prices for {state_name} Across All Years",
    xaxis_title="Month",
    yaxis_title="Price",
    xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
    height=600,
    width=800
)

fig3.show()
