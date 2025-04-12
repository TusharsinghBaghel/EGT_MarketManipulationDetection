import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
# Removed unused import
import warnings
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
# Wholesale price function (returns average price as scalar)
selling_df = pd.read_csv("selling.csv")
k = 130

# Strategy assignment function
def assign_strategy(row, market_index):
    strat_string = ""
    avg_arrival = np.percentile(df_market[market_index]['ArrivalQuintals'], 63)
    if row['ArrivalQuintals'] < avg_arrival:
        strat_string += 'lArrival_'
    else:
        strat_string += 'hArrival_'
    
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
    return 0.82*demand_factor #*scale

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
    print(f"whole sale price:{price_value} of {state} in {year} for month {month_name}")
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
    df_market[i]['Strategy'] = df_market[i].apply(assign_strategy, axis=1, market_index=i)
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

    strategy_columns = ['lArrival_lPrice', 'lArrival_hPrice', 'hArrival_lPrice', 'hArrival_hPrice']
    for col in strategy_columns:
        weekly_summary[col] = weekly_summary['Strategy'].apply(lambda x: x.get(col, 0))
    weekly_summary.drop(columns=['Strategy'], inplace=True)
    weekly_summaries.append(weekly_summary)

titles = ['Low Arrival Low Price', 'Low Arrival High Price', 'High Arrival Low Price', 'High Arrival High Price']
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
#add a column of month in  df_market[i] for each market i
for i in range(len(df_market)):
    df_market[i]['month'] = df_market[i]['Reported Date'].dt.month_name()

###############################################################################################################
# For each market, sort the data by reported date, add new columns for week and year, 
# then group by 3 weeks and calculate the frequency ratio of each strategy for that 3-week period.

grouped_markets = []  # Initialize a list to store grouped data for all markets

for i in range(len(df_market)):
    # Sort data by reported date
    df_market[i] = df_market[i].sort_values(by=['Reported Date'])
    
    # Add week and year columns
    df_market[i]['Week'] = df_market[i]['Reported Date'].dt.isocalendar().week
    df_market[i]['Year'] = df_market[i]['Reported Date'].dt.year
    
    # Calculate the minimum date across all markets
    
    # Assign group index: every 3 weeks = 1 group, based on the global minimum date
    df_market[i]['Week_Group'] = (df_market[i]['Reported Date'].dt.dayofyear // 21)
    
    # Group by Week_Group and calculate frequency ratios for each strategy
    grouped = df_market[i].groupby('Week_Group').agg({
        'Reported Date': lambda x: f"{x.min().date()} to {x.max().date()}",
        'Strategy': lambda x: x.value_counts(normalize=True).to_dict(),
        'month': 'first',
        'year': 'first',
        'profit': 'sum'
    }).reset_index()
    
    # Add frequency ratio columns for each strategy
    strategy_columns = ['lArrival_lPrice', 'lArrival_hPrice', 'hArrival_lPrice', 'hArrival_hPrice']
    for col in strategy_columns:
        grouped[col] = grouped['Strategy'].apply(lambda x: x.get(col, 0))
    
    # Drop the original Strategy column
    grouped.drop(columns=['Strategy'], inplace=True)
    
    # Append the grouped data for the current market to the list
    grouped_markets.append(grouped)
    
    # Print the grouped data for the current market
    print(f"Market {i}:")
    print(grouped.head())

print(len(grouped_markets))
#print the size of each market
for i in range(len(grouped_markets)):
    print(f"Market {i} size: {grouped_markets[i].shape}")

def normalizing_factor(row, marketIndex):
    year = row['year']
    groupNo = row['Week_Group']

    # Initialize the summation of payoffs
    total_payoff = 0
    previous_year = row['year'] - 1
    try:
        current_payoff = grouped_markets[marketIndex][
            (grouped_markets[marketIndex]['Week_Group'] == row['Week_Group']) & 
            (grouped_markets[marketIndex]['year'] == previous_year)
        ]['profit'].sum()
    except KeyError:
        current_payoff = 0  # Default to 0 if no data is found
        print(f"No data found for market {marketIndex} in year {previous_year}")

    # Iterate through all markets
    for market in grouped_markets:
        # Filter data for the same group number and years up to the current year -1
        relevant_data = market[(market['Week_Group'] == groupNo) & (market['year'] < year)]

        
        # Calculate the summation of payoffs for the relevant data
        total_payoff += relevant_data['profit'].sum()
    
    num_years = len(relevant_data['year'].unique())
    num_markets = len(grouped_markets)
    nf = total_payoff - num_years*num_markets*current_payoff
    print(f"Normalizing factor for market {marketIndex} in year {year}, group {groupNo}: {nf}")
    return nf


def predictedStrategy(row, market_index):
    # Create a log file to store structured data
    log_file = "learning_log.csv"
    with open(log_file, "a") as f:
        # Write header if the file is empty
        if f.tell() == 0:
            f.write("Market,Year,GroupNo,SkippedCount,LearningCount,StrategyDiff,lArrival_lPrice,lArrival_hPrice,hArrival_lPrice,hArrival_hPrice\n")

    # Create a current strategy array with 4 values corresponding to the 4 strategies
    previous_data = grouped_markets[market_index][
        (grouped_markets[market_index]['Week_Group'] == row['Week_Group']) & 
        (grouped_markets[market_index]['year'] == row['year'] - 1)
    ]

    if previous_data.empty:
        # If no previous data exists, initialize to zeros
        current_strategy = np.zeros(4)
        current_payoff = 0
    else:
        # Use the previous year's strategy and payoff
        current_strategy = previous_data[['lArrival_lPrice', 'lArrival_hPrice', 'hArrival_lPrice', 'hArrival_hPrice']].values.flatten()
        current_payoff = previous_data['profit'].sum()

    older_strategy = current_strategy.copy()

    normalizing_factor_value = normalizing_factor(row, market_index)
    year = row['year']
    groupNo = row['Week_Group']
    final_strategy = current_strategy
    skipped_count = 0
    learning_count = 0
    
    for market in grouped_markets:
        starting_year = min(market['year'])
        for past_year in range(starting_year, year):  
            strategy = market[(market['Week_Group'] == groupNo) & (market['year'] == past_year)][['lArrival_lPrice', 'lArrival_hPrice', 'hArrival_lPrice', 'hArrival_hPrice']].values.flatten()
            if strategy.size == 0:
                skipped_count += 1  # Increment skipped count
                continue  # Skip if no strategy data is available

            payoff = market[(market['Week_Group'] == groupNo) & (market['year'] == past_year)]['profit'].values
            payoff_diff = payoff - current_payoff
            alpha = payoff_diff / normalizing_factor_value
            strategy_diff = strategy - current_strategy
            current_strategy = current_strategy + alpha * strategy_diff
            learning_count += 1  # Increment learning count

    # Calculate strategy difference
    strategy_diff = current_strategy - older_strategy

    # Log the data to the file
    with open(log_file, "a") as f:
        f.write(f"{markets[market_index]},{year},{groupNo},{skipped_count},{learning_count},{strategy_diff.tolist()},{current_strategy[0]},{current_strategy[1]},{current_strategy[2]},{current_strategy[3]}\n")

    return current_strategy
#print length of grouped markets
print("Length of grouped markets: ", len(grouped_markets))

for i in range(len(grouped_markets)):
    predicted_strategies = grouped_markets[i].apply(predictedStrategy, axis=1, market_index=i)
    grouped_markets[i]['predicted_lArrival_lPrice'] = predicted_strategies.apply(lambda x: x[0])
    grouped_markets[i]['predicted_lArrival_hPrice'] = predicted_strategies.apply(lambda x: x[1])
    grouped_markets[i]['predicted_hArrival_lPrice'] = predicted_strategies.apply(lambda x: x[2])
    grouped_markets[i]['predicted_hArrival_hPrice'] = predicted_strategies.apply(lambda x: x[3])

# Plot separate comparisons for each strategy column (actual vs predicted) for each market using Plotly
# Define strategy columns
strategy_columns = ['lArrival_lPrice', 'lArrival_hPrice', 'hArrival_lPrice', 'hArrival_hPrice']
predicted_columns = ['predicted_lArrival_lPrice', 'predicted_lArrival_hPrice', 'predicted_hArrival_lPrice', 'predicted_hArrival_hPrice']

# Sort grouped_markets by 'year' and 'Week_Group' to ensure chronological order
for market in grouped_markets:
    market.sort_values(by=['year', 'Week_Group'], inplace=True)

# Create a subplot for each strategy column
for strategy, predicted in zip(strategy_columns, predicted_columns):
    fig = make_subplots(rows=len(grouped_markets), cols=1, shared_xaxes=True, 
                        subplot_titles=[f"{markets[i]} - {strategy}" for i in range(len(grouped_markets))])

    for i, market in enumerate(grouped_markets):
        # Add actual strategy trace
        fig.add_trace(
            go.Scatter(
                x=market['Week_Group'], 
                y=market[strategy], 
                mode='lines+markers', 
                name=f"Actual {strategy} - {markets[i]}",
                line=dict(color='blue')
            ),
            row=i+1, col=1
        )
        
        # Add predicted strategy trace
        fig.add_trace(
            go.Scatter(
                x=market['Week_Group'], 
                y=market[predicted], 
                mode='lines+markers', 
                name=f"Predicted {strategy} - {markets[i]}",
                line=dict(color='red', dash='dot')
            ),
            row=i+1, col=1
        )

        # Add year and month group labels to the x-axis
        fig.update_xaxes(
            tickmode='array',
            tickvals=market['Week_Group'],
            ticktext=[f"{row['year']} - {row['month']}" for _, row in market.iterrows()],
            row=i+1, col=1,
            categoryorder='array',  # Ensure sorting of x-axis labels
            categoryarray=sorted(market['Week_Group'].unique())  # Sort by Week_Group
        )

    # Update layout
    fig.update_layout(
        height=300 * len(grouped_markets),  # Adjust height based on the number of markets
        width=1000,
        title_text=f"Actual vs Predicted {strategy} for Each Market",
        xaxis_title="Year - Month Group",
        yaxis_title=f"{strategy} Frequency"
    )

    fig.show()