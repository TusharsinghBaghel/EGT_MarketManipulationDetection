# Plot the average of the wholesale price and selling price for all the months of the year for Madhya Pradesh, only for years 2021-2024.
state_name = "Madhya Pradesh"
years_to_plot = [2021, 2022, 2023, 2024]

# Filter data for Madhya Pradesh from df_market and only for selected years
df_madhyapradesh = pd.concat([market for market in df_market if market['State'].iloc[0] == state_name])
df_madhyapradesh['Year'] = df_madhyapradesh['Reported Date'].dt.year
df_madhyapradesh['Month'] = df_madhyapradesh['Reported Date'].dt.month
df_madhyapradesh = df_madhyapradesh[df_madhyapradesh['Year'].isin(years_to_plot)]

# Group by year and month, then calculate the average wholesale price and selling price
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

# Update layout with larger legend font
fig3.update_layout(
    title=f"Average Monthly Prices for {state_name} (2021-2024)",
    xaxis_title="Month",
    yaxis_title="Price",
    xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
    height=600,
    width=800,
    legend=dict(font=dict(size=16))
)
fig3.show()

# Add a column of month in df_market[i] for each market i
for i in range(len(df_market)):
    df_market[i]['month'] = df_market[i]['Reported Date'].dt.month_name()
