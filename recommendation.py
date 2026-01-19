import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from mpl_toolkits.mplot3d import Axes3D

# Load the data (assuming CSV format)
file_path = "agmarknet2.csv"  # Update with the actual file path
df = pd.read_csv(file_path, parse_dates=["Reported Date"], dayfirst=True)
df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d-%b-%y')

#create a list of unique market names
unique_marketnames = df['Market Name'].unique()

# Create Year, Month, and Week columns
df['year'] = df['Reported Date'].dt.year
df['week'] = df['Reported Date'].dt.isocalendar().week  # Week number of the year
df["Market-Week-Year"] = df["Market Name"] + "-" + df["year"].astype(str) + "-W" + df["week"].astype(str)

# Ensure numeric columns are properly converted
numeric_columns = ["Arrivals (Tonnes)", "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)", "Modal Price (Rs./Quintal)"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define aggregation functions
agg_funcs = {
    "Arrivals (Tonnes)": "sum",  # Total arrivals in that period
    "Min Price (Rs./Quintal)": "mean",  # Average of min prices
    "Max Price (Rs./Quintal)": "mean",  # Average of max prices
    "Modal Price (Rs./Quintal)": "mean"  # Average of modal prices
}

# Aggregate by Market-Week-Year
df_weekly = df.groupby(["State Name", "District Name", "Market-Week-Year", "Variety", "Group"]).agg(agg_funcs).reset_index()

# Prepare data for matrix factorization
df_weekly["Arrival-Price-Week"] = (
    df_weekly["Variety"] + "-Arrivals-" + df_weekly["Arrivals (Tonnes)"].astype(str) +
    "-Price-" + df_weekly["Modal Price (Rs./Quintal)"].astype(str)
)

pivot_table = df_weekly.pivot_table(
    index=["Market-Week-Year"],
    columns=["Arrival-Price-Week"],
    values="Arrivals (Tonnes)",
    fill_value=0
)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_table)

# Apply Truncated SVD for Latent Vector Extraction
svd = TruncatedSVD(n_components=3, random_state=42)  # Reduce to 3 latent dimensions
latent_vectors = svd.fit_transform(X_scaled)

# Compute cosine distances to find unique market-weeks
distance_matrix = cosine_distances(latent_vectors)
market_uniqueness = np.mean(distance_matrix, axis=1)  # Higher value = more unique

# Create a DataFrame with uniqueness scores
unique_markets = pd.DataFrame({
    "Market-Week-Year": pivot_table.index,
    "Uniqueness Score": market_uniqueness
})

# Sort by most unique market-weeks
unique_markets = unique_markets.sort_values(by="Uniqueness Score", ascending=False)

# Save results
unique_markets.to_csv("unique_market_weeks.csv", index=False)
print("Data preprocessing and latent vector computation completed! Unique market-weeks identified.")

# Print latent vector of all market-weeks with their names and scores
latent_vector_df = pd.DataFrame(latent_vectors, index=pivot_table.index, columns=[f"Latent Dimension {i+1}" for i in range(latent_vectors.shape[1])])
latent_vector_df["Uniqueness Score"] = market_uniqueness
# Plot these latent vectors in 3D space to visualize the uniqueness using Plotly
import plotly.express as px
# Add a new column for Market IDs
# Assign a unique index to each market
market_index_mapping = {market: idx for idx, market in enumerate(unique_marketnames)}

# Extract Market Name robustly (everything before the trailing '-YYYY-WWW')
index_series = latent_vector_df.index.to_series()
market_names_from_index = index_series.str.replace(r'-(\d+)-W\d+$', '', regex=True)
latent_vector_df["Market Index"] = market_names_from_index.map(lambda m: market_index_mapping.get(m, -1))

# Also extract Year and Week from the index for hover info
year_week = index_series.str.extract(r'-(?P<Year>\d+)-W(?P<Week>\d+)$')
latent_vector_df["Year"] = year_week["Year"].astype(str)
latent_vector_df["Week"] = year_week["Week"].astype(str)
# Create a 3D scatter plot with Market IDs as labels
fig = px.scatter_3d(
    latent_vector_df,
    x="Latent Dimension 1",
    y="Latent Dimension 2",
    z="Latent Dimension 3",
    color="Uniqueness Score",
    size_max=10,
    title="3D Visualization of Latent Vectors and Uniqueness Scores",
    hover_data=None,  # Remove all default hover data
    # Provide only the desired fields in custom_data for hover
    custom_data=["Market Index", "Week", "Year"]
)

fig.update_traces(
    hovertemplate=(
    "<b>market %{customdata[0]} - year %{customdata[2]} - week %{customdata[1]}</b><extra></extra>"
    )
)

fig.update_layout(
    scene=dict(
        xaxis=dict(title=dict(text="Latent Dim-1", font=dict(size=30))),
        yaxis=dict(title=dict(text="Latent Dim-2", font=dict(size=30))),
        zaxis=dict(title=dict(text="Latent Dim-3", font=dict(size=30)))
    ),
    coloraxis_colorbar=dict(
        title=dict(text="Uniqueness Score", font=dict(size=20)),
        tickfont=dict(size=20)
    ),
    hoverlabel=dict(
        font=dict(size=18)
    )
)

fig.show()

print(latent_vector_df)
