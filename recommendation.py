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
svd = TruncatedSVD(n_components=3, random_state=42)  # Reduce to 5 latent dimensions
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
latent_vector_df["Market ID"] = latent_vector_df.index.map(lambda x: market_index_mapping.get(x.split("-")[0], -1))

# Create a 3D scatter plot with Market IDs as labels
fig = px.scatter_3d(
    latent_vector_df,
    x="Latent Dimension 1",
    y="Latent Dimension 2",
    z="Latent Dimension 3",
    color="Uniqueness Score",
    # Use a high-contrast color scale
    size_max=10,
    title="3D Visualization of Latent Vectors and Uniqueness Scores",
    hover_name=latent_vector_df.index.map(
        lambda x: f"Market {market_index_mapping.get(x.split('-')[0], -1)}, Week {x.split('-')[-1][1:]}, Year {x.split('-')[-2]}"
    )  # Add "Market i, Week w, Year y" as labels
)

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis=dict(title=dict(text="Latent Dimension 1", font=dict(size=15))),
        yaxis=dict(title=dict(text="Latent Dimension 2", font=dict(size=15))),
        zaxis=dict(title=dict(text="Latent Dimension 3", font=dict(size=15)))
    ),
    coloraxis_colorbar=dict(title="Uniqueness Score")
)

# Show the plot
fig.show()

print(latent_vector_df)