import pandas as pd

# Load datasets
agmarknet_df = pd.read_csv("agmarknet2.csv", parse_dates=['Reported Date'])
selling_df = pd.read_csv("selling.csv")

# Standardize column names and formats
print(selling_df['Year'].unique())
#check dtype of 'Year' column
print(selling_df['Year'].dtype)
#check for null values in 'Year' column
print(selling_df['Year'].isnull().sum())
#check for null values in 'Price' column
print(selling_df['Price'].isnull().sum())
#dtype of all columns
print(selling_df.dtypes)
#dtype of all columns of agmarknet_df
print(agmarknet_df.dtypes)
# import pandas as pd

# # Load data
# df = pd.read_csv("selling.csv")

# # Convert 'Year' to integer safely
# df['Year'] = df['Year'].fillna(0).astype(int)  # Replace NaN with 0 before conversion

# # Save the cleaned data
# df.to_csv("selling_cleaned.csv", index=False)

# print("Successfully converted Year to integer and saved as 'selling_cleaned.csv'")
