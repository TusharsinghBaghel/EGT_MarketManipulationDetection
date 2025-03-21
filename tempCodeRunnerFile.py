import pandas as pd

file_path = "sitedata.csv"  
df = pd.read_csv(file_path)

df_melted = df.melt(id_vars=["State"], var_name="Month-Year", value_name="Price")

# Split 'Month-Year' column into 'Month' and 'Year'
df_melted[['Month', 'Year']] = df_melted['Month-Year'].str.split('-', expand=True)

# Convert two-digit year to four-digit year
df_melted['Year'] = df_melted['Year'].apply(lambda x: '20' + x if len(str(x)) == 2 else x)

# Convert 'Year' column to integer
df_melted['Year'] = df_melted['Year'].astype(int)

df_melted['Price'] = df_melted['Price'] 

# Reorder columns
df_final = df_melted[['State', 'Price', 'Month', 'Year']]

# Save the transformed data
output_file = "selling.csv"
df_final.to_csv(output_file, index=False)

print(f"Transformed data saved to {output_file}")
