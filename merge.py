import pandas as pd

# List of file names
file_list = ['bhopal.csv', 'ujjain.csv', 'hoshangabad.csv', 'indore.csv']

# Read and append all files
combined_df = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)

# Save to final.csv
combined_df.to_csv('finalultrapromax.csv', index=False)

print("All files merged into 'final.csv' successfully!")
