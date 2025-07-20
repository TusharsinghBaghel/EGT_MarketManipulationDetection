import pandas as pd

# Load CSV files
df1 = pd.read_csv('out_of_threshold_strategies.csv')
df2 = pd.read_csv('unique_market_weeks.csv', nrows=64)  # Top 66 rows only

# Ensure correct types
df1['GroupNo'] = df1['GroupNo'].astype(int)
df1['Year'] = df1['Year'].astype(int)
df1['Market'] = df1['Market'].astype(str)

# Expand df1: create 3 weeks per GroupNo
expanded_rows = []
for _, row in df1.iterrows():
    for week_offset in range(-1,3):
        week = (row['GroupNo'] - 1) * 3 + week_offset + 1
        key = f"{str(row['Market']).strip()}-{row['Year']}-W{week}"
        expanded_rows.append({
            'Market': row['Market'],
            'Year': row['Year'],
            'Week': week,
            'Key': key
        })

# Create new expanded DataFrame
df1_expanded = pd.DataFrame(expanded_rows)

# Clean df2 and make sure headers are correct
df2.columns = ['Key', 'Uniqueness Score']
df2['Key'] = df2['Key'].astype(str).str.strip()

# Debug print: show a few key values to inspect formatting
print("🔍 Sample df1 Keys:", df1_expanded['Key'].unique()[:5])
print("🔍 Sample df2 Keys:", df2['Key'].unique()[:5])

# Compare
set1 = set(df1_expanded['Key'])
set2 = set(df2['Key'])

common_keys = set1 & set2

# Calculate accuracy
accuracy = len(common_keys) / 64

# Print results
print(f"\n✅ Common Entries: {len(common_keys)}")
print(f"✅ Accuracy: {accuracy:.2%}")

# Optional: list matched keys
print("\n🎯 Matched Keys:")
for key in sorted(common_keys):
    print(f" - {key}")
