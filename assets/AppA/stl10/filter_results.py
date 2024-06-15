import pandas as pd

# Load the CSV file
file_path = 'scores_data.csv'
df = pd.read_csv(file_path)

# Convert 'FID Score' column to string and split it
df['FID Score'] = df['FID Score'].astype(str)
df['Score'] = df['FID Score'].astype(float)
df.drop(columns=['FID Score'], inplace=True)

# Split the 'Name' column to extract the parts
df[['Prefix1', 'Prefix2', 'Seed', 'Date', 'Time']] = df['Name'].str.split('_', n=5, expand=True)[[1, 2, 3, 4, 5]]
df['Seed'] = df['Seed'].astype(int)
print(df)
df['Common_Name'] = df['Name'].str.split('_').str[:3].str.join('_')

# Pivot the DataFrame to get scores for each seed
pivot_df = df.pivot(index='Common_Name', columns='Seed', values='Score')

# Define the seed values
seeds = [11111, 22222, 33333, 44444, 55555]

# Create a new DataFrame with the required structure
output_df = pd.DataFrame(index=pivot_df.index)

for seed in seeds:
    if seed in pivot_df.columns:
        output_df[f'{seed}'] = pivot_df[seed]
    else:
        output_df[f'{seed}'] = None

# Reset the index to include 'Common_Name' as a column
output_df.reset_index(inplace=True)

# Sorting the rows based on the common naming pattern
def custom_sort(name):
    parts = name.split('_')
    base = parts[0]
    number = parts[1].split('-')
    return (base, float(number[-1]))

output_df = output_df.sort_values(by='Common_Name', key=lambda col: col.map(custom_sort))

# Write the output DataFrame to a new CSV file
output_file_path = 'rearranged_sorted_scores_data.csv'
output_df.to_csv(output_file_path, index=False)

output_file_path
