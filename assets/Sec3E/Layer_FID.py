import pandas as pd
import matplotlib.pyplot as plt

# Reading data from Excel file
df = pd.read_excel('sorted_scores.xlsx', engine='openpyxl')

# Clamping FID scores at 30 if they exceed this value
df['FID score'] = df['FID score'].apply(lambda x: min(x, 30) if x > 30 else x)

# Identifying unique ranks within the DataFrame
unique_ranks = df['Rank'].unique()

# Setting up the figure for plotting
fig, ax = plt.subplots(figsize=(9, 6))  # Adjust the size as needed


rank = sorted(unique_ranks)[6]
rank_df = df[df['Rank'] == rank]

# Plotting
ax.bar(rank_df['Layers'], rank_df['FID score'], label=f'Rank {rank}')
ax.set_title(f'Rank {rank}, Compressed Layers vs FID Score', fontsize=18)
ax.set_xlabel('Layers', fontsize=16)
ax.set_ylabel('FID Score', fontsize=16)
ax.axhline(y=4.97, color='r', linestyle='--', linewidth=2, label='Target FID (4.97)')
ax.legend(fontsize=16)


plt.tight_layout()
plt.savefig('Layer_FID_128.png')  # Save the plot as a PNG file
