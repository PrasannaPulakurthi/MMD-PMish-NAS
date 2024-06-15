import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Create a DataFrame from the table data
data = {
    'Method': [
        'Fixed Rank', 'Fixed Rank', 'Fixed Rank', 'Fixed Rank',
        'Fixed Rank with NC_9_13', 'Fixed Rank with NC_9_13', 'Fixed Rank with NC_9_13', 'Fixed Rank with NC_9_13',
        'Adaptive Rank', 'Adaptive Rank', 'Adaptive Rank', 'Adaptive Rank', 'Adaptive Rank'
    ],
    'Rank or RP': [128, 256, 512, 680, 128, 256, 512, 680, 1, 1/5, 1/10, 1/15, 1/20],
    'Size in MB': [1.88, 2.74, 4.48, 5.61, 5.02, 5.75, 7.22, 8.18, 2.55, 3.21, 5.25, 6.62, 8.32],
    'FID': [35.87, 18.25, 10.09, 9.08, 13.76, 8.26, 6.44, 6.24, 13.22, 9.52, 7.67, 6.63, 5.85]
}

df = pd.DataFrame(data)

# Create a larger plot with a grid
fig, ax = plt.subplots(figsize=(12, 8))

# Plot Fixed Rank with dotted line
fixed_rank = df[df['Method'] == 'Fixed Rank']
ax.plot(fixed_rank['Size in MB'], fixed_rank['FID'], marker='s', linestyle='-', color='gold', markersize=10, linewidth=2, label='Fixed Rank')

# Plot Fixed Rank with NC_9_13 with dotted line
fixed_rank_nc = df[df['Method'] == 'Fixed Rank with NC_9_13']
ax.plot(fixed_rank_nc['Size in MB'], fixed_rank_nc['FID'], marker='o', linestyle='-', color='blue', markersize=8, linewidth=2, label='Fixed Rank with NC_9_13')

# Plot Adaptive Rank with dotted line
adaptive_rank = df[df['Method'] == 'Adaptive Rank']
ax.plot(adaptive_rank['Size in MB'], adaptive_rank['FID'], marker='*', linestyle='-', color='red', markersize=12, linewidth=2, label='Adaptive Rank')

# Customize the plot
ax.set_xlabel('Size in MB', fontsize=14)
ax.set_ylabel('FID', fontsize=14)
ax.set_title('Size vs FID for Different Methods', fontsize=16)
ax.legend(fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

# Save the new plot
plot_path_updated = "size_vs_fid_plot.png"
plt.savefig(plot_path_updated)
plot_path_updated