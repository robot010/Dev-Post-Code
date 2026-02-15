import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Prepare the data from your experiments
data = {
    'Experiment': [
        'Exp 1: 10M Rows\n(Text + Int)', 'Exp 1: 10M Rows\n(Text + Int)',
        'Exp 2: 10M Rows\n(Text Only)', 'Exp 2: 10M Rows\n(Text Only)',
        'Exp 3: 1M Rows\n(Text Only)', 'Exp 3: 1M Rows\n(Text Only)'
    ],
    'Pandas Version': [
        'Pandas 2.3 (Object)', 'Pandas 3.0 (Arrow)',
        'Pandas 2.3 (Object)', 'Pandas 3.0 (Arrow)',
        'Pandas 2.3 (Object)', 'Pandas 3.0 (Arrow)'
    ],
    'Memory (MB)': [
        734.33, 343.32,  # Exp 1
        658.04, 267.03,  # Exp 2
        65.80, 26.70     # Exp 3 (Scaled)
    ]
}

df = pd.DataFrame(data)

# 2. Set the visual style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# Define custom colors (Pro blue and Success green)
colors = ["#3498db", "#2ecc71"]
ax = sns.barplot(
    data=df, 
    x='Experiment', 
    y='Memory (MB)', 
    hue='Pandas Version', 
    palette=colors
)

# 3. Add Title and Labels
plt.title('Memory Savings from Upgrading to pandas 3.0 (NumPy Object vs. PyArrow String)', 
          fontsize=16, pad=20, fontweight='bold')
plt.ylabel('Memory Usage (MB)', fontsize=12)
plt.xlabel('', fontsize=12)
plt.legend(title='Engine')

# 4. Add data labels and reduction percentages
for i in range(len(ax.containers)):
    # Annotate the actual MB values on top of bars
    ax.bar_label(ax.containers[i], fmt='%.2f MB', padding=3, fontsize=10)
    
    # Calculate and annotate the % reduction for the Pandas 3.0 bars
    if i == 1: # The "Pandas 3.0 (Arrow)" containers
        for j, bar in enumerate(ax.containers[i]):
            val_30 = df.iloc[j*2 + 1]['Memory (MB)']
            val_23 = df.iloc[j*2]['Memory (MB)']
            reduction = ((val_23 - val_30) / val_23) * 100
            
            # Position the text inside or above the green bar
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() / 2, 
                f'-{reduction:.1f}%', 
                ha='center', va='center', color='white', 
                fontweight='bold', fontsize=11
            )

# Clean up the layout and save
plt.tight_layout()
plt.savefig('pandas_3_migration_benchmark.png', dpi=300)
plt.show()