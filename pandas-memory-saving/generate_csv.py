import pandas as pd
import numpy as np
import string

n_rows = 10_000_000 # 10M rows

# 1. Generate the data dictionary
data = {
    'text_col': [''.join(np.random.choice(list(string.ascii_letters), 20)) for _ in range(n_rows)],
    'int_col': np.random.randint(low=0, high=1000000, size=n_rows) # Integer column
}

# 2. Create the DataFrame and save to CSV
df_dummy = pd.DataFrame(data)
df_dummy.to_csv("test_loading_10M.csv", index=False)