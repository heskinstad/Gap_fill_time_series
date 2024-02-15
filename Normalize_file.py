import pandas as pd

# Read the CSV file
df = pd.read_csv('data/Train/Daily-train.csv')

# Extract descriptive rows and columns
descriptive_rows = df.iloc[:, :1]  # First column as descriptive
descriptive_columns = df.iloc[:1, :]  # First row as descriptive

# Exclude descriptive rows and columns for normalization
data = df.iloc[1:, 1:]  

# Normalize the data
normalized_data = (data - data.min()) / (data.max() - data.min())

# Concatenate descriptive rows and columns with normalized data
normalized_df = pd.concat([descriptive_rows, normalized_data], axis=1)
normalized_df = pd.concat([descriptive_columns, normalized_df], axis=0)

# Save the normalized data to a new CSV file
normalized_df.to_csv('data/Train/normalized_file.csv', index=False)
