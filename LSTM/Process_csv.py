import csv
import numpy as np
import pandas as pd


# Reads a csv file and places all values of a specified column into its own array
def process_csv_column(path, column_index, has_header=True, datetimes=False):
    column_data = []

    with open(path, mode='r') as file:
        reader = csv.reader(file)

        # Skip header if there is one
        if has_header:
            next(reader, None)

        # Iterate over each row. Add data from column
        if datetimes:
            for row in reader:
                column_data.append(row[column_index])
        else:
            for row in reader:
                column_data.append(float(row[column_index]))

    return column_data

def process_csv_column_with_timestamps(path):
    df = pd.read_csv(path)

    return df

# Reads a csv file and places all values of a specified row into its own array
def process_csv_row(path, row_index):
    row_data = None

    with open(path, mode='r') as file:
        reader = csv.reader(file)

        # Iterate over each row
        for i, row in enumerate(reader):
            if i == row_index:
                row_data = np.array([float(value) for value in row[1:] if value != ''], dtype=float)
                break

    return row_data