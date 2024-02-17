import os

# Define the path to the folder containing the CSV files
folder_path = '../data/NDBC - Train/41002'

# Define the path for the output file
output_file_path = '../data/NDBC - Train/41002all.csv'
output_file_name = os.path.basename(output_file_path)  # Extracts the file name from the path

# Make sure the output file is not in the same folder as the input files
if folder_path == os.path.dirname(output_file_path):
    raise ValueError("The output file should not be in the same folder as the source files to avoid infinite loops.")

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        # Skip the output file if it exists in the folder
        if file_name == output_file_name:
            continue

        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)

        # Check if the current object is a file and not a directory
        if os.path.isfile(file_path):
            # Open the current file in read mode
            with open(file_path, 'r') as input_file:
                # Skip the first line (header)
                next(input_file)
                # Read the remaining lines and write them to the output file
                for line in input_file:
                    output_file.write(line)
