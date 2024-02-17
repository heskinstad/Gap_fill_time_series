import os
import re

# Define the path to the folder containing the text files
folder_path = "../data/NDBC - Train/41002"

# Loop through all the files in the specified folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Process each line to replace spaces with a comma, preserving line shifts
        modified_lines = [re.sub(r"\s+", ",", line.strip()) for line in lines]

        # Join the modified lines back together, adding a newline character after each line
        modified_content = "\n".join(modified_lines)

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(modified_content)

print("All files have been modified, preserving line shifts.")
