import re

def search_and_replace(file_path):
   with open(file_path, 'r') as file_contents:
       file_contents = file_contents.read()

       # Step 1: Replace dots with dashes in the date
       file_contents = re.sub(r"(\d{2})\.(\d{2})\.(\d{4})", r"\1-\2-\3", file_contents)

       # Step 2: Replace the comma in the numeric value with a dot
       file_contents = re.sub(r",", ".", file_contents)

       # Step 3: Replace semicolons with commas
       file_contents = re.sub(r";", ",", file_contents)

       # Step 4: Add :00 to the time
       file_contents = re.sub(r"(\d{2}:\d{2})", r"\1:00", file_contents)

       print(file_contents)

   with open(file_path, 'w') as file:
       file.write(file_contents)


# Example usage
file_path = '../data/NVE/Gaulfoss-Vannstand-time-v1.csv'
search_and_replace(file_path)