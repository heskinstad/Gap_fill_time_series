import os


def clean_data(file_path):
    temp_file_path = file_path + ".temp"

    with open(file_path, 'r') as read_file, open(temp_file_path, 'w') as write_file:
        headers = read_file.readline()
        write_file.write(headers)

        columns = headers.strip().split(',')
        wtmp_index = len(columns) - 3

        for line in read_file:
            elements = line.strip().split(',')
            if elements[wtmp_index] != "999.0":
                write_file.write(line)

    os.remove(file_path)
    os.rename(temp_file_path, file_path)


def process_folder(folder_path):
    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if it is a file
        if os.path.isfile(file_path):
            print(f"Processing {filename}...")
            clean_data(file_path)
            print(f"{filename} has been processed.")


# Example usage
folder_path = '../data/NDBC - Train/41002'
process_folder(folder_path)
