import shutil

# Define the target directory
target_directory = "validation_data/"

# Read the file containing the paths
with open("data/testing_list.txt", "r") as file:
    for line in file:
        file_path = line.strip()
        shutil.copy(file_path, target_directory)
