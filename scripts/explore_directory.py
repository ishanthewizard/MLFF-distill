import os
import re

# Define the regex pattern for files that match the 'mp-number.extxyz' format
pattern = re.compile(r'^mp-\d+\.extxyz$')

# Path to the directory
folder_path = '/data/shared/MPTrj/original'

# Get a list of all files in the directory
files = os.listdir(folder_path)

# Print the total number of files in the directory
total_files = len(files)
print(f"Total number of files in the directory: {total_files}")

# List to store files that don't match the pattern
non_matching_files = []

# Check each file
for file in files:
    if not pattern.match(file):
        non_matching_files.append(file)

# Print or store the non-matching files
if non_matching_files:
    print("Files that don't match 'mp-number.extxyz' pattern:")
    for non_matching_file in non_matching_files:
        print(non_matching_file)
else:
    print("All files match the 'mp-number.extxyz' pattern.")
