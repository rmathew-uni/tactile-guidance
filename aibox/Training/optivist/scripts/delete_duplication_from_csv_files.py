# This is script to remove the duplicates from csv file.
# since the script creating the csv file from json annotation have created with 'a' by writing csv.
# There are some duplication.

import os
import csv
import sys
import pandas as pd

# the directory path
directory = '/share/neurobiopsychologie/datasets/egohands/csv_val/'

# Get all the file names in the directory
file_names = [folder for folder in os.listdir(directory) if os.path.isfile(os.path.join(directory, folder))]
# only keep the .csv files
csv_files = [file for file in file_names if file.endswith('.csv')]

# go through each csv and delete the duplication and write a new csv file.
for csv_file in csv_files:
    # Read the input CSV file
    df = pd.read_csv(f'/share/neurobiopsychologie/datasets/egohands/csv_val/{csv_file}', header=None)
    # Remove duplicates based on all columns
    df.drop_duplicates(inplace=True)
    # Write the unique rows to a new CSV file
    df.to_csv(f'/share/neurobiopsychologie/datasets/egohands/csv_val/{csv_file}', header=False, index=False)



