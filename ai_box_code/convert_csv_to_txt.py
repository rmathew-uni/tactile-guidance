import os
import csv
import sys

# the directory for csv path
directory = 'csv'

# Get all the file names in the directory
file_names = [folder for folder in os.listdir(directory) if os.path.isfile(os.path.join(directory, folder))]
# only keep the .csv files
csv_files = [file for file in file_names if file.endswith('.csv')]
# Remove the ".csv" extension from file names
csv_files = [os.path.splitext(file)[0] for file in file_names if file.endswith('.csv')]

# csv to txt file and writing to txt folder 
for file_name in csv_files:
    csv_file = f'csv/{file_name}.csv'
    txt_file = f'txt/{file_name}.txt'
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()