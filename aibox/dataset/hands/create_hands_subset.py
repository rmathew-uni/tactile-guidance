from pathlib import Path
import random
import os
import csv

csvfile = 'hands_fp_raw.csv'

with open(csvfile) as f:
    reader = csv.reader(f)
    next(reader)  # skip the header
    data = [row for row in reader]

for i in range(400):

    chosen_row = random.choice(data)
    img_name = chosen_row[0]

    new_csv = 'hands_subset.csv'

    # check if the CSV file exists and if it's empty
    if os.path.isfile(new_csv):
        # if the file exists append the new rows without the header
        with open(new_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(chosen_row)
            # print('added row')
    else:
        # if the file does not exist or is empty, write the header and the new rows

        with open(csvfile, mode='r', newline='') as sf, open(new_csv, mode='w', newline='') as df:
            reader = csv.reader(sf)
            writer = csv.writer(df)
            writer.writerow(next(reader))
            writer.writerow(chosen_row)
            # print('created file and added row')