import os
import csv
import subprocess
import signal
import sys
import numpy as np

current_path = os.getcwd()

def signal_handler(signal, frame):
    os.chdir(current_path)
    sys.exit(0)

def get_dataset(filename):
    # Go to the root of the git repository
    root = subprocess.check_output("git rev-parse --show-toplevel", shell=True).rstrip()
    os.chdir(root)
    signal.signal(signal.SIGINT, signal_handler)

    #initialize the dataset/headers
    dataset = np.zeros(1)
    header = []

    num_lines = sum(1 for line in open(filename))

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        header = [h.lower() for h in reader.next()]

        #assume that the timestamp is the first col
        dataset = np.zeros((num_lines - 1, len(header)), dtype='float64')
        for x,row in enumerate(reader):
            for y, item in enumerate(row):
                if y == 0:
                    continue;
                dataset[x, y] = item

    os.chdir(current_path)
    return dataset, header[1:]

# filename = 'btc_data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv'
filename = 'stock_data/Stocks/aapl.us.txt'
data, headers = get_dataset(filename)

print headers
print data[0]
