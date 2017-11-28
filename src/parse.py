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
    y = np.zeros(1)
    header = []

    num_lines = sum(1 for line in open(filename))

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        header = [h.lower() for h in reader.next()]

        #assume that the timestamp is the first col
        dataset = np.zeros((num_lines, len(header) - 1), dtype='float64')
        y = np.zeros((num_lines, 1), dtype='float64')
        for x,row in enumerate(reader):
            for y, item in enumerate(row):
                if y == 0:
                    continue;
                dataset[x, y-1] = item


    os.chdir(current_path)
    # data(OHLC), y(open of the next day), headers(data labels)
    return dataset[:-1], dataset[1:, 0], header[1:]

# filename = 'btc_data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv'
filename = 'stock_data/Stocks/aple.us.txt'
data, y, headers = get_dataset(filename)

print headers
print y[0]
print data[0]
