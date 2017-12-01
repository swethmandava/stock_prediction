import os
import csv
import subprocess
import signal
import sys
import numpy as np
from collections import deque

current_path = os.getcwd()

class DataStreaming:
    def __init__(self, seed, reader, predict_dist):
        self.reader = reader
        # print seed
        self.last = deque(seed.tolist())
        self.predict_dist = predict_dist

    def __iter__(self):
        return self

    def next(self): # Python 3: def __next__(self)
        n = self.reader.next()
        data = np.zeros(len(n)-1, dtype='float64')
        for i, f in enumerate(n[1:]):
            data[i] = f
        current = self.last.popleft()
        self.last.append(data)
        return current, data[0]

def signal_handler(signal, frame):
    os.chdir(current_path)
    sys.exit(0)

def get_data(filename, initial_size=200, predict_dist=1):
    # Go to the root of the git repository
    root = subprocess.check_output("git rev-parse --show-toplevel", shell=True).rstrip()
    os.chdir(root)
    signal.signal(signal.SIGINT, signal_handler)

    num_lines = sum(1 for line in open(filename))
    initial_data_size = min(num_lines, initial_size)

    csvfile =  open(filename, 'rb')
    reader = csv.reader(csvfile)
    header = [h.lower() for h in reader.next()]

    #assume that the timestamp is the first col
    dataset = np.zeros((initial_data_size, len(header) - 1), dtype='float64')
    y = np.zeros((num_lines, 1), dtype='float64')
    for x,row in enumerate(reader):
        if x >= initial_size:
            if x <= 0:
                raise ValueError('File should contain more data')
            stream = DataStreaming(dataset[x-predict_dist:], reader, predict_dist=predict_dist)
            break;

        for y, item in enumerate(row):
            if y == 0:
                continue
            try:
                float(item)
                dataset[x, y-1] = (item)
            except ValueError:
                continue


    os.chdir(current_path)
    # data(OHLC), y(open of the day predict_dist later), headers(data labels)
    return dataset[:-predict_dist], dataset[predict_dist:, 0], stream, header[1:]

if __name__ == '__main__':
    # filename = 'btc_data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv'
    # filename = 'stock_data/Stocks/aple.us.txt'
    # filename = 'stock_data/NASDAQ_AAPL.txt'
    filename = 'stock_data/AAPL.csv'
    # data, y, stream, headers = get_data(filename, initial_size=300, predict_dist=2)
    data, y, stream, headers = get_data(filename, initial_size=300)

    # print headers
    # print y[0]
    print data[0]
    print stream.next()
    print stream.next()
    #Sample safe streaming till end
    try:
        while True:
            print stream.last, "*"
            a = stream.next()
            print a
    except StopIteration:
        pass
