import os
import csv
import subprocess
import signal
import sys


# Go to the root of the
current_path = os.getcwd()
root = subprocess.check_output("git rev-parse --show-toplevel", shell=True).rstrip()
os.chdir(root)

def signal_handler(signal, frame):
    os.chdir(current_path)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


dataset = {}
header = []

with open('btc_data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    header = reader.next()
    for row in reader:
        dataset[int(row[0])] = row[1:]

print header
print dataset.keys
print dataset[dataset.keys[0]]

os.chdir(current_path)
