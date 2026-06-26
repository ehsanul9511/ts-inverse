"""
@author JK

This script uses Niklas Berliner's data interface module to read in
some room data for a home, and then displays it as a graph, indexed by
the room type

Usage: python roomdata_example.py --homeid=62 --measure='humidity' --samplerate=1000
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IdealDataInterface import IdealDataInterface

# Set up arguments to be parsed and their default values
parser = argparse.ArgumentParser(description='Display some IDEAL room data from CSV files')
parser.add_argument('--homeid',type=int, default=106, help='home to process, default 106')
parser.add_argument('--measure',type=str, default='temperature', help='room measurement to use: humidity, temperature or light, default temperature')
parser.add_argument('--inputdir', type=str, default='../../sensordata/', help='directory where source files are')
parser.add_argument('--samplerate', type=str, default="600", help='sample rate in seconds')
args = parser.parse_args()

# initialize the data interface
idi = IdealDataInterface(args.inputdir)

# grab some readings: all 12s room readings for a home that correspond
# with the 'measure' (temperature, humidity or light)
readings = idi.get(homeid=args.homeid,category='room',subtype=args.measure)

# display the data on a graph
rlist=[]
legend=[]
for res in readings:
    # downsample the data to the specified granuarity - best to do this before plotting
    rlist.append(res['readings'].resample(args.samplerate+"s").mean())
    legend.append(res['room_type'])
    
combo=pd.concat(rlist, axis=1)
combo /= 10
plt.figure()
plt.plot(combo)
plt.legend(legend)
plt.title("Home "+str(args.homeid)+": room "+args.measure)
plt.xlabel("Date / time")
plt.ylabel(args.measure)
plt.show()

