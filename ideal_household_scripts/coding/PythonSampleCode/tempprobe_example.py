"""
@author JK

This script uses Niklas Berliner's data interface module to read in
some temperature probe data for a home, and display it as a graph

Usage: python tempprobe_example.py --homeid=62 --samplerate=1000
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IdealDataInterface import IdealDataInterface

# Set up arguments to be parsed and their default values
parser = argparse.ArgumentParser(description='Display some IDEAL room data from CSV files')
parser.add_argument('--homeid',type=int, default=106, help='home to process, default 106')
parser.add_argument('--roomid',type=str, default='1085', help='room to process, default 1085')
parser.add_argument('--inputdir', type=str, default='../../sensordata/', help='directory where source files are')
parser.add_argument('--samplerate', type=str, default="600", help='sample rate in seconds')
args = parser.parse_args()

# initialize the data interface
idi = IdealDataInterface(args.inputdir)

print("room "+str(args.roomid))
readings = idi.get(homeid=args.homeid,roomid=args.roomid,category='tempprobe')

# display the data on a graph
rlist=[]
legend=[]
for res in readings:
    if not res['subtype'].startswith('battery'):
        # downsample the data to the specified granuarity - best to do this before plotting
        rlist.append(res['readings'].resample(args.samplerate+"s").mean())
        legend.append(res['subtype'])
    
combo=pd.concat(rlist, axis=1)
combo /= 10
plt.figure()
plt.plot(combo)
plt.legend(legend)
plt.title("Home "+str(args.homeid)+": clamp temperature")
plt.xlabel("Date / time")
plt.ylabel("temp clamp")
plt.show()

