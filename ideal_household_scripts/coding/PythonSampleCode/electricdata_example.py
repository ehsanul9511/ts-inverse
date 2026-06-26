"""
@author Cillian Brewitt, Jonathan Kilgour

This script reads mains electricity readings and graphs them. For enhanced homes it will show real vs apparent power.
JK edited to use Niklas Berliner's IdealDataInterface
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from IdealDataInterface import IdealDataInterface

# get arguments
parser = argparse.ArgumentParser(description='Clean electrical sensor readings and merge .')
parser.add_argument('--homeid',type=int, default=106, help='home to process, default all')
parser.add_argument('--outputdir', type=str, default='./mains_readings/', help='directory to write files')
parser.add_argument('--inputdir', type=str, default='../../sensordata/', help='directory where source files are')
parser.add_argument('--samplerate',type=str, default='300', help='sample rate in seconds')
args = parser.parse_args()

homeids = [args.homeid]

# initialize the data interface
idi = IdealDataInterface(args.inputdir)

# read in the 1 second combined IDEAL data
print('homeid: {0}'.format(args.homeid))
readings = idi.get(homeid=args.homeid,category='electric-mains',subtype='electric-combined')
rlist=[]
legend=[]
for res in readings:
    # downsample the data to the specified granuarity - best to do this before plotting
    myreadings = res['readings'][res['readings'].index > '2016-08-07']
    rlist.append(myreadings.resample(args.samplerate+"s").mean())
    legend.append(res['subtype'])


# If there is such a thing, now read in the 5 second OEM mains data (real power in Watts, 5 second data)
readings2 = idi.get(homeid=args.homeid,category='electric-subcircuit',subtype='mains')
for res in readings2:
    myreadings = res['readings'][res['readings'].index > '2017-07-07']
    # downsample the data to the specified granuarity
    rlist.append((myreadings).resample(args.samplerate+"s").mean())
    legend.append('OEM')
    
combo=pd.concat(rlist, axis=1)
plt.figure()
plt.plot(combo)
plt.legend(legend)
plt.title("Home "+str(args.homeid)+": electric mains")
plt.xlabel("Date / time")
plt.ylabel("Watts")
plt.show()
