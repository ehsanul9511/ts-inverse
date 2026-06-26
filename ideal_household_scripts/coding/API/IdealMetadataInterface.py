# University of Edinburgh, United Kingdom
# IDEAL Project, 2019

import re
import warnings
import numpy as np
import pandas as pd
import re
from pathlib import Path

__TIME_FORMAT__ = '%Y-%m-%d %H:%M:%S'

class IdealMetadataInterface(object):
    """Interface to the IDEAL Metadata Interface."""

    def __init__(self, folder_path):

        # Make sure the warning is issued every time the user instantiates the class
        warnings.filterwarnings("always", category=UserWarning,
                                module='IdealMetadataInterface')

        self.folder_path = Path(folder_path)
        self.metadata = self._mapping(self.folder_path)

        if len(self.metadata) == 0:
            warnings.warn('The specified folder path '+str(self.folder_path)+' does not seem to contain any metadata files.')

    def _mapping(self, folder_path):
        homes=None; rooms=None; appliances=None;
        sensorboxes=None; sensors=None; people=None;
        locations=None; weatherfeeds=None;
        try:
            homes=self._metafile("home");
            rooms=self._metafile("room");
            appliances=self._metafile("appliance");
            sensorboxes=self._metafile("sensorbox");
            sensors=self._metafile("sensor");
            people=self._metafile("person");
            locations=self._metafile("location");
            weatherfeeds=self._metafile("weatherfeed");
        except FileNotFoundError:
            warnings.warn('The specified folder path does not contain the correct metadata files.')
            exit

        data = {'homes': homes, 'rooms': rooms, 'appliances':appliances, 'sensorboxes':sensorboxes,
                'sensors':sensors,'people': people, 'locations': locations, 'weatherfeeds':weatherfeeds}
        columns = ['homes', 'rooms', 'appliances', 'sensorboxes', 'sensors', 'people', 'locations','weatherfeeds']

        df = pd.DataFrame(data, columns=columns, dtype=str)
        return df


    def _metafile(self, name):

        res = list()
        fname = self.folder_path / Path(name+".csv")
        df = pd.read_csv(fname, header=0, sep=',', encoding='utf-8-sig')
        return {name: df}

