# University of Edinburgh, United Kingdom
# IDEAL Project, 2019

import re
import warnings
import numpy as np
import pandas as pd
import re
from pathlib import Path

__TIME_FORMAT__ = '%Y-%m-%d %H:%M:%S'

class IdealDataInterface(object):
    """Interface to the IDEAL Local Data Interface."""

    def __init__(self, folder_path):

        # Make sure the warning is issued every time the user instantiates the class
        warnings.filterwarnings("always", category=UserWarning,
                                module='IdealDataInterface')

        # This will be used to search for the reading files in the directory.
        self.file_identifier = 'home*.csv.gz'

        self.folder_path = Path(folder_path)
        self.sensorid_mapping = self._mapping(self.folder_path)

        if len(self.sensorid_mapping) == 0:
            warnings.warn('The specified folder path does not seem to contain any sensor reading files.')


    def _mapping(self, folder_path):
        homeid = list()
        roomid = list()
        roomtype = list()
        sensorid = list()
        category = list()
        subtype = list()
        filename = list()

        for file in folder_path.glob(self.file_identifier):

            home_, room_, sensor_, category_, subtype_ = file.name.split('_')

            filename.append(str(file.name))
            homeid.append(int(re.sub('\D', '', home_)))
            roomid.append(int(re.sub('\D', '', room_)))
            roomtype.append(str(re.sub('\d', '', room_)))
            category.append(str(category_))
            subtype.append(str(subtype_[:-7]))

            assert sensor_[:6] == 'sensor'
            sensorid.append(str(sensor_[6:]))

        data = {'homeid': homeid, 'roomid': roomid, 'room_type':roomtype, 'sensorid':sensorid,
                'category': category, 'subtype': subtype, 'filename':filename}
        columns = ['homeid', 'roomid', 'room_type', 'category', 'subtype', 'sensorid', 'filename']

        df = pd.DataFrame(data, columns=columns, dtype=str)
        df.set_index(['homeid', 'roomid', 'room_type', 'category', 'subtype', 'sensorid'], inplace=True)

        print('Found entries for {} sensor readings.'.format(df.shape[0]))

        return df


    def _filter(self, homeid=None, roomid=None, room_type=None, category=None, subtype=None, sensorid=None):
        def check_input(x):
            """ Assert that the input is a list of strings. """
            if isinstance(x, int):
                x = [str(x), ]
            elif isinstance(x, str):
                x = [x, ]

            if not hasattr(x, '__iter__'):
                raise ValueError('Input {} not understood'.format(x))

            return [str(i) for i in x]

        # Select the matching sensors
        if homeid is None:
            homeid = slice(None)
        else:
            homeid = check_input(homeid)

        if roomid is None:
            roomid = slice(None)
        else:
            roomid = (roomid)

        if room_type is None:
            room_type = slice(None)
        else:
            room_type = check_input(room_type)

        if category is None:
            category = slice(None)
        else:
            category = check_input(category)

        if subtype is None:
            subtype = slice(None)
        else:
            subtype = check_input(subtype)

        if sensorid is None:
            sensorid = slice(None)
        else:
            sensorid = check_input(sensorid)

        filename = slice(None)

        # If homeid, roomid, and room_type are specified, the result will be a Series. This will be converted back
        # to a DataFrame (needs transposing to get it back into the original format)
        try:
            return self.sensorid_mapping.loc(axis=0)[homeid, roomid, room_type, category, subtype, sensorid, filename].to_frame().T
        except AttributeError:
            return self.sensorid_mapping.loc(axis=0)[homeid, roomid, room_type, category, subtype, sensorid, filename]


    def read_csv_(self, fname, subtype):
        """ Load the file to pandas DataFrame. """
        df = pd.read_csv(fname, header=None, names=['time', 'value'], parse_dates=['time'])

        # Sanity check make sure the date
        assert np.issubdtype(df.dtypes['time'], np.datetime64)

        # Convert input to pd.Series
        ts = pd.Series(df['value'].values, index=df['time'], name=subtype)

        return ts


    def load(self, homeid=None, roomid=None, room_type=None, category=None, subtype=None, sensorid=None):
        """ Iterator to load the readings specified by the filter options. """

        df = self._filter(homeid=homeid, roomid=roomid, room_type=room_type, category=category,
                          subtype=subtype, sensorid=sensorid)

        for (homeid, roomid, room_type, category, subtype, sensorid), row in df.iterrows():
            fname = self.folder_path / Path(row['filename'])

            ts = self.read_csv_(fname, subtype=subtype)

            yield ts, {'homeid': homeid, 'roomid': roomid, 'room_type': room_type,
                       'category': category, 'subtype': subtype, 'sensorid': sensorid}


    def get(self, homeid=None, roomid=None, room_type=None, category=None, subtype=None, sensorid=None):

        warnings.warn('get() is deprecated and might be removed in the future.', category=DeprecationWarning)

        df = self._filter(homeid=homeid, roomid=roomid, room_type=room_type, category=category,
                           subtype=subtype, sensorid=sensorid)

        readings = list()
        for (homeid, roomid, room_type, category, subtype, sensorid), row in df.iterrows():
            fname = self.folder_path / Path(row['filename'])

            ts = self.read_csv_(fname, subtype=subtype)

            readings.append({'homeid': homeid, 'roomid': roomid, 'room_type': room_type,
                             'category': category, 'subtype': subtype, 'sensorid': sensorid,
                             'readings': ts})

        return readings


    def view(self, homeid=None, roomid=None, room_type=None, category=None, subtype=None, sensorid=None):
        """ Get a list of available sensors given the filtering conditions. """
        df = self._filter(homeid=homeid, roomid=roomid, room_type=room_type, category=category,
                          subtype=subtype, sensorid=sensorid)

        return df.reset_index().loc[:,['homeid', 'roomid', 'room_type', 'category', 'subtype', 'sensorid']]


    def categories(self):
        """ Returns pd.DataFrame of the available categories and subtypes. """
        return self.sensorid_mapping.reset_index().loc[:,['category', 'subtype']].drop_duplicates().reset_index(drop=True)


    def room_types(self):
        """ Returns pd.DataFrame with room types and count in data set"""
        return self.sensorid_mapping.reset_index()[['roomid', 'room_type']]\
                                    .drop_duplicates()\
                                    .groupby('room_type')\
                                    .size()\
                                    .sort_values(ascending=False)\
                                    .rename('Number of rooms')
