import os
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from copy import deepcopy
import re

# from lcpr.collection_site import CollectionSite


class CollectionSite:

    @property
    def meter_pulses(self):
        return self.data['Water_Meter_Pulses']

    @property
    def time_stamps(self):
        return self.data['TimeStamp']

    @property
    def battery_voltages(self):
        return self.data['Battery_Voltage']

    @property
    def record_numbers(self):
        return self.data['RecordNumber']

    @property
    def memory_available(self):
        return self.data['Mem_Space_Available']

    def __init__(self, logger_name, site_name, description, data):
        self.logger_name = logger_name
        self.site_name = site_name
        self.description = description
        self.data = data

    def __str__(self):
        return self.site_name

    @staticmethod
    def create_from_file(path):

        logger_name_match = re.compile('.*DataloggerName: (?P<logger_name>[0-9]+).*', flags=re.IGNORECASE)
        site_name_match = re.compile(r'.*(?P<site_name>Site [0-9]+).*', flags=re.IGNORECASE)
        desc_name_match = re.compile(r'.*SiteDescription: (?P<desc>.*)\n', flags=re.IGNORECASE)

        with open(path, 'r') as fin:
            # Skip the first line of the CSV files - its always the same
            fin.readline()

            log_line = fin.readline()
            lmatch = logger_name_match.match(log_line)
            logger_name = lmatch.group('logger_name')

            site_name_line = fin.readline()
            smatch = site_name_match.match(site_name_line)
            site_name = smatch.group('site_name')

            desc_line = fin.readline()
            desc_match = desc_name_match.match(desc_line)
            desc = desc_match.group('desc')

            df = pd.read_csv(path, skiprows=4)
            df = df.drop(["N/A", "N/A.1", "N/A.2"], axis=1)
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S.%f")

            return CollectionSite(logger_name, site_name, desc, df)

    def dropna(self):
        self.data.dropna(0, inplace=True, how='any')

    def lenpoints(self):
        return len(self.data)

    def _plot(self):
        x = self.time_stamps
        y = self.meter_pulses

        plt.plot(x, y)
        plt.ylim(bottom=0)
        plt.xticks(rotation=90)

    def _show(self):
        plt.show()
        plt.clf()

    def plot(self):
        self._plot()
        self._show()

    def save_plot(self, path, fname=None, dpi=1000, ftype="png"):
        self._plot()

        if fname is None:

            fname = "{ts}-{name}.{format}".format(
                ts=str(self.time_stamps[0]),
                name=self.site_name,
                format=ftype
            )

        plt.savefig(os.path.join(path, fname), format=ftype, dpi=dpi)


def load_data(path):
    # site_data = defaultdict(lambda: [])

    for dirpath, dnames, fnames in os.walk(path):
        for fname in fnames:
            if fname.endswith(".csv"):
                path = os.path.join(os.path.abspath(dirpath), fname)
                site = CollectionSite.create_from_file(path)
                yield dirpath, site


def filter_site_events(site, min_data_points=0, min_pulse_threshold=0):
    """
    Splits data into individual DataFrames where each new frame contains
    non-zero measurements that occur between 0-valued measurements.

    For example:
        data = [0, 0, 0, 1, 2, 3, 0, 0, 3, 3, 4, 3, 0, 0, 5]
        ... <filtering occurs>
        data = [[1, 2, 3], [3, 3, 4, 3], [5]]
    """

    df = DataFrame({'timestamp': site.time_stamps, 'pulse': site.meter_pulses})
    df.dropna(0, inplace=True, how='any')

    event_groups = []
    for _, values in df[df.pulse.ne(0)].groupby(df.pulse.eq(0).cumsum()):

        # filter out groups having less than 3 points of data
        if len(values) < min_data_points:
            continue

        # filter out groups whose pulses don't pass a certain threshold
        if any([v < min_pulse_threshold for v in values.pulse]):
            continue

        event_groups.append(values)

    return event_groups


def save_plot(df, path, fname=None, dpi=300, ftype="png"):
    x = df.timestamp
    y = df.pulse

    plt.plot(x, y)
    plt.xlabel('Time (DD HH:MM)')
    plt.ylabel('Water Meter Pulse (RPM)')

    start = df.timestamp.iloc[0]
    end = df.timestamp.iloc[-1]

    start = datetime.strptime(str(start), "%Y-%m-%d %H:%M:%S.%f")
    end = datetime.strptime(str(end), "%Y-%m-%d %H:%M:%S.%f")
    tdelta = end - start
    minutes = int(tdelta.seconds / 60)
    seconds = int(tdelta.seconds - (minutes * 60))

    title = "{s} - {e}, {m} minutes, {sec} seconds".format(s=start.strftime("%I:%M %p"),
                                                           e=end.strftime("%I:%M %p"),
                                                           m=minutes, sec=seconds)

    plt.title(title)
    # plt.tight_layout()

    if fname is None:

        fname = "{ts}.{format}".format(
            ts=str(df.timestamp.iloc[0]),
            format=ftype
        )

    plt.savefig(os.path.join(path, fname), format=ftype, dpi=dpi)
    plt.clf()


def group_events():
    datapath = "./data/"
    sitespath = os.path.abspath(os.path.join(datapath, "sites", "site0002"))
    outputpath = os.path.abspath(os.path.join(datapath, "output"))

    sites = [site for _, site in load_data(sitespath)]
    event_groups = []

    for i, site in enumerate(sites):
        groups: list = filter_site_events(site, min_data_points=2, min_pulse_threshold=3)

        event_groups.append(groups)

        for i, df in enumerate(groups):  # type: DataFrame
            df.reindex()
            start = deepcopy(df.timestamp.iloc[0]) - timedelta(seconds=4)

            df.loc[-1] = (start, 0)
            df.index = df.index + 1
            df = df.sort_index()

            end = df.timestamp.iloc[-1] + timedelta(seconds=4)
            df.loc[-1] = (end, 0)

            groups[i] = df
            # save_plot(df, outputpath)

    lengroups = sum([len(groups) for groups in event_groups])
    totalpoints = sum([site.lenpoints() for site in sites])
    print("Event Groups: %d" % len(event_groups))
    print("Total Events: %d" % lengroups)
    print("Total points: %d" % totalpoints)

    return event_groups


