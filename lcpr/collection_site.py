import os
import pandas as pd
import matplotlib.pyplot as plt
import re


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













