# pylint: disable=no-member

# Copyright 2017 Bhaskar Mookerji
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset management utilities"""

import numpy as np
import pandas as pd
import scipy.stats


class Dataset(object):
    """Utility container for load-throughput data, using a
    pandas.DataFrame under the hood.

    Parameters
    ----------
    data: pandas.DataFrame
        Underlying data.
    load_col: str
        Name of the load/concurrency column.
    tput_col: str
        Name of the throughput column.
    tput_err_col: str
        Name of the throughput standard deviation column.

    Returns
    -------
    Dataset

    """

    def __init__(self, data, load_col=None, tput_col=None, tput_err_col=None):
        self._data = data
        self._load_col = load_col
        self._tput_col = tput_col
        self._tput_err_col = tput_err_col

    @property
    def load(self):
        """Returns float array of load data.

        """
        assert self._load_col is not None
        return self._data.get(self._load_col, None)

    @property
    def load_col(self):
        """Returns name of load column."""
        return self._load_col

    @load_col.setter
    def load_col(self, value):
        """Set the name of the load column."""
        self._load_col = value

    @property
    def has_fixed_load(self):
        """Returns True if the dataset has a fixed/constant load/concurrency,
        False otherwise.

        """
        return np.isclose(self.load.std(), 0)

    @property
    def has_normal_throughput(self):
        """Returns True if the dataset has a normally-distributed throughput,
        False otherwise.

        """
        p_value, _ = scipy.stats.shapiro(self.throughput)
        ALPHA = 0.05
        return p_value > ALPHA

    @property
    def throughput(self):
        """Returns float array of throughput data."""
        assert self._tput_col is not None
        return self._data.get(self._tput_col, None)

    @property
    def throughput_col(self):
        """Returns name of throughput column."""
        return self._tput_col

    @throughput_col.setter
    def throughput_col(self, value):
        """Set the name of the throughput column."""
        self._tput_col = value

    @property
    def error(self):
        """Returns float array of throughput stddev."""
        if self._tput_err_col is not None:
            return self._data.get(self._tput_err_col, None)

    @property
    def error_col(self):
        """Returns name of throughput stddev column."""
        return self._tput_err_col

    @error_col.setter
    def error_col(self, value):
        """Set the name of the throughput stddev column."""
        self._tput_err_col = value

    @property
    def weights(self):
        """Returns a weighting of throughput measurements."""
        return 1 / np.square(self.error)

    def drop_outliers(self, z_score=3):
        """
        Drops, in-place, throughput outliers from the dataset.

        Parameters
        ----------
        z_score : float

        Returns
        -------
        True if outliers dropped, False otherwise.

        """
        zs = np.abs(scipy.stats.zscore(self._data.throughput))
        self._data = self._data[zs < z_score]
        return not (zs < z_score).all()

    def to_csv(self, filename):
        """Write dataset to a file.

        Parameters
        ----------
        filename : str
        """
        self._data.sort_values(by=self._load_col).to_csv(filename)


def read_frame(filename, load_col=None, tput_col=None, tput_err_col=None):
    """Reads a CSV file into a Dataset.

    Parameters
    ----------
    filename : str
        Filename to read
    load_col : str
        Name of the load column
    tput_col : str
        Name of the throughput column
    tput_err_col : str
        Name of the throughput stddev column

    Returns
    -------
    sca_tools.datasets.Dataset
    """
    return Dataset(
        pd.read_csv(filename, infer_datetime_format=True, index_col=0),
        load_col,
        tput_col,
        tput_err_col,
    )


def aggregate_frames(frames, load_column, throughput_column,
                     throughput_errors_column):
    """Aggregates datasets from multi-trial throughput measurements.

    Parameters
    ----------
    frames : [sca_tools.datasets.Dataset]
        Set of throughput measurements, where each Dataset has a
        different load/concurrency.
    load_column : str
        Name of the load column
    throughput_column : str
        Name of the throughput column
    throughput_errors_column : str
        Name of the throughput stddev column

    Returns
    -------
    sca_tools.datasets.Dataset

    """
    records = []
    for frame in frames:
        assert frame.has_fixed_load
        assert frame.has_normal_throughput
        records.append({
            load_column: frame.load.mean(),
            throughput_column: frame.throughput.mean(),
            throughput_errors_column: frame.throughput.std(),
        })
    return Dataset(
        pd.DataFrame(records),
        load_column,
        throughput_column,
        throughput_errors_column,
    )
