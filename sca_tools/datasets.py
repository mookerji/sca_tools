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

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, load_col=None, tput_col=None, tput_err_col=None):
        self._data = data
        self._load_col = load_col
        self._tput_col = tput_col
        self._tput_err_col = tput_err_col

    @property
    def load(self):
        assert self._load_col is not None
        return self._data.get(self._load_col, None)

    @property
    def has_fixed_load(self):
        return np.isclose(self.load.std(), 0)

    @property
    def throughput(self):
        assert self._tput_col is not None
        return self._data.get(self._tput_col, None)

    @property
    def error(self):
        if self._tput_err_col is not None:
            return self._data.get(self._tput_err_col, None)
        else:
            return None

    @property
    def weights(self):
        return 1 / np.square(self.error)

    def to_csv(self, filename):
        return self._data.sort_values(by=self._load_col).to_csv(filename)


def read_frame(filename, load_col=None, tput_col=None, tput_err_col=None):
    return Dataset(
        pd.read_csv(filename, infer_datetime_format=True, index_col=0),
        load_col,
        tput_col,
        tput_err_col,
    )


def aggregate_frames(dfs, load_column, throughput_column,
                     throughput_errors_column):
    records = []
    for df in dfs:
        assert df.has_fixed_load
        records.append({
            load_column: df.load.mean(),
            throughput_column: df.throughput.mean(),
            throughput_errors_column: df.throughput.std(),
        })
    return Dataset(
        pd.DataFrame(records),
        load_column,
        throughput_column,
        throughput_errors_column,
    )
