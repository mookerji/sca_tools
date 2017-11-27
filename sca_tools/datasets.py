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

import pandas as pd


class Dataset(object):
    def __init__(self, data, load_col=None, tput_col=None):
        self._data = data
        self._load_col = load_col
        self._tput_col = tput_col

    @property
    def load(self):
        assert self._load_col is not None
        return self._data[self._load_col]

    @property
    def throughput(self):
        assert self._tput_col is not None
        return self._data[self._tput_col]


def read_frame(filename, load_col=None, tput_col=None):
    return Dataset(
        pd.read_csv(filename, infer_datetime_format=True, index_col=0),
        load_col,
        tput_col,
    )
