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

from StringIO import StringIO

import pandas as pd
import pytest

import sca_tools.datasets as dset


@pytest.fixture(scope="module")
def default_data():
    """From fixtures/specsdm91.csv

    """
    df = pd.DataFrame.from_records({
        1: {
            'load': 1.0,
            'throughput': 64.9
        },
        2: {
            'load': 18.0,
            'throughput': 995.9
        },
        3: {
            'load': 36.0,
            'throughput': 1652.4
        },
        4: {
            'load': 72.0,
            'throughput': 1853.2
        },
        5: {
            'load': 108.0,
            'throughput': 1828.9
        },
        6: {
            'load': 144.0,
            'throughput': 1775.0
        },
        7: {
            'load': 216.0,
            'throughput': 1702.2
        }
    }, ).T
    df['throughput_error'] = df['throughput'] * 0.05
    return df


@pytest.fixture(scope="module")
def default_data_as_csv(default_data):
    buf = StringIO()
    default_data.to_csv(buf)
    buf.seek(0)
    return buf


@pytest.fixture(scope="module")
def default_data_as_df(default_data_as_csv):
    df = dset.read_frame(default_data_as_csv, 'load', 'throughput')
    default_data_as_csv.seek(0)
    return df
