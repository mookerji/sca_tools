# pylint: disable=missing-docstring

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

import os

import numpy as np
import pytest

import sca_tools.datasets as dset

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../fixtures',
)


def test_dataset_read_frame(default_data_as_df):
    frame = default_data_as_df
    assert isinstance(frame, dset.Dataset)
    assert np.allclose(
        frame.load.values,
        [1., 18., 36., 72., 108., 144., 216.],
    )
    assert np.allclose(
        frame.throughput.values,
        [64.9, 995.9, 1652.4, 1853.2, 1828.9, 1775., 1702.2],
    )


@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'sysbench_cpu_64_60_sec_2000_prime.txt.csv'),
    on_duplicate='ignore',
)
def test_dataset_aggregate(datafiles):
    for datafile in datafiles.listdir():
        frame = dset.read_frame(
            datafile,
            'threads',
            'throughput',
            'throughput_stddev',
        )
        result = dset.aggregate_frames(
            frames=[frame],
            load_column='threads',
            throughput_column='throughput',
            throughput_errors_column='throughput_stddev',
        )
        assert isinstance(result, dset.Dataset)
        assert np.allclose(result.load.values, [64])
        assert np.allclose(result.throughput.values, [27506.885])
        assert np.allclose(result.error.values, [1493.63347])


@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'sysbench_cpu_64_60_sec_2000_prime.txt.csv'),
    on_duplicate='ignore',
)
def test_dataset_outliers(datafiles):
    for datafile in datafiles.listdir():
        frame = dset.read_frame(
            datafile,
            'threads',
            'throughput',
            'throughput_stddev',
        )
        assert frame._data.shape == (30, 4)
        assert frame.drop_outliers()
        assert frame._data.shape == (28, 4)
        result = dset.aggregate_frames(
            frames=[frame],
            load_column='threads',
            throughput_column='throughput',
            throughput_errors_column='throughput_stddev',
        )
        assert isinstance(result, dset.Dataset)
        assert np.allclose(result.load.values, [64])
        assert np.allclose(result.throughput.values, [27114.65])
        assert np.allclose(result.error.values, [38.226])
