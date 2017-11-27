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

import sca_tools.datasets as dset


def test_dataset_read_frame(default_data_as_csv):
    df = dset.read_frame(default_data_as_csv, 'load', 'throughput')
    assert isinstance(df, dset.Dataset)
    assert np.allclose(
        df.load.values,
        [1., 18., 36., 72., 108., 144., 216.],
    )
    assert np.allclose(
        df.throughput.values,
        [64.9, 995.9, 1652.4, 1853.2, 1828.9, 1775., 1702.2],
    )
