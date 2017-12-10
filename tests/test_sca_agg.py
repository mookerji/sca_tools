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

from click.testing import CliRunner

import pytest

import sca_tools.sca_agg as sca_agg

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../fixtures',
)


@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'sysbench_cpu_64_60_sec_2000_prime.txt.csv'),
    on_duplicate='ignore',
)
def test_main(datafiles):
    runner = CliRunner()
    for datafile in datafiles.listdir():
        args = [
            '--load_column',
            'threads',
            '--throughput_column',
            'throughput',
            str(datafile),
            str(datafile),
            str(datafile),
        ]
        result = runner.invoke(sca_agg.main, args)
        assert result.exit_code == 0
