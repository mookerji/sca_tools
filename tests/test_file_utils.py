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

import sca_tools.file_utils as futils


def test_get_file_directory():
    dir_ = futils.get_file_directory('sca_tools/bar/foo.py')
    assert dir_.endswith('sca_tools/bar')


def test_get_file_basename():
    basename = futils.get_file_basename('sca_tools/bar/foo.py')
    assert basename.endswith('foo')
