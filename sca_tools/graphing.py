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

import logging
import matplotlib
matplotlib.use('Agg')
import os

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class GraphResult(object):
    def __init__(self, name, artifact):
        self.name = name
        self.artifact = artifact

    def to_png(self, basename, output_directory):
        filename = "%s/%s-%s.png" % (output_directory, basename, self.name)
        logger.debug('Writing to image to %s' % filename)
        if isinstance(self.artifact, list) and len(self.artifact) > 0:
            assert len(self.artifact) == 1
            self.artifact[0].get_figure().savefig(filename)
        else:
            self.artifact.get_figure().savefig(filename)


def render_graphs(graphs, basename, output_directory):
    for graph in graphs:
        graph.to_png(basename, output_directory)
