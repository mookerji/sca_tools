# pylint: disable=too-few-public-methods

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
"""Graphing utilities."""

import logging
import os

import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))


class GraphResult(object):
    """Container for a graphing artifact.

    Parameters
    ----------
    name: str
        Name of the artifact
    artifact: str
        Matplotlib Axes

    Returns
    -------
    GraphResult
    """

    def __init__(self, name, artifact):
        self.name = name
        self.artifact = artifact

    def to_png(self, basename, output_directory):
        """Render artifact to a png file.

        Parameters
        ----------
        basename: [str]
            Prefix for filename
        output_directory: str
            Directory to write to
        """
        filename = "%s/%s-%s.png" % (output_directory, basename, self.name)
        logging.debug('Writing to image to %s', filename)
        if isinstance(self.artifact, list) and self.artifact:
            assert len(self.artifact) == 1
            self.artifact[0].get_figure().savefig(filename)
        else:
            self.artifact.get_figure().savefig(filename)


def render_graphs(graphs, basename, output_directory):
    """
    Parameters
    ----------
    graphs: [sca_tools.graphing.GraphResult]
        Graph artifacts to render
    basename: [str]
        Prefix for filename
    output_directory: str
        Directory to write to

    """
    for graph in graphs:
        graph.to_png(basename, output_directory)
