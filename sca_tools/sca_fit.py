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

import click
import sys

import sca_tools.datasets as dat
import sca_tools.graphing as graph
import sca_tools.file_utils as futils
import sca_tools.usl as usl


class UnsupportedModelException(Exception):
    pass


@click.command()
@click.option('--load_column', default='load')
@click.option('--throughput_column', default='throughput')
@click.option('--throughput_errors_column', default=None)
@click.option('--model_type', default='usl')
@click.option('--output_directory', default=None)
@click.argument('filename')
def main(load_column, throughput_column, throughput_errors_column, model_type,
         output_directory, filename):
    df_spec = dat.read_frame(filename)
    df_spec._load_col = load_column
    df_spec._tput_col = throughput_column
    if throughput_errors_column:
        raise NotImplementedError()
    if not output_directory:
        output_directory_ = futils.get_file_directory(filename)
    else:
        output_directory_ = output_directory
    basename = futils.get_file_basename(filename)
    if model_type == 'usl':
        model = usl.USLModel()
        model_fit = model.fit(data=df_spec.throughput.values,
                              load=df_spec.load.values, lambda_=1000,
                              sigma_=0.1, kappa=0.001)
        graphs = usl.generate_graphs(model_fit, df_spec, title=basename,
                                     xlabel=load_column,
                                     ylabel=throughput_column)
        graph.render_graphs(graphs, basename, output_directory_)
        usl.summarize(model_fit)
    else:
        raise UnsupportedModelException()
    return 1


if __name__ == '__main__':
    sys.exit(main())
