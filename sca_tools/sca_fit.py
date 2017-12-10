# pylint: disable=no-value-for-parameter,too-many-arguments

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
"""CLI entrypoint for sca_fit, a tool for fitting analyzing
benchmarking trial data.

"""

import sys

import click

import sca_tools.datasets as dat
import sca_tools.graphing as graph
import sca_tools.file_utils as futils
import sca_tools.usl as usl


class UnsupportedModelException(Exception):
    """The requested model type is not supported by sca_fit."""


@click.command()
@click.option('--load_column', default='load')
@click.option('--throughput_column', default='throughput')
@click.option('--throughput_errors_column', default='throughput_stddev')
@click.option('--model_type', default='usl')
@click.option('--output_directory', default=None)
@click.argument('filename')
def main(load_column, throughput_column, throughput_errors_column, model_type,
         output_directory, filename):
    """sca_fit fits a scalability model to a load-throughput benchmarking
    data.

    Some examples!

    Let's say we have a CSV file from a CPU throughput benchmark.

    >>> cat aggregated.csv

    ,threads,throughput,throughput_stddev
    6,1.0,8272.55783333,90.5286256646
    0,2.0,16589.7894915,277.689391919
    9,4.0,27322.0743333,1076.90640997
    ...
    5,1024.0,27346.7111667,1010.17893612
    1,2048.0,27411.8171667,1147.08779945

    We can then analyze it with:

    >>> python sca_tools/sca_fit.py --model_type usl \
        --load_column threads --throughput_column throughput \
        aggregated.csv

    """
    df_spec = dat.read_frame(filename)
    df_spec.load_col = load_column
    df_spec.throughput_col = throughput_column
    df_spec.error_col = throughput_errors_column
    if not output_directory:
        output_directory_ = futils.get_file_directory(filename)
    else:
        output_directory_ = output_directory
    basename = futils.get_file_basename(filename)
    if model_type == 'usl':
        model = usl.USLModel()
        if df_spec.error is not None:
            weights = df_spec.weights
        else:
            weights = None
        # TODO: Don't pass weights to nlopt fit until the results are
        # within the realm of my understanding.
        model_fit = model.fit(data=df_spec.throughput.values,
                              load=df_spec.load.values, lambda_=1000,
                              sigma_=0.1, kappa=0.001, weights=weights)
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
