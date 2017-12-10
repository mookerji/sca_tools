# pylint: disable=no-value-for-parameter

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
"""
CLI entrypoint for sca_agg, a tool for aggregating benchmarking
trials from independent load/concurrency measurements.
"""

import logging
import os
import sys

import click

import sca_tools.datasets as dat
import sca_tools.file_utils as futils

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))


@click.command()
@click.option('--load_column', default='load')
@click.option('--throughput_column', default='throughput')
@click.option('--throughput_errors_column', default='throughput_stddev')
@click.option('--output_directory', default=None)
@click.option('--clean', default=True)
@click.argument('filenames', nargs=-1)
def main(load_column, throughput_column, throughput_errors_column,
         output_directory, clean, filenames):
    """sca_agg merges results from independent benchmarking measurements
    into a single CSV file for use in sca_fit. It assumes (and checks)
    that a single CSV file contains multiple trial measurements of
    throughput for a fixed load or concurrency and adds a row of load,
    measured throughput mean, and measured throughput standard
    deviation to the output.

    Some examples!

    Let's say you've taken a CPU benchmark for an application at 1,
    16, 128, and 1024 systems threads. Each file of benchmark results
    can contains multiple samples of throughput for a particular
    concurrency.

    >>> ls *.csv

    cpu_1024_120_sec.csv
    cpu_128_120_sec.csv
    ...
    cpu_1_120_sec.csv

    Each file contains multiple samples for a fixed concurrency:

    >>> cat cpu_1024_120_sec.csv

    ,latency,quantile,threads,throughput
    0,68.05,99,1024,33058.0
    1,244.38,99,1024,32933.6
    2,612.21,99,1024,27480.81
    3,694.45,99,1024,27443.04
    4,1280.93,99,1024,27595.84
    5,1069.86,99,1024,27573.87
    6,1050.76,99,1024,27597.22
    ...
    119,1050.76,99,1024,27525.99

    You can pass all the measurement files to sca_agg to aggregate.

    >>> find . -type f -name 'cpu_*.csv' \
      |  xargs python sca_tools/sca_agg.py --load_column threads \
          --throughput_column throughput
          --throughput_errors_column throughput_stddev

    The result will look like:

    >>> cat aggregated.csv

    ,threads,throughput,throughput_stddev
    6,1.0,8272.55783333,90.5286256646
    0,2.0,16589.7894915,277.689391919
    9,4.0,27322.0743333,1076.90640997
    ...
    5,1024.0,27346.7111667,1010.17893612
    1,2048.0,27411.8171667,1147.08779945

    The trials for each load measurement is summarized by a normal
    distribution.

    """
    frames = []
    for filename in filenames:
        logging.debug('Reading in %s', filename)
        df_spec = dat.read_frame(filename)
        df_spec.load_col = load_column
        df_spec.throughput_col = throughput_column
        if not df_spec.has_fixed_load:
            logging.warn(
                'Skipping %s. load.stddev=%f',
                filename,
                df_spec.load.std(),
            )
            continue
        if clean:
            if df_spec.drop_outliers():
                logging.warn('Dropped some throughput outliers!')
        frames.append(df_spec)
    result = dat.aggregate_frames(frames, load_column, throughput_column,
                                  throughput_errors_column)
    if not output_directory:
        output_directory_ = futils.get_file_directory(filenames[0])
    else:
        output_directory_ = output_directory
    outfile = os.path.join(output_directory_, 'aggregated.csv')
    logging.info('Logging to %s', outfile)
    result.to_csv(outfile)
    return 1


if __name__ == '__main__':
    sys.exit(main())
