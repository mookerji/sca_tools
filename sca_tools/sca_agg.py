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
import logging
import os
import sys

import sca_tools.datasets as dat
import sca_tools.file_utils as futils

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


@click.command()
@click.option('--load_column', default='load')
@click.option('--throughput_column', default='throughput')
@click.option('--throughput_errors_column', default='throughput_stddev')
@click.option('--output_directory', default=None)
@click.argument('filenames', nargs=-1)
def main(load_column, throughput_column, throughput_errors_column,
         output_directory, filenames):
    dfs = []
    for filename in filenames:
        logging.debug('Reading in %s' % filename)
        df_spec = dat.read_frame(filename)
        df_spec._load_col = load_column
        df_spec._tput_col = throughput_column
        if not df_spec.has_fixed_load:
            logging.warn('Skipping %s. load.stddev=%f' \
                         % (filename, df_spec.load.std()))
            continue
        dfs.append(df_spec)
    result = dat.aggregate_frames(dfs, load_column, throughput_column,
                                  throughput_errors_column)
    if not output_directory:
        output_directory_ = futils.get_file_directory(filenames[0])
    else:
        output_directory_ = output_directory
    outfile = os.path.join(output_directory_, 'aggregated.csv')
    logging.info('Logging to %s' % outfile)
    result.to_csv(outfile)
    return 1


if __name__ == '__main__':
    sys.exit(main())
