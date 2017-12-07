#! /usr/bin/env python

"""Parses the sysbench CPU benchmark as seen in example.txt. Extracts
output into a CSV format for sca_fit.

Usage:
  cat example.txt | python parse_sysbench.py
  python parse_sysbench.py example.txt

"""

import fileinput
import parse
import pandas as pd
import sys

LINE_UNKNOWN = 'unknown'
LINE_STARTED = "Threads started!"
LINE_DATA = "[ {:d}s ] thds: {:d} eps: {:f} lat ({},{:d}%): {:f}"


def result_to_dict(result):
    """Produces a dict from parse.Result.

    """
    # (30, 4096, 4597.7, 'ms', 99, 4683.57)
    assert result[3] == 'ms'
    return {
        'threads': result[1],
        'throughput': result[2],
        'quantile': result[4],
        'latency': result[5],
    }


def main():
    state = LINE_UNKNOWN
    trials = []
    for line in fileinput.input():
        if parse.parse(LINE_STARTED, line) and state == LINE_UNKNOWN:
            state = LINE_DATA
        elif state == LINE_DATA and parse.parse(LINE_DATA, line):
            trials.append(result_to_dict(parse.parse(LINE_DATA, line)))
        else:
            pass
    # Silly Python! fileinput.isstdin() doesn't seem to actually work
    # here.
    if '<stdin>' not in fileinput.filename():
        output_filename = '%s.csv' % fileinput.filename()
    else:
        output_filename = 'trial.csv'
    print 'Output to: %s' % output_filename
    pd.DataFrame(trials).to_csv(output_filename)


if __name__ == '__main__':
    sys.exit(main())
