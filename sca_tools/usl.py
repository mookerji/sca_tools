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

from uncertainties import ufloat
from uncertainties.umath import *

import lmfit
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import sca_tools.graphing as graph

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

## USL-specific function definitions


def _usl_func(load, lambda_, sigma_, kappa):
    return lambda_ * load / (1 + sigma_ * (load - 1) + kappa * load *
                             (load - 1))


def _calc_latency_from_queue_size(q_size, lambda_, sigma_, kappa):
    return (1 + sigma_ * (q_size - 1) + kappa * q_size * (q_size - 1)) / lambda_


def _calc_latency_from_throughput(tput, lambda_, sigma_, kappa):
    norm = tput**2 (kappa ** 2 \
                          + 2 * kappa (sigma_ - 2) + sigma_ ** 2) \
                          + 2 * lambda_ * tput (kappa - sigma_) \
                          + lambda_**2
    return (-np.sqrt(norm) + kappa * tput \
        + lambda_ - sigma_ * tput) / \
        (2 * kappa * tput**2)


def _calc_throughput_from_latency(latency, lambda_, sigma_, kappa):
    norm = sigma_ ** 2 + kappa ** 2 \
           + 2 * kappa * (2 * lambda_ * latency + sigma_ - 2) \
           - kappa + sigma_
    return np.sqrt(norm) / (2 * kappa * latency)


def _calc_queue_size_from_latency(latency, lambda_, sigma_, kappa):
    norm = sigma_ ** 2 + kappa ** 2 \
           + 2 * kappa * (2 * lambda_ * latency + sigma_ - 2)
    return (kappa - sigma_ + np.sqrt(norm)) / (2 * kappa)


class USLModel(lmfit.Model):

    def __init__(self, independent_vars=['load'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(USLModel, self).__init__(_usl_func, **kwargs)

    def guess(self, **kwargs):
        raise NotImplementedError()


def calc_throughput(usl_fit, load):
    return usl_fit.eval(load=load)


def plot_throughput(usl_fit, load, throughput):
    confidence_band = usl_fit.eval_uncertainty(load, sigma=2)
    inputs = np.arange(load.min(), load.max(), 1)
    best_fit_ = usl_fit.eval(load=inputs)
    ax = plt.fill_between(load, usl_fit.best_fit - confidence_band,
                          usl_fit.best_fit + confidence_band, color='#888888',
                          alpha=0.5)
    plt.plot(inputs, best_fit_)
    plt.scatter(load, throughput)
    return ax


def _unpack_params(usl_fit):
    keys = ['lambda_', 'sigma_', 'kappa']
    return {k: ufloat(usl_fit.params[k].value, usl_fit.params[k].stderr) for k in keys}


def max_load(usl_fit):
    p = _unpack_params(usl_fit)
    return sqrt((1 - p['sigma_']) / p['kappa'])


def max_throughput(usl_fit):
    return calc_throughput(usl_fit, load=max_load(usl_fit))


def calc_efficiency(usl_fit):
    raise NotImplementedError()


def plot_efficiency(usl_fit):
    raise NotImplementedError()


def summarize(usl_fit):
    print
    print '----- Summary -----'
    print
    print usl_fit.fit_report()


def generate_graphs(model_fit, data):
    result = []
    all_throughput = plot_throughput(model_fit, data.load, data.throughput)
    result.append(graph.GraphResult(name='throughput_model',
                                    artifact=all_throughput))
    return result
