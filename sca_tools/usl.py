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


def _unpack_params(usl_fit):
    keys = ['lambda_', 'sigma_', 'kappa']
    return {k: ufloat(usl_fit.params[k].value, usl_fit.params[k].stderr) for k in keys}


def plot_loglog_model(xvalues, yvalues, title, xlabel, ylabel):
    plt.figure()
    ax = plt.loglog(xvalues, yvalues)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return ax


def _usl_func(load, lambda_, sigma_, kappa):
    return lambda_ * load / (1 + sigma_ * (load - 1) + kappa * load *
                             (load - 1))


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


def _calc_latency_from_queue_size(q_size, lambda_, sigma_, kappa):
    return (1 + sigma_ * (q_size - 1) + kappa * q_size * (q_size - 1)) / lambda_


class LatencyQueueSizeModel(lmfit.Model):
    def __init__(self, independent_vars=['q_size'], prefix='',
                 nan_policy='raise', **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(LatencyQueueSizeModel,
              self).__init__(_calc_latency_from_queue_size, **kwargs)


def _calc_latency_from_throughput(tput, lambda_, sigma_, kappa):
    norm = tput**2 * (kappa ** 2 \
                          + 2 * kappa * (sigma_ - 2) + sigma_ ** 2) \
                          + 2 * lambda_ * tput * (kappa - sigma_) \
                          + lambda_**2
    return (-np.sqrt(norm) + kappa * tput \
        + lambda_ - sigma_ * tput) / \
        (2 * kappa * tput**2)


class LatencyThroughputModel(lmfit.Model):
    def __init__(self, independent_vars=['tput'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(LatencyThroughputModel,
              self).__init__(_calc_latency_from_throughput, **kwargs)


def _calc_throughput_from_latency(latency, lambda_, sigma_, kappa):
    norm = sigma_ ** 2 + kappa ** 2 \
           + 2 * kappa * (2 * lambda_ * latency + sigma_ - 2) \
           - kappa + sigma_
    return np.sqrt(norm) / (2 * kappa * latency)


class ThroughputLatencyModel(lmfit.Model):
    def __init__(self, independent_vars=['latency'], prefix='',
                 nan_policy='raise', **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(ThroughputLatencyModel,
              self).__init__(_calc_throughput_from_latency, **kwargs)


def _calc_queue_size_from_latency(latency, lambda_, sigma_, kappa):
    norm = sigma_ ** 2 + kappa ** 2 \
           + 2 * kappa * (2 * lambda_ * latency + sigma_ - 2)
    return (kappa - sigma_ + np.sqrt(norm)) / (2 * kappa)


class QueueSizeLatencyModel(lmfit.Model):
    def __init__(self, independent_vars=['latency'], prefix='',
                 nan_policy='raise', **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(QueueSizeLatencyModel,
              self).__init__(_calc_queue_size_from_latency, **kwargs)


def calc_throughput(usl_fit, load):
    return usl_fit.eval(load=load)


def plot_throughput(usl_fit, load, throughput, title, xlabel, ylabel):
    confidence_band = usl_fit.eval_uncertainty(load, sigma=2)
    inputs = np.arange(load.min(), load.max(), 1)
    best_fit_ = usl_fit.eval(load=inputs)
    plt.figure()
    ax = plt.fill_between(load, usl_fit.best_fit - confidence_band,
                          usl_fit.best_fit + confidence_band, color='#888888',
                          alpha=0.5)
    plt.plot(inputs, best_fit_)
    plt.scatter(load, throughput)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return ax


def max_load(usl_fit):
    p = _unpack_params(usl_fit)
    return sqrt((1 - p['sigma_']) / p['kappa'])


def max_throughput(usl_fit):
    return calc_throughput(usl_fit, load=max_load(usl_fit))


def _get_mean(series):
    return series.apply(lambda x: x.n).values


def _get_stddev(series):
    return series.apply(lambda x: x.s).values


def plot_overhead(usl_fit, load):
    p = _unpack_params(usl_fit)
    inputs = load[load > 1]
    plt.figure()
    fig, ax = plt.subplots()
    # Plot ideal
    b0 = plt.bar(inputs.index, 1 / inputs)
    # Plot contention
    contention = p['sigma_'] * (1 - 1 / inputs)
    b1 = plt.bar(inputs.index,
                 _get_mean(contention), yerr=_get_stddev(contention))
    # Plot coherence
    coherence = 0.5 * p['kappa'] * (inputs - 1)
    b2 = plt.bar(inputs.index, _get_mean(coherence),
                 yerr=_get_stddev(coherence))
    plt.xticks(inputs.index, inputs.values)
    plt.legend((b0, b1, b2), ('Ideal', 'Contential', 'Coherence'))
    plt.title('Execution Overhead')
    plt.xlabel('Load')
    plt.ylabel('Efficiency (fraction)')
    return ax


def plot_efficiency(usl_fit, load, throughput):
    p = _unpack_params(usl_fit)
    plt.figure()
    fig, ax = plt.subplots()
    linear_throughput = _usl_func(1, p['lambda_'], p['sigma_'],
                                  p['kappa']) * load
    efficiency = throughput / linear_throughput
    plt.bar(load.index, _get_mean(efficiency), yerr=_get_stddev(efficiency))
    plt.xticks(load.index, load.values)
    plt.title('Measured Execution Efficiency')
    plt.xlabel('Load')
    plt.ylabel('Efficiency (Fraction)')
    return ax


def plot_models(usl_fit):
    result = []
    # Calculate and plot latency as a function of assumed queue size
    xvalues = np.arange(1, 5000, 1)
    yvalues = LatencyQueueSizeModel().eval(usl_fit.params, q_size=xvalues)
    artifact = plot_loglog_model(
        xvalues,
        yvalues,
        title='Latency vs Queue Depth',
        xlabel='Queue Depth',
        ylabel='Latency (sec)',
    )
    result.append(graph.GraphResult('latency-from-queue', artifact))
    # Calculate and plot latency as a function of assumed throughput
    xvalues = np.arange(1, 10000, 1)
    yvalues = LatencyThroughputModel().eval(usl_fit.params, tput=xvalues)
    artifact = plot_loglog_model(
        xvalues,
        yvalues,
        title='Latency vs Throughput',
        xlabel='Throughput',
        ylabel='Latency',
    )
    result.append(graph.GraphResult('latency-from-throughput', artifact))
    # Calculate and plot throughput as a function of assumed latency
    xvalues = np.arange(0.001, 100, 0.001)
    yvalues = ThroughputLatencyModel().eval(usl_fit.params, latency=xvalues)
    artifact = plot_loglog_model(
        xvalues,
        yvalues,
        title='Throughput vs Latency',
        xlabel='Latency (sec)',
        ylabel='Throughput',
    )
    result.append(graph.GraphResult('throughput-from-latency', artifact))
    # Calculate and plot queue-size as a function of latency
    xvalues = np.arange(0.001, 100, 0.001)
    yvalues = QueueSizeLatencyModel().eval(usl_fit.params, latency=xvalues)
    artifact = plot_loglog_model(
        xvalues,
        yvalues,
        title='Queue Depth vs Latency',
        xlabel='Latency (sec)',
        ylabel='Queue Depth',
    )
    result.append(graph.GraphResult('queue-size-from-latency', artifact))
    return result


def summarize(usl_fit):
    print
    print '----- Summary -----'
    print
    print usl_fit.fit_report()


def generate_graphs(model_fit, data, title, xlabel, ylabel):
    result = []
    all_throughput = plot_throughput(model_fit, data.load, data.throughput,
                                     title, xlabel, ylabel)
    result.append(graph.GraphResult(name='throughput_model',
                                    artifact=all_throughput))
    efficiency = plot_efficiency(model_fit, data.load, data.throughput)
    result.append(graph.GraphResult(name='efficiency_model',
                                    artifact=efficiency))
    overhead = plot_overhead(model_fit, data.load)
    result.append(graph.GraphResult(name='overhead_model', artifact=overhead))
    return result + plot_models(model_fit)
