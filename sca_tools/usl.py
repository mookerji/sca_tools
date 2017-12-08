# pylint: disable=no-name-in-module,dangerous-default-value

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
"""Module implementing the Universal Scalability Model (USL), as well
as some derived models that are computed from the USL's free
parameters. We rely heavily on the lmfit and uncertainties packages
for performing nonlinear fitting, presenting results, and propagating
model uncertanties.

"""

import logging
import os

from uncertainties import ufloat
from uncertainties.umath import sqrt

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sca_tools.graphing as graph

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))


def _unpack_params(usl_fit):
    """Unpacks parameters from USL model with covariance/uncertainty
    parameters.

    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result

    Returns
    -------
    dict
        Contains lambda_, sigma_, kappa uncertainty.ufloat parameters.

    """
    keys = ['lambda_', 'sigma_', 'kappa']

    def to_ufloat(k):
        return ufloat(usl_fit.params[k].value, usl_fit.params[k].stderr)

    return {k: to_ufloat(k) for k in keys}


def plot_loglog_model(xvalues, yvalues, title, xlabel, ylabel):
    """utility function for plotting a log-log plot.

    Parameters
    ----------
    xvalues : array-like of floats

    yvalues : array-like of floats

    title : str

    xlabel : str

    ylabel : str


    Returns
    -------
    Matplotlib Axes
    """
    plt.figure()
    axis = plt.loglog(xvalues, yvalues)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return axis


def _usl_func(load, lambda_, sigma_, kappa):
    """USL implementation of a load-throughput model.

    Parameters
    ----------
    load : float, array-like
        Assumed load
    lambda_ : float, uncertainties.ufloat
        Unit throughput coefficient
    sigma_ : float, uncertainties.ufloat
        Serialization coefficient
    kappa_ : float, uncertainties.ufloat
        Crosstalk coefficient

    Returns
    -------
    float or array-like
        Throughput
    """
    return lambda_ * load / (1 + sigma_ * (load - 1) + kappa * load *
                             (load - 1))


class USLModel(lmfit.Model):
    """
    lmfit implementation of the Universal Scalability Law (USL).

    Parameters
    ----------
    independent_vars: [str]
        Arguments to func that are independent variables.
    prefix: string, optional
       String to prepend to parameter names
    missing:  str or None, optional
        How to handle NaN and missing values in data. One of:
        - 'none' or None: Do not check for null or missing values (default).
        - 'drop': Drop null or missing observations in data. if pandas is
          installed, `pandas.isnull` is used, otherwise `numpy.isnan` is used.
        - 'raise': Raise a (more helpful) exception when data contains null
          or missing values.
    **kwargs : optional
        Keyword arguments to pass to :class:`Model`.

    Returns
    -------
    USLModel
    """

    def __init__(self, independent_vars=['load'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(USLModel, self).__init__(_usl_func, **kwargs)

    def guess(self, data, **kwargs):
        """Not implemented."""
        raise NotImplementedError()

    def copy(self, **kwargs):
        """Not implemented."""
        raise NotImplementedError()


def _calc_latency_from_queue_size(q_size, lambda_, sigma_, kappa):
    """Calculates latency (sec) from a set of assumed queue sizes.

    Parameters
    ----------
    q_size : float, array-like
        Assumed queue size
    lambda_ : float, uncertainties.ufloat
        Unit throughput coefficient
    sigma_ : float, uncertainties.ufloat
        Serialization coefficient
    kappa_ : float, uncertainties.ufloat
        Crosstalk coefficient

    Returns
    -------
    float or array-like
        Latency (sec)
    """
    return (1 + sigma_ * (q_size - 1) + kappa * q_size * (q_size - 1)) / lambda_


class LatencyQueueSizeModel(lmfit.Model):
    """lmfit implementation of latency as a function of assumed queue size
    (derived via the USL and Little's Law).

    Parameters
    ----------
    Identical to USLModel.

    Returns
    -------
    LatencyQueueSizeModel
    """

    def __init__(self, independent_vars=['q_size'], prefix='',
                 nan_policy='raise', **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(LatencyQueueSizeModel,
              self).__init__(_calc_latency_from_queue_size, **kwargs)

    def copy(self, **kwargs):
        raise NotImplementedError()


def _calc_latency_from_throughput(tput, lambda_, sigma_, kappa):
    """Calculates latency (sec) from a set of assumed throughput.

    Parameters
    ----------
    t_put : float, array-like
        Assumed throughput
    lambda_ : float, uncertainties.ufloat
        Unit throughput coefficient
    sigma_ : float, uncertainties.ufloat
        Serialization coefficient
    kappa_ : float, uncertainties.ufloat
        Crosstalk coefficient

    Returns
    -------
    float or array-like
        Latency (sec)
    """
    norm = tput**2 * (kappa ** 2 +
                      + 2 * kappa * (sigma_ - 2) + sigma_ ** 2) \
        + 2 * lambda_ * tput * (kappa - sigma_) \
        + lambda_**2
    return (-np.sqrt(norm) + kappa * tput
            + lambda_ - sigma_ * tput) / \
        (2 * kappa * tput**2)


class LatencyThroughputModel(lmfit.Model):
    """lmfit implementation of latency as a function of assumed throughput
    (derived via the USL and Little's Law).

    Parameters
    ----------
    Identical to USLModel.

    Returns
    -------
    LatencyThroughputModel
    """

    def __init__(self, independent_vars=['tput'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(LatencyThroughputModel,
              self).__init__(_calc_latency_from_throughput, **kwargs)

    def copy(self, **kwargs):
        raise NotImplementedError()


def _calc_throughput_from_latency(latency, lambda_, sigma_, kappa):
    """Calculates throughput from assumed latency (sec).

    Parameters
    ----------
    latency : float, array-like
        Assumed latency (sec)
    lambda_ : float, uncertainties.ufloat
        Unit throughput coefficient
    sigma_ : float, uncertainties.ufloat
        Serialization coefficient
    kappa_ : float, uncertainties.ufloat
        Crosstalk coefficient

    Returns
    -------
    float or array-like
        Throughput (Hz)
    """
    norm = sigma_ ** 2 + kappa ** 2 \
        + 2 * kappa * (2 * lambda_ * latency + sigma_ - 2) \
        - kappa + sigma_
    return np.sqrt(norm) / (2 * kappa * latency)


class ThroughputLatencyModel(lmfit.Model):
    """lmfit implementation of throughput as a function of assumed latency
    (derived via the USL and Little's Law).

    Parameters
    ----------
    Identical to USLModel.

    Returns
    -------
    ThroughputLatencyModel
    """

    def __init__(self, independent_vars=['latency'], prefix='',
                 nan_policy='raise', **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(ThroughputLatencyModel,
              self).__init__(_calc_throughput_from_latency, **kwargs)

    def copy(self, **kwargs):
        raise NotImplementedError()


def _calc_queue_size_from_latency(latency, lambda_, sigma_, kappa):
    """Calculates queue_size from assumed latency (sec).

    Parameters
    ----------
    latency : float, array-like
        Assumed latency (sec)
    lambda_ : float, uncertainties.ufloat
        Unit throughput coefficient
    sigma_ : float, uncertainties.ufloat
        Serialization coefficient
    kappa_ : float, uncertainties.ufloat
        Crosstalk coefficient

    Returns
    -------
    float or array-like
        Queue size
    """
    norm = sigma_ ** 2 + kappa ** 2 \
        + 2 * kappa * (2 * lambda_ * latency + sigma_ - 2)
    return (kappa - sigma_ + np.sqrt(norm)) / (2 * kappa)


class QueueSizeLatencyModel(lmfit.Model):
    """lmfit implementation of queue size as a function of assumed latency
    (derived via the USL and Little's Law).

    Parameters
    ----------
    Identical to USLModel.

    Returns
    -------
    QueueSizeLatencyModel
    """

    def __init__(self, independent_vars=['latency'], prefix='',
                 nan_policy='raise', **kwargs):
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars
        }, )
        super(QueueSizeLatencyModel,
              self).__init__(_calc_queue_size_from_latency, **kwargs)

    def copy(self, **kwargs):
        raise NotImplementedError()


def calc_throughput(usl_fit, load):
    """Uses a USL scalability model to compute throughput for an assumed
    load or concurrency.

    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result
    load : float, array-like
        Assumed load/concurrency

    Returns
    ----------
    throughput : float, array-like
        Array of throughput

    """
    return usl_fit.eval(load=load)


def plot_throughput(usl_fit, load, throughput, errors, title, xlabel, ylabel):
    """Utility function for plotting load/throughput data, its model fit,
    and model confidence interval.

    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result
    load : float, array-like
        Load/concurrency data
    throughput : float, array-like
        Throughput data
    errors : float, array-like
        Throughput error/stddev data
    title : str

    xlabel : str

    ylabel : str

    Returns
    ----------
    Matplotlib Axes

    """
    confidence_band = usl_fit.eval_uncertainty(load, sigma=2)
    inputs = np.arange(load.min(), load.max(), 1)
    best_fit_ = usl_fit.eval(load=inputs)
    plt.figure()
    axis = plt.fill_between(load, usl_fit.best_fit - confidence_band,
                            usl_fit.best_fit + confidence_band, color='#888888',
                            alpha=0.5)
    plt.plot(inputs, best_fit_)
    if isinstance(errors, pd.Series) and errors.any():
        plt.errorbar(load, throughput, yerr=errors, fmt='o', ecolor='b')
    else:
        plt.scatter(load, throughput)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return axis


def max_load(usl_fit):
    """What's the estimated maximum load in the benchmark?

    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result

    Returns
    ----------
    sca_tools.graphing.GraphResult
        Container of graph artifacts to render
    """
    params = _unpack_params(usl_fit)
    return sqrt((1 - params['sigma_']) / params['kappa'])


def max_throughput(usl_fit):
    """What's the estimated maximum throughput in the benchmark?

    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result

    Returns
    ----------
    sca_tools.graphing.GraphResult
        Container of graph artifacts to render
    """
    return calc_throughput(usl_fit, load=max_load(usl_fit))


def _get_mean(series):
    """Pulls means from an array of uncertainy.ufloat's.

    Parameters
    ----------
    series : pandas.Series
        Array of uncertainy.ufloat's

    Returns
    ----------
    np.array of means's

    """
    return series.apply(lambda x: x.n).values


def _get_stddev(series):
    """Pulls errors from an array of uncertainy.ufloat's.

    Parameters
    ----------
    series : pandas.Series
        Array of uncertainy.ufloat's

    Returns
    ----------
    np.array of stddev's
    """
    return series.apply(lambda x: x.s).values


def plot_overhead(usl_fit, load):
    """
    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result
    load : array-like
        Load/concurrency data

    Returns
    ----------
    sca_tools.graphing.GraphResult
        Container of graph artifacts to render
    """
    params = _unpack_params(usl_fit)
    inputs = load[load > 1]
    plt.figure()
    _, axis = plt.subplots()
    # Plot ideal
    bar0 = plt.bar(inputs.index, 1 / inputs)
    # Plot contention
    contention = params['sigma_'] * (1 - 1 / inputs)
    bar1 = plt.bar(inputs.index,
                   _get_mean(contention), yerr=_get_stddev(contention))
    # Plot coherence
    coherence = 0.5 * params['kappa'] * (inputs - 1)
    bar2 = plt.bar(inputs.index,
                   _get_mean(coherence), yerr=_get_stddev(coherence))
    plt.xticks(inputs.index, inputs.values)
    plt.legend((bar0, bar1, bar2), ('Ideal', 'Contential', 'Coherence'))
    plt.title('Execution Overhead')
    plt.xlabel('Load')
    plt.ylabel('Efficiency (fraction)')
    return axis


def plot_efficiency(usl_fit, load, throughput):
    """
    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result
    load : float, array-like
        Load/concurrency data
    throughput : float, array-like
        Throughput data

    Returns
    ----------
    matplotlib.Axes

    """
    params = _unpack_params(usl_fit)
    plt.figure()
    _, axis = plt.subplots()
    linear_throughput = _usl_func(1, params['lambda_'], params['sigma_'],
                                  params['kappa']) * load
    efficiency = throughput / linear_throughput
    plt.bar(load.index, _get_mean(efficiency), yerr=_get_stddev(efficiency))
    plt.xticks(load.index, load.values)
    plt.title('Measured Execution Efficiency')
    plt.xlabel('Load')
    plt.ylabel('Efficiency (Fraction)')
    return axis


def plot_models(usl_fit):
    """
    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result

    Returns
    ----------
    sca_tools.graphing.GraphResult
        Container of graph artifacts to render
    """
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
    """
    Produces a summary report of the fit.

    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result
    """
    print
    print '----- Summary -----'
    print
    print usl_fit.fit_report()


def generate_graphs(usl_fit, data, title, xlabel, ylabel):
    """
    Parameters
    ----------
    usl_fit : lmfit.model.ModelResult
        USL scalabilty model result
    data : sca_tools.datasets.Dataset

    title : str

    xlabel : str

    ylabel : str

    Returns
    -------
    [sca_tools.graphing.GraphResult]
        Set of graphs to render
    """
    result = []
    all_throughput = plot_throughput(usl_fit, data.load, data.throughput,
                                     data.error, title, xlabel, ylabel)
    result.append(graph.GraphResult(name='throughput_model',
                                    artifact=all_throughput))
    efficiency = plot_efficiency(usl_fit, data.load, data.throughput)
    result.append(graph.GraphResult(name='efficiency_model',
                                    artifact=efficiency))
    overhead = plot_overhead(usl_fit, data.load)
    result.append(graph.GraphResult(name='overhead_model', artifact=overhead))
    return result + plot_models(usl_fit)
