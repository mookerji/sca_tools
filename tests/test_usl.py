# pylint: disable=protected-access,redefined-outer-name,missing-docstring

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

import numpy as np
import pytest

import sca_tools.usl as usl

DEFAULT_PARAMS = {'lambda_': 90, 'sigma_': 0.03, 'kappa': 0.0001}


def test_usl_func_eval():

    assert np.isclose(usl._usl_func(load=20, **DEFAULT_PARAMS), 1119.40298507)
    input_loads = np.arange(1, 100, 20)
    ans = np.array([
        90.,
        1151.03532278,
        1560.91370558,
        1734.04927353,
        1800.88932806,
    ])
    assert np.allclose(usl._usl_func(input_loads, **DEFAULT_PARAMS), ans)


@pytest.fixture
def usl_func_fit(default_data):
    model = usl.USLModel()
    model_fit = model.fit(data=default_data.throughput.values,
                          load=default_data.load.values, lambda_=1000,
                          sigma_=0.1, kappa=0.001)
    return model_fit


def test_usl_func_fit(usl_func_fit):
    assert np.isclose(usl_func_fit.params['lambda_'].value, 89.995)
    assert np.isclose(usl_func_fit.params['lambda_'].stderr, 14.2129)
    assert np.isclose(usl_func_fit.params['sigma_'].value, 0.0277286)
    assert np.isclose(usl_func_fit.params['sigma_'].stderr, 0.00912139)
    assert np.isclose(usl_func_fit.params['kappa'].value, 0.000104365)
    assert np.isclose(usl_func_fit.params['kappa'].stderr, 1.9875e-05)


def test_latency_from_queue(usl_func_fit):
    assert np.isclose(
        usl._calc_latency_from_queue_size(q_size=10, **DEFAULT_PARAMS),
        0.0142111111111,
    )
    model = usl.LatencyQueueSizeModel()
    assert np.isclose(model.eval(q_size=10, **DEFAULT_PARAMS), 0.0142111111111)
    assert np.isclose(
        model.eval(usl_func_fit.params, q_size=10),
        0.0139890407887,
    )


def test_latency_from_throughput(usl_func_fit):
    assert np.isclose(
        usl._calc_latency_from_throughput(tput=10, **DEFAULT_PARAMS),
        0.0108137163393,
    )
    model = usl.LatencyThroughputModel()
    assert np.isclose(model.eval(tput=10, **DEFAULT_PARAMS), 0.0108137163393)
    assert np.isclose(model.eval(usl_func_fit.params, tput=10), 0.0108368336576)


def test_throughput_from_latency(usl_func_fit):
    ans = usl._calc_throughput_from_latency(latency=10, **DEFAULT_PARAMS)
    assert np.isclose(ans, 312.413)
    model = usl.ThroughputLatencyModel()
    assert np.isclose(ans, model.eval(latency=10, **DEFAULT_PARAMS))
    assert np.isclose(
        model.eval(usl_func_fit.params, latency=10),
        304.390458968,
    )


def test_queue_size_from_latency(usl_func_fit):
    assert np.isclose(
        usl._calc_queue_size_from_latency(latency=20, **DEFAULT_PARAMS),
        4094.63127153,
    )
    model = usl.QueueSizeLatencyModel()
    assert np.isclose(model.eval(latency=20, **DEFAULT_PARAMS), 4094.631271532)
    assert np.isclose(
        model.eval(usl_func_fit.params, latency=20),
        4021.50102647,
    )
    # TODO: Evaluate for uncertainty arrays
    # print model.eval(usl_func_fit.params, latency=ufloat(20,1))
    ans = np.array([
        800.66746176,
        4124.03239117,
        5814.33318714,
        7120.87591603,
        8225.61201396,
    ])
    modeled = model.eval(usl_func_fit.params, latency=np.arange(1, 100, 20))
    assert np.allclose(modeled, ans)
