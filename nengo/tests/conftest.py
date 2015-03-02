import hashlib
import os
import re

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.neurons import Direct, LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.rc import rc
from nengo.simulator import Simulator as ReferenceSimulator
from nengo.utils.compat import ensure_bytes
from nengo.utils.testing import Analytics, Plotter

test_seed = 0  # changing this will change seeds for all tests


def pytest_configure(config):
    rc.reload_rc([])
    rc.set('decoder_cache', 'enabled', 'false')


@pytest.fixture(scope="session")
def Simulator(request):
    """the Simulator class being tested.

    Please use this, and not nengo.Simulator directly,
    unless the test is reference simulator specific.
    """
    return ReferenceSimulator


@pytest.fixture(scope="session")
def RefSimulator(request):
    """the reference simulator.

    Please use this if the test is reference simulator specific.
    Other simulators may choose to implement the same API as the
    reference simulator; this allows them to test easily.
    """
    return ReferenceSimulator


def construct_recorder_dirname(request, name):
    simulator, nl = ReferenceSimulator, None
    if 'Simulator' in request.funcargnames:
        simulator = request.getfuncargvalue('Simulator')
    if 'nl' in request.funcargnames:
        nl = request.getfuncargvalue('nl')
    elif 'nl_nodirect' in request.funcargnames:
        nl = request.getfuncargvalue('nl_nodirect')

    dirname = "%s.%s" % (simulator.__module__, name)
    if nl is not None:
        dirname = os.path.join(dirname, nl.__name__)
    return dirname


def activate_recorder(cls, request, name):
    record = request.config.getvalue(name)
    if record is not True and record is not False:
        dirname = record
        record = True
    else:
        dirname = construct_recorder_dirname(request, name)

    recorder = cls(
        dirname, request.module.__name__, request.function.__name__,
        record=record)
    request.addfinalizer(lambda: recorder.__exit__(None, None, None))
    return recorder.__enter__()


@pytest.fixture
def plt(request):
    """a pyplot-compatible plotting interface.

    Please use this if your test creates plots.

    This will keep saved plots organized in a simulator-specific folder,
    with an automatically generated name. savefig() and close() will
    automatically be called when the test function completes.

    If you need to override the default filename, set `plt.saveas` to
    the desired filename.
    """
    return activate_recorder(Plotter, request, 'plots')


@pytest.fixture
def analytics(request):
    return activate_recorder(Analytics, request, 'analytics')


@pytest.fixture
def analytics_data(request):
    paths = request.config.getvalue('compare')
    function_name = re.sub(
        '^test_[a-zA-Z0-9]*_', 'test_', request.function.__name__, count=1)
    return [Analytics.load(
        p, request.module.__name__, function_name) for p in paths]


def function_seed(function, mod=0):
    c = function.__code__

    # get function file path relative to Nengo directory root
    nengo_path = os.path.abspath(os.path.dirname(nengo.__file__))
    path = os.path.relpath(c.co_filename, start=nengo_path)

    # take start of md5 hash of function file and name, should be unique
    hash_list = os.path.normpath(path).split(os.path.sep) + [c.co_name]
    hash_string = ensure_bytes('/'.join(hash_list))
    i = int(hashlib.md5(hash_string).hexdigest()[:15], 16)
    return (i + mod) % npext.maxint


@pytest.fixture
def rng(request):
    """a seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    # add 1 to seed to be different from `seed` fixture
    seed = function_seed(request.function, mod=test_seed + 1)
    return np.random.RandomState(seed)


@pytest.fixture
def seed(request):
    """a seed for seeding Networks.

    This should be used in lieu of an integer seed so that we can ensure that
    tests are not dependent on specific seeds.
    """
    return function_seed(request.function, mod=test_seed)


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl", [Direct, LIF, LIFRate, RectifiedLinear, Sigmoid])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl_nodirect", [LIF, LIFRate, RectifiedLinear, Sigmoid])


def pytest_addoption(parser):
    parser.addoption(
        '--plots', nargs='?', default=False, const=True,
        help='Save plots (optional with directory to save them in).')
    parser.addoption(
        '--analytics', nargs='?', default=False, const=True,
        help='Save analytics (optional with directory to save the data in).')
    parser.addoption('--compare', nargs=2, help='Compare analytics results.')
    parser.addoption('--noexamples', action='store_false', default=True,
                     help='Do not run examples')
    parser.addoption(
        '--slow', action='store_true', default=False,
        help='Also run slow tests.')


def pytest_runtest_setup(item):
    if (item.config.getvalue('compare') and
            not getattr(item.obj, 'compare', None)):
        return

    for mark, option, message in [
            ('example', 'noexamples', "examples not requested"),
            ('slow', 'slow', "slow tests not requested")]:
        if getattr(item.obj, mark, None) and not item.config.getvalue(option):
            pytest.skip(message)

    if getattr(item.obj, 'noassertions', None):
        skip = True
        skipreasons = []
        for fixture_name, option, message in [
                ('analytics', 'analytics', "analytics not requested"),
                ('plt', 'plots', "plots not requested")]:
            if fixture_name in item.fixturenames:
                if item.config.getvalue(option):
                    skip = False
                else:
                    skipreasons.append(message)
        if skip:
            pytest.skip(" and ".join(skipreasons))


def pytest_collection_modifyitems(session, config, items):
    compare = config.getvalue('compare') is None
    for item in list(items):
        if (getattr(item.obj, 'compare', None) is None) != compare:
            items.remove(item)


def pytest_terminal_summary(terminalreporter):
    reports = terminalreporter.getreports('passed')
    if not reports or terminalreporter.config.getvalue('compare') is None:
        return
    terminalreporter.write_sep("=", "PASSED")
    for rep in reports:
        for name, content in rep.sections:
            terminalreporter.writer.sep("-", name)
            terminalreporter.writer.line(content)
