import argparse
import os.path
import pathlib

import pytest


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


@pytest.fixture
def path_data() -> str:
    return os.path.join(str(pathlib.Path(__file__).parent.resolve()), "test_data")


@pytest.fixture
def path_examples() -> str:
    return os.path.join(
        str(pathlib.Path(__file__).parent.parent.resolve()),
        "examples",
        "pocket_list_test",
    )


@pytest.fixture
def path_epocs_run_script() -> str:
    return os.path.join(
        str(pathlib.Path(__file__).parent.parent.resolve()), "run_epocs.py"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--esm_parameters_path",
        default="esm2_t36_3B_UR50D.pt",
        help="Path for the ESM parameters or the name of the ESM model of interest.",
    )
    parser.addoption(
        "--use_gpu",
        help="Use GPU for ESM generation. Default: True",
        default=True,
        type=str2bool,
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    esm_parameters_path = metafunc.config.option.esm_parameters_path
    use_gpu = metafunc.config.option.use_gpu
    for name, value in [
        ["esm_parameters_path", esm_parameters_path],
        ["use_gpu", use_gpu],
    ]:
        if name in metafunc.fixturenames and value is not None:
            metafunc.parametrize(name, [value])
