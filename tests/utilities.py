import itertools
from pathlib import Path
from typing import Any, Callable, Iterator, List, Tuple

import numpy as np
import pytest


def retrieve_file_tests(
        input_func: Callable[[str], Any] = lambda f: f,
        expected_func: Callable[[str], Any] = lambda f: f,
        id_func: Callable[[str, int, Any], str] = lambda name, expected, test_id: test_id,
        file_suffix: str = None, file_names: List[str] = None) -> List[pytest.param]:
    if file_names:
        test_data_file_names = Path(__file__).parent.glob("|".join(file_names))
    elif file_suffix:
        test_data_file_names = Path(__file__).parent.glob(f"{file_suffix}*.txt")
    else:
        raise ValueError("Please provide a file suffix or a list of files to parse.")
    test_data = []
    test_num = itertools.count(1)
    for file_path in test_data_file_names:
        test_name = file_path.stem
        for X, y in retrieve_x_y(file_path):
            processed_X = input_func(X)
            processed_y = expected_func(y)
            test_data.append(
                pytest.param(processed_X, processed_y, id=id_func(test_name, processed_y, next(test_num))))
    return test_data


def retrieve_x_y(file_path: Path) -> Iterator[Tuple[str, str]]:
    for raw_test in retrieve_raw_test_data(file_path):
        yield raw_test.rsplit('\n', 1)


def retrieve_raw_test_data(file_path: Path) -> Iterator[str]:
    TEST_DATA_ENDING = "\n==END TEST=="
    return file_path.open().read().split(TEST_DATA_ENDING)[:-1]  # Last item is empty


def retrieve_int_vector(string: str) -> np.ndarray:
    return retrieve_vector(string, np.int)


def retrieve_float_vector(string: str) -> np.ndarray:
    return retrieve_vector(string, np.float)


def retrieve_vector(string: str, data_type: type = None):
    return np.fromstring(string, sep=" ", dtype=data_type)


def retrieve_boolean(string: str) -> bool:
    if string == "Y":
        return True
    elif string == "N":
        return False
    else:
        raise ValueError(f"Invalid boolean value: {string}")


def retrieve_nonzero_indices(string: str) -> np.ndarray:
    return convert_int_matrix(string).flatten().nonzero()[0]


def convert_int_matrix(string: str) -> np.ndarray:
    # noinspection PyTypeChecker
    return np.loadtxt(iter(string.split('\n')), dtype=np.int8, delimiter=' ')


def retrieve_int(string: str) -> int:
    return int(string)


def idfn_id_expected(file_name, expected, test_id):
    return f"{test_id}-{expected}"


def idfn_name_id(file_name, expected, test_id):
    return f"{file_name}.{test_id}"


def idfn_name_id_expected(file_name, expected, test_num):
    return f"{file_name}.{test_num}-{expected}"
