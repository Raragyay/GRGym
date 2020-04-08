import itertools
from pathlib import Path
from typing import Any, Callable, Iterator, List, TextIO, Tuple

import numpy as np
import pytest


def retrieve_file_tests(
        input_func: Callable[[str], Any] = lambda f: f,
        expected_func: Callable[[str], Any] = lambda f: f,
        id_func: Callable[[str, int, Any], str] = lambda name, id, exoected: id,
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
        for X, y in retrieve_x_y(file_path):
            processed_X = input_func(X)
            processed_y = expected_func(y)
            test_data.append(
                pytest.param(processed_X, processed_y, id=id_func(processed_X, processed_y, next(test_num))))
    return test_data


def retrieve_x_y(file_path: Path) -> Iterator[Tuple[str, str]]:
    for raw_test in retrieve_raw_test_data(file_path):
        yield raw_test.rsplit('\n', 1)


def retrieve_raw_test_data(file_path: Path) -> Iterator[str]:
    TEST_DATA_ENDING = "==END TEST=="
    return map(lambda s: s[:-1], file_path.open().read().split(TEST_DATA_ENDING)[:-1])  # Last item is empty


if __name__ == '__main__':
    test_data_file_names = Path(__file__).parent.glob("|".join(["deadwood/tm_any.txt"]))
    for file_name in test_data_file_names:
        print(list(retrieve_raw_test_data(file_name)))


def retrieve_int_vector(string: str):
    return np.fromstring(string, np.int8, sep=" ")


def retrieve_boolean(file: TextIO):
    s = file.readline().strip()
    if s == "Y":
        return True
    elif s == "N":
        return False
    else:
        raise ValueError(f"Invalid can_knock value: {s}")


def nonzero_indices(string: str):
    return convert_matrix(string).flatten().nonzero()[0]


def convert_matrix(string: str):
    return np.fromstring(string, np.int8, sep=' ')


def single_int(string: str):
    return int(string)


def idfn_id_expected(test_name, test_id, expected):
    return f"{test_id}-{expected}"


def idfn_name_id(test_name, test_id, expected):
    return f"{test_name}.{test_id}"


def idfn_name_id_expected(test_name, test_num, expected):
    return f"{test_name}.{test_num}-{expected}"
