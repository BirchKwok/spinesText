import os
from functools import wraps


FILE_PATH = os.path.dirname(__file__)


def _give_it_name(func, name):
    @wraps(func)
    def wrap():
        return func(name)
    return wrap


class _BuiltInDataSetsBase:
    def __init__(self, file_name):
        self._FILEPATH = os.path.join(FILE_PATH, './built-in-datasets/', file_name)
        assert os.path.exists(self._FILEPATH)
        with open(self._FILEPATH, 'r', encoding='utf-8') as f:
            self._d = f.read()

    @property
    def data(self):
        return self._d


LoadSDYXZ = _give_it_name(_BuiltInDataSetsBase, name='shediao.txt')
