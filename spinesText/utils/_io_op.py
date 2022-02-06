import datetime
import json
import os

import numpy as np
import psutil
from ._batch_op import iter_count
from typing import *


def json_rw(json_path: str, method: str = 'r', json_dict=None, encoding='utf-8',
            ensure_ascii: bool = False) -> Union[Dict, Iterator, None]:
    """
    :param json_path: json file path
    :param method: r for read method, w for write method, same as python open function method
    :param json_dict: dict-like object, only be useful in write method
    :param encoding: file encoding
    :param ensure_ascii: if ensure_ascii is true (the default), the output is guaranteed to have all
                        incoming non-ASCII characters escaped.
                        if ensure_ascii is false, these characters will be output as-is.
    :return: Union[Dict, iterator, None]
    """
    if 'r' in method:
        if 'b' in method:
            encoding = None
        try:
            with open(json_path, method, encoding=encoding) as f:
                _ = json.load(f)
            return _
        except FileNotFoundError:
            raise FileNotFoundError(f"No such file or directory:{json_path}")

    elif 'w' in method or 'a' in method:
        if 'b' in method:
            encoding = None
        if json_dict is None:
            raise ValueError("If method is 'w', the parameter json_dict must be not None.")

        with open(json_path, method, encoding=encoding) as f:
            json.dump(json_dict, f, ensure_ascii=ensure_ascii)
        return


def return_useful_mem():
    mem = psutil.virtual_memory()
    # 系统总计内存
    zj = float(mem.total) / 1024 / 1024 / 1024
    # 系统已经使用内存
    ysy = float(mem.used) / 1024 / 1024 / 1024

    # 系统空闲内存
    kx = float(mem.free) / 1024 / 1024 / 1024

    print(f'# 系统总计内存:{round(zj, 2)}GB, 已用{round(ysy, 2)}GB, 可用{round(kx, 2)}GB.')


def find_last_modify_file(fpath):
    """返回当前路径下最后修改的文件名"""
    # 列出目录下所有的文件
    list_ = os.listdir(fpath)
    assert len(list_) > 0, f'directory {fpath} is empty.'
    # 对文件修改时间进行升序排列
    list_.sort(key=lambda fn: os.path.getmtime(fpath + '\\' + fn))
    # 获取最新修改时间的文件
    filetime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath + list_[-1]))
    # 获取文件所在目录
    filepath = os.path.join(fpath, list_[-1])
    return filepath


def reduce_mem_data(series):
    """zip data."""
    assert isinstance(series, np.ndarray)
    c_min = np.min(series)
    c_max = np.max(series)
    if series.dtype == int:
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                np.int8).max:
            series = series.astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                np.int16).max:
            series = series.astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                np.int32).max:
            series = series.astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                np.int64).max:
            series = series.astype(np.int64)
    else:
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                np.float16).max:
            series = series.astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                np.float32).max:
            series = series.astype(np.float32)
        elif c_min > np.finfo(np.float64).min and c_max < np.finfo(
                np.float64).max:
            series = series.astype(np.float64)
    return series
