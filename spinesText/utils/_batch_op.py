import datetime
import itertools
import time
from functools import wraps
import os.path
import numpy as np


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        print(f"\r{func.__name__} execute at {datetime.datetime.now().strftime('%H:%M:%S %Y-%m-%d')}")
        tik = time.time()
        res = func(*args, **kwargs)
        tok = time.time()
        print(
            f"\n\r# {func.__name__} executed in {round((tok - tik), 2)}s, "
            f"finished {datetime.datetime.now().strftime('%H:%M:%S %Y-%m-%d')}"
        )
        return res

    return wrap


def iter_count(file_name, encoding='utf-8'):  # GB18030
    """计算文件行数"""
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024  # 设置指定缓冲块大小
    with open(file_name, encoding=encoding) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def batch_reader(fp, encoding='utf-8', batch_size=1000, **open_func_kwargs):  # GB18030
    """
    支持传入文件路径或list类型变量
    return： List[str]
    """
    if isinstance(fp, str) and os.path.exists(fp):
        lines_num = iter_count(fp)
        assert lines_num > 0
        range_cycles = range(int(np.ceil(lines_num / batch_size)))
        with open(fp, 'r', encoding=encoding, **open_func_kwargs) as f:
            for i in range_cycles:
                text = []
                for j in range(batch_size):
                    line = f.readline()
                    text.append(line)

                yield text
    else:
        assert isinstance(fp, list)
        lines_num = len(fp)
        assert lines_num > 0
        range_cycles = range(int(np.ceil(lines_num / batch_size)))
        for i in range_cycles:
            start_p = batch_size * i
            yield fp[start_p: batch_size * (i+1)]


def flatten(list_of_list):
    """将两层或多层iterable数据结构，展平到一层list"""
    return list(itertools.chain.from_iterable(list_of_list))

