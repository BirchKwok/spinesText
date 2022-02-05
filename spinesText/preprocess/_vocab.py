import collections
import warnings
from spinesText.base import VocabMixin
import datetime
import gc
import os.path
import time
from typing import Dict, Union
import numpy as np
from spinesText.preprocess._text_split import TextCutWords, TextSplitSentence
from spinesText.utils import batch_reader, iter_count, json_rw, return_useful_mem


def count_corpus(tokens):
    """统计词元的频率。"""
    # 这里的 `tokens` 是 1D 列表或 2D 列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成使用词元填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab(VocabMixin):
    """构建文本词汇表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None, vocab_dict: Union[Dict, None] = None,
                 loading_vocab_path=None, unknown_words='<unk>'):
        """
        tokens: str-like or [str]
        min_freq: 限制最小词频
        reserved_tokens: 初始化tokens表
        local_vocab: 本地词典, dict-like
        vocab_path: 若词典为文件, 此参数即为文件路径
        """
        if isinstance(vocab_dict, (dict, BatchVocab)):  # 支持传入本地vocab
            if isinstance(vocab_dict, BatchVocab):
                self.token_to_idx = vocab_dict.token_to_idx
            else:
                self.token_to_idx = vocab_dict
            self.idx_to_token = list(self.token_to_idx.keys())

            if unknown_words in self.idx_to_token:
                self.unk = self.token_to_idx[unknown_words]
            else:
                self.unk = 0 if 0 not in self.token_to_idx.values() else max(self.token_to_idx.values()) + 1

        elif loading_vocab_path is not None:
            assert os.path.exists(loading_vocab_path), f"No such file or directory:{loading_vocab_path}"
            self.token_to_idx = json_rw(loading_vocab_path)
            self.idx_to_token = list(self.token_to_idx.keys())

            if unknown_words in self.idx_to_token:
                self.unk = self.token_to_idx[unknown_words]
            else:
                self.unk = 0 if 0 not in self.token_to_idx.values() else max(self.token_to_idx.values()) + 1
        else:
            if tokens is None:
                tokens = []
            if reserved_tokens is None:
                reserved_tokens = []
            # 按出现频率排序
            counter = count_corpus(tokens)
            self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                      reverse=True)
            # 未知词元的索引为0
            self.unk, uniq_tokens = 0, [unknown_words] + reserved_tokens
            uniq_tokens += [token for token, freq in self.token_freqs
                            if freq >= min_freq and token not in uniq_tokens]
            self.idx_to_token, self.token_to_idx = [], dict()
            for token in uniq_tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1


class TextVecTransformer(Vocab):
    """将传入的文章转为等长矩阵"""

    def __init__(self, vocab_dict: Union[Dict, None] = None, loading_vocab_path=None,
                 stopwords_path=None, domainwords_path=None, needcut=False, cut_all=False, clear_stopwords=False):
        """
        local_vocab: 本地词典, dict-like
        vocab_path: 若词典为文件, 此参数即为文件路径
        stopwords_path: 停用词文件路径, 默认为None, 使用内置停用词
        domainwords_path: 领域词文件路径, 默认为None, 使用内置领域词汇
        needcut: 是否需要切词
        cut_all: 是否需要采用全模式切词
        """
        super(TextVecTransformer, self).__init__(vocab_dict=vocab_dict, loading_vocab_path=loading_vocab_path)
        self._needcut = needcut
        self._stopwords_path = stopwords_path
        self._domainwords_path = domainwords_path
        self._cut_all = cut_all
        self._clear_stopwords=clear_stopwords

    def transform(self, tokens, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.0):
        if isinstance(tokens, str):
            if self._needcut:
                self._text_to_words = TextCutWords(stopwords_path=self._stopwords_path,
                                                   domainwords_path=self._domainwords_path).text_to_words
                self.words_list = self._text_to_words(tokens, cut_all=self._cut_all, clear_stopwords=self._clear_stopwords)
            else:
                self.words_list = [[tokens]]
        elif isinstance(tokens, (list, tuple, np.ndarray)):
            if self._needcut:
                self._text_to_words = TextCutWords(stopwords_path=self._stopwords_path,
                                                   domainwords_path=self._domainwords_path).text_to_words
                self.words_list = self._text_to_words(tokens, cut_all=self._cut_all,
                                                      clear_stopwords=self._clear_stopwords)
            else:
                if isinstance(tokens[0], (list, tuple, np.ndarray)):
                    self.words_list = tokens
                else:
                    self.words_list = [tokens]

        self.words_list = super().__getitem__(self.words_list)

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        _ = pad_sequences(self.words_list, padding=padding, maxlen=maxlen,
                          dtype=dtype, truncating=truncating, value=value)
        return _


class BatchVocab(VocabMixin):
    """将传入的本地文件或者字符串分词, 生成词典dict, 支持多文件生成vocab"""

    def __init__(self, fpath_or_str, batch_size=1000, encoding='utf-8', min_freq=0, reserved_tokens=None, needcut=True,
                 stopwords_path=None, domainwords_path=None, cut_all=True, save_path=None, clear_stopwords=False):
        """
        fpath_or_str: str-like, file path or string; Or, file path list, 如果其中一个不存在，默认跳过
        batch_size: 每次读取的文件行数
        encoding: 文件读取指定编码
        min_freq: 限制最小词频
        reserved_tokens: 初始化tokens表
        needcut: 是否需要切词
        stopwords_path: 停用词文件路径, 默认为None, 使用内置停用词
        domainwords_path: 领域词文件路径, 默认为None, 使用内置领域词汇
        cut_all: 是否需要采用全模式切词
        save_path: vocab保存路径, 默认为None
        """
        self.batch_size = batch_size

        text_to_words = TextCutWords(stopwords_path=stopwords_path,
                                     domainwords_path=domainwords_path).text_to_words

        if reserved_tokens is None:
            reserved_tokens = []

        # 未知词元的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        self.idx_to_token, self.token_to_idx = [], dict()

        if isinstance(fpath_or_str, str):
            fpath_or_str = fpath_or_str if os.path.exists(fpath_or_str) \
                else TextSplitSentence().split_sen(fpath_or_str)
            self.token_freqs = self._run_generator(fpath_or_str, encoding, batch_size, needcut, text_to_words, cut_all,
                       clear_stopwords)
        elif isinstance(fpath_or_str, (list, tuple, np.ndarray)):
            fpath_list = []
            for i in fpath_or_str:
                if not os.path.exists(i):
                    warnings.warn(f"file {i} not exists, default to skip.")
                    continue
                fpath_list.append(i)

            self.token_freqs = self._run_generator(fpath_list, encoding, batch_size, needcut, text_to_words, cut_all,
                                                   clear_stopwords)
        else:
            raise ValueError(f"param {fpath_or_str} is invalid, only support to string or "
                             f"sequence(list, tuple, array) type.")

        self.token_freqs = sorted(self.token_freqs.items(), key=lambda x: x[1],
                                  reverse=True)

        # 唯一的token
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]

        for token in uniq_tokens:
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

        if save_path is not None:
            # 保存所有batch_size结果
            self.save_vocab(save_path, encoding='utf-8')

    @staticmethod
    def _run_generator(fpath_or_str, encoding, batch_size, needcut, text_to_words, cut_all,
                       clear_stopwords):

        lines_num = iter_count(fpath_or_str, encoding=encoding) if isinstance(fpath_or_str, str) and os.path.exists(
            fpath_or_str) \
            else len(fpath_or_str)

        cycles = int(np.ceil(lines_num / batch_size))
        _ = 1
        print(fpath_or_str, 'starts processing...')

        token_freqs = collections.Counter(())

        if isinstance(fpath_or_str, str):
            for t in batch_reader(fpath_or_str, encoding=encoding, batch_size=batch_size):
                print(f"Cycle {_} execute at {datetime.datetime.now().strftime('%H:%M:%S %Y-%m-%d')}, "
                      f"with {cycles - _} cycles remaining.")
                tik = time.time()
                if needcut:
                    # 将文本连接起来, 组成字符串
                    t = ''.join(t)
                    # 对传入tokens分词
                    t = text_to_words(t, cut_all=cut_all, clear_stopwords=clear_stopwords)

                # 按出现频率排序
                # 需要修改，保存每次计算的结果，遍历完数据后，再进行新一次排序
                counter = count_corpus(t)
                token_freqs += counter

                del counter
                gc.collect()
                tok = time.time()
                return_useful_mem()

                print(
                    f"# executed in {round((tok - tik), 2)}s, finished {datetime.datetime.now().strftime('%H:%M:%S %Y-%m-%d')}")
                _ += 1
        else:
            for i in fpath_or_str:
                for t in batch_reader(i, encoding=encoding, batch_size=batch_size):
                    print(f"Cycle {_} execute at {datetime.datetime.now().strftime('%H:%M:%S %Y-%m-%d')}, "
                          f"with {cycles - _} cycles remaining.")
                    tik = time.time()
                    if needcut:
                        # 将文本连接起来, 组成字符串
                        t = ''.join(t)
                        # 对传入tokens分词
                        t = text_to_words(t, cut_all=cut_all, clear_stopwords=clear_stopwords)

                    # 按出现频率排序
                    # 需要修改，保存每次计算的结果，遍历完数据后，再进行新一次排序
                    counter = count_corpus(t)
                    token_freqs += counter

                    del counter
                    gc.collect()
                    tok = time.time()
                    return_useful_mem()

                    print(
                        f"# executed in {round((tok - tik), 2)}s, finished {datetime.datetime.now().strftime('%H:%M:%S %Y-%m-%d')}")
                    _ += 1
        return token_freqs


