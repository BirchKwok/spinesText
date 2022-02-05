import copy
import os.path
import numpy as np
from collections import Counter
from pandas import Series
from spinesText.utils import batch_reader


is_exist_path = lambda s: os.path.exists(s)


class TFIDFModel:
    def __init__(self, fp_or_list, vocab, encoding=None):
        self.vocab = vocab
        self.vocab_counter = Counter({k: 0 for k in self.vocab})
        self.fp = fp_or_list
        self.fp_type = self._validate_fp()
        self.tfidf_weight_dict = Counter({k: 1 for k in self.vocab})
        self.contain_doc_counter = Counter({k: 1 for k in self.vocab})
        self.doc_num = 0
        self.__call__(encoding=encoding)

    @staticmethod
    def _split_words(string):
        return string.strip().split()

    def _validate_fp(self):
        if isinstance(self.fp, str):
            if is_exist_path(self.fp):
                self.fp = [self.fp]
                return True
            else:
                raise KeyError('fp_or_list expect path-like string or corpus list'
                               f'{self.fp} not a existent directory or file.')
        else:
            if isinstance(self.fp, (list, tuple, np.ndarray, Series)):
                if is_exist_path(self.fp[0]):
                    for i in self.fp:
                        if not is_exist_path(i):
                            print(f'# {i} not a existent directory or file, will be ignore.')
                    return True
                else:
                    return False

    @staticmethod
    def _tf(counter):
        """计算词频"""
        counter_ = copy.deepcopy(counter)
        _ = np.sum(list(counter_.values()))
        for i in counter_:
            counter_[i] /= _

        return counter_

    @staticmethod
    def _contain_docs(doc_list, counter):
        """计算包含vocab中每个词的文档数"""
        for i in counter:
            for j in doc_list:  # list[list[str str str]]
                if i in j:
                    counter[i] += 1
        return counter

    @staticmethod
    def _idf(counter, doc_num):
        counter_ = copy.deepcopy(counter)
        for i in counter_:
            counter_[i] = np.log(doc_num / (counter_[i] + 1))
        return counter_

    def __call__(self, encoding=None):
        """
        计算传入文本的tfidf值
        """
        if self.fp_type:
            for fp in self.fp:
                import sys
                encoding = sys.getdefaultencoding() if encoding is None else encoding
                for lines in batch_reader(fp, encoding=encoding):  # list[str]
                    # 分词
                    lines = [self._split_words(i) for i in lines]  # list[list[str]]
                    for line in lines:
                        # 统计词量
                        self.vocab_counter += Counter(line)
                        self.doc_num += 1
                    # 统计文档数
                    self.contain_doc_counter = self._contain_docs(lines, self.contain_doc_counter)
        else:
            for lines in batch_reader(self.fp):  # list[str]
                # 分词
                lines = [self._split_words(i) for i in lines]
                for line in lines:
                    # 统计词量
                    self.vocab_counter += Counter(line)
                    self.doc_num += 1
                # 统计文档数
                self.contain_doc_counter = self._contain_docs(lines, self.contain_doc_counter)

        self.tfidf_weight_dict = self._tf(self.vocab_counter)
        idf_dict = self._idf(self.contain_doc_counter, self.doc_num)
        for i in self.tfidf_weight_dict:
            self.tfidf_weight_dict[i] = self.tfidf_weight_dict[i] / idf_dict.get(i, 1)

    def get_tfidf(self):
        return self.tfidf_weight_dict




