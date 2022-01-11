import os.path
import numpy as np
from collections import Counter
from pandas import Series
from spines_text.utils import batch_reader



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

    def _tf(self, word_list):
        """计算词频"""
        c_list = Counter(word_list)  # list[str str str]
        _ = np.sum(list(c_list.values()))
        self.doc_num += _
        for i in c_list:
            c_list[i] /= _

        return c_list

    def _contain_docs(self, doc_list):
        """计算包含vocab中每个词的文档数"""
        for i in self.contain_doc_counter:
            for j in doc_list:  # list[list[str str str]]
                if i in j:
                    self.contain_doc_counter[i] += 1
        return

    def _idf(self):
        for i in self.contain_doc_counter:
            self.tfidf_weight_dict = np.log(self.doc_num / (np.array(self.contain_doc_counter.values()) + 1))

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
                    lines = [self._split_words(i) for i in lines]
                    for line in lines:
                        self._tf(line)




