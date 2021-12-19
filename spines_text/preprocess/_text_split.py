import copy
import gc
import os
import re
from typing import List, Union, Set
import logging
try:
    import jieba_fast as jieba
    from jieba_fast import load_userdict, lcut
except ModuleNotFoundError:
    import jieba
    from jieba import load_userdict, lcut

jieba.setLogLevel(logging.INFO)

from joblib import Parallel, delayed
from numba import jit
from warnings import filterwarnings

filterwarnings('ignore')

FILE_PATH = os.path.dirname(__file__)
STOPWORDS_PATH = os.path.join(FILE_PATH, 'stopwords.txt')


def load_stopwords(stopwords_path=STOPWORDS_PATH):
    with open(stopwords_path, 'r', encoding='utf8') as f:
        d = f.readlines()

    stopwords = []
    for i in d:
        _ = i.replace('\n', '')
        stopwords.append(_)
    del d
    gc.collect()
    return set(stopwords)


@jit(forceobj=True)  # 使用对象模式，缩短模式侦测时间
def _clear_stopwords(stopwords, series: List[str]):
    _ = []
    for i in series:
        if i not in stopwords and i.strip() != '' and len(i) > 1:
            _.append(i)
    return _


class WordCut:
    """句子分词"""

    def __init__(self, stopwords_path=None, domainwords_path=None, types='word', jieba_init=False):
        assert types in ['word', 'char']

        if jieba_init:
            jieba.initialize()

        self.types = types
        if self.types == 'word':
            # 加入特定领域词
            if domainwords_path is not None:
                load_userdict(domainwords_path)

        self.stopwords_path = stopwords_path

    def cut(self, string: str, clear_stopwords=True, max_len=None, cut_all=False, all_zh=True):
        assert isinstance(string, str), "string must be str type."
        assert max_len is None or (isinstance(max_len, int) and max_len != 0 and max_len >= 1)

        if all_zh:
            s = re.sub('[^\u4e00-\u9fa5]', '', string)
        else:
            s = re.sub('([\\t\r\!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n，'
                       '？！~·‘、：《》【】（）￥…」「『 』])|([0-9]{7,})', '', string)

        s = s.lower().strip()

        if max_len is not None:
            s = s[:max_len]

        if self.types == 'word':
            self._words = lcut(s, cut_all=cut_all)
            if clear_stopwords:
                # 获取停用词表
                if not getattr(WordCut, 'stopwords'):
                    if self.stopwords_path is not None:
                        with open(self.stopwords_path, 'r', encoding='utf8') as f:
                            d = f.readlines()

                        self.stopwords = []
                        for i in d:
                            _ = i.replace('\n', '')
                            self.stopwords.append(_)
                        del d
                        gc.collect()
                    else:
                        self.stopwords = load_stopwords()

                self._words = _clear_stopwords(self.stopwords, self._words)
        else:
            self._words = list(s)

        return self._words  # List[str]


class TextSplitSentence:
    """文章分句"""

    def split_sen(self, text: Union[str, List[str]], sen_sep_re='。!?？！(……)： (...)\\n\\t(||)',
                  filters: Union[Set, None] = None):
        """filters: if None, default to {'\r', '\t', '“', '”', '"', "'", '’', '《', '》', '‘'}"""
        if filters is None:
            filters = {'\r', '“', '”', '"', "'", '’', '《', '》', '‘'}

        if isinstance(text, str):
            for i in filters:
                t = text.replace(i, '')
        else:
            if isinstance(text[0], str):
                for i in filters:
                    t = ''.join([sen.replace(i, '') for sen in text])
            else:
                for i in filters:
                    t = ''.join([s.replace(i, '') for sen in text for s in sen])

        self._sen = re.split(f'[{sen_sep_re}]', t)
        del t
        gc.collect()
        return [i for i in self._sen if i is not None and i.strip() != '']


class TextCutWords(WordCut, TextSplitSentence):
    """文章分词"""
    def __init__(self, stopwords_path=None, domainwords_path=None, types='word', jieba_init=False):
        super(TextCutWords, self).__init__(stopwords_path, domainwords_path,
                                           types, jieba_init)

    def text_to_words(self, text, sen_sep_re='。?!？！(……)： (...)\\n\\t(||)',
                      filters: Union[List, None] = None, cut_all=False, clear_stopwords=False, all_zh=True):
        t = copy.deepcopy(text)
        sen = self.split_sen(t, sen_sep_re=sen_sep_re, filters=filters)
        words_list = []
        for i in sen:
            _ = self.cut(i, cut_all=cut_all, clear_stopwords=clear_stopwords, all_zh=all_zh)
            if len(_) > 0:
                words_list.append(_)
        del t, sen
        gc.collect()

        return words_list

    def text_to_text(self, text, join_sep= ' ', sen_sep_re='。?!？！(……)： (...)\\n\\t(||)',
                      filters: Union[List, None] = None, cut_all=False, clear_stopwords=False):
        """
        切词后按指定join_sep连接每个词汇
        """
        word_list = self.text_to_words(text, sen_sep_re, filters, cut_all, clear_stopwords)

        def _join(i):
            return join_sep.join(i)

        return Parallel(n_jobs=-1)(delayed(_join)(i) for i in word_list)
