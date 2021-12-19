import numpy as np

from spines_text.utils import json_rw


class KerasMixin:
    def _callback(self, monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True):
        from tensorflow import keras

        self._callback = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta,
                                                       patience=patience, restore_best_weights=restore_best_weights)

    def _get_batch_size(self, x, batch_size='auto'):
        if batch_size == 'auto':
            self._batch_size = 32 if len(x) < 800 else len(x) // 40
        else:
            assert isinstance(batch_size, int) and batch_size > 0
            self._batch_size = batch_size

    def _get_input_shape(self, x):
        assert isinstance(x, np.ndarray)
        self._shape = x.shape[-1]

    def _fit(self, x, batch_size='auto'):
        self._get_batch_size(x, batch_size)
        self._get_input_shape(x)


class VocabMixin:
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """按索引返回token文本"""
        if not isinstance(indices, (list, tuple, np.ndarray)):
            return self.idx_to_token[indices]
        elif isinstance(indices[0], (list, tuple, np.ndarray)) and indices.ndim == 2:
            _ = []
            for index in indices:
                _.append(self.to_tokens(index))
            return _
            # return [self.idx_to_token[i] for index in indices for i in index]
        return [self.idx_to_token[index] for index in indices]

    def transform(self, text_index_list, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.0):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return pad_sequences(text_index_list, padding=padding, maxlen=maxlen,
                             dtype=dtype, truncating=truncating, value=value)

    def save_vocab(self, path, encoding='utf-8'):
        json_rw(json_path=path, method='w', json_dict=self.token_to_idx, encoding=encoding)
        print(f"# file {path} saved.")
