from collections import OrderedDict


class PipeDict(OrderedDict):

    def __init__(self, length, element=None):
        super(PipeDict, self).__init__()
        self._length = length
        if element is not None:
            for k, v in element.items():
                self.__setitem__(k, v)

    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if len(self) - containsKey >= self._length:
            self.popitem(last=False)
        if containsKey:
            del self[key]
        OrderedDict.__setitem__(self, key, value)










