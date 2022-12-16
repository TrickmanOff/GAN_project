import io
import sys
from abc import abstractmethod
from collections import defaultdict


class Logger:
    def __init__(self) -> None:
        self._levels_msgs = defaultdict(lambda: defaultdict(dict))
        # level:
        #   module1: {msg}
        #   module2: {msg}

    def log(self, level: str, module: str, msg: dict, accum: bool = True) -> None:
        if not accum:
            self._flush(level, module, msg)
        else:
            level_dict = self._levels_msgs[level]
            level_dict[module] |= msg

    def flush(self, level: str) -> None:
        for module, msg in self._levels_msgs[level].items():
            self._flush(level, module, msg)

    @abstractmethod
    def config(self, config: dict) -> None:
        pass

    @abstractmethod
    def _flush(self, level: str, module: str, msg: dict) -> None:
        pass


class StreamLogger(Logger):
    def __init__(self, stream: io.StringIO = sys.stdout) -> None:
        super().__init__()
        self._stream = stream

    def config(self, config: dict) -> None:
        # TODO: добавить поддержку произвольного кол-ва уровней вложенности
        print('== Config ==', file=self._stream)
        margin = 2
        for name1, val in config.items():
            if isinstance(val, dict):
                print(name1 + ':', file=self._stream)
                for name2, inner_val in val.items():
                    print(' '*margin + name2 + ':', inner_val, file=self._stream)
            else:
                print(name1 + ':', val, file=self._stream)
        print('============', file=self._stream)

    def _flush(self, level: str, module: str, msg: dict) -> None:
        if level == 'batch' or level == 'epoch':
            return
        print(level + '/' + module, file=self._stream)
        margin = 2
        for name, val in msg.items():
            if isinstance(val, float):
                val = round(val, 5)
            print(' '*margin + name + ':', val, file=self._stream)
