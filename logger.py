import io
import sys
from abc import abstractmethod
from collections import defaultdict
from typing import DefaultDict


class Handler:
    @abstractmethod
    def flush(self, module: str, msg: dict) -> None:
        pass


class StreamHandler(Handler):
    def __init__(self, stream: io.StringIO = sys.stdout) -> None:
        super().__init__()
        self._stream = stream

    def flush(self, module: str, msg: dict) -> None:
        print(module, file=self._stream)
        margin = 2
        for name, val in msg.items():
            if isinstance(val, float):
                val = round(val, 5)
            print(' '*margin + name + ':', val, file=self._stream)


class Logger:
    """
    level - уровень логгера. Различные уровни независимы между собой.
    Пример уровней: "service" - лог, содержащий какую-то служебную информацию,
                    "user"    - лог, содержащий сообщения для пользователя
                    "config" (зарезервированное имя) - лог для конфигурации

    module - логирующая единица. Это может быть какой-то объект, пользователь и т.д. Разделение на
    модули своё для каждого уровня логгера.
    """
    def __init__(self) -> None:
        self._levels_msgs = defaultdict(lambda: defaultdict(dict))
        self._handlers: DefaultDict[str, list[Handler]] = defaultdict(list)
        # level:
        #   module1: {msg}
        #   module2: {msg}

    def log(self, level: str, module: str, msg: dict, accum: bool = True) -> None:
        level_dict = self._levels_msgs[level]
        level_dict[module].update(msg)

        if not accum:
            self.flush(level)

    def flush(self, level: str) -> None:
        for handler in self._handlers[level]:
            for module, msg in self._levels_msgs[level].items():
                handler.flush(module, msg)

    def add_handler(self, level: str, handler: Handler) -> None:
        self._handlers[level].append(handler)

    def config(self, config: dict) -> None:
        self.log(level='config', module='config', msg=config, accum=False)
