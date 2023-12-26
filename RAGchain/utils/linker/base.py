from abc import ABC, abstractmethod
from typing import Union
from uuid import UUID


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = \
                super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseLinker(metaclass=Singleton):
    @abstractmethod
    def put_json(self, id: Union[UUID, str], json: dict):
        pass

    @abstractmethod
    def get_json(self, ids: list[Union[UUID, str]]):
        pass

    @abstractmethod
    def flush_db(self):
        pass
