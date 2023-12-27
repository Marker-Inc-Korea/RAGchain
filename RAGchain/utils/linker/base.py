from abc import abstractmethod
from typing import Union
from uuid import UUID
import warnings


class Singleton(type):
    _instances = {"CHILD_CREATED": False}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            if cls._instances["CHILD_CREATED"]:
                raise SingletonCreationError("Instance of linker already created. Cannot create another linker.")
            cls._instances[cls] = super().__call__(*args, **kwargs)
            cls._instances["CHILD_CREATED"] = True
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


class SingletonCreationError(Exception):
    """
    Exception to be raised when trying to create another singleton instance.
    """
    pass
