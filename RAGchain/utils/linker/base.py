from abc import abstractmethod
from typing import Union
from uuid import UUID


class Singleton(type):
    _instances = {"CHILD_CREATED": False}

    def __call__(cls, allow_multiple_instances=False, *args, **kwargs):
        if cls not in cls._instances or allow_multiple_instances:
            if cls._instances["CHILD_CREATED"] and not allow_multiple_instances:
                raise SingletonCreationError("Instance of linker already created. Cannot create another linker.")
            cls._instances[cls] = super().__call__(*args, **kwargs)
            cls._instances["CHILD_CREATED"] = True
        return cls._instances[cls]


class BaseLinker(metaclass=Singleton):
    @abstractmethod
    def put_json(self, ids: list[Union[UUID, str]], json_data_list: list[dict]):
        pass

    @abstractmethod
    def get_json(self, ids: list[Union[UUID, str]]):
        pass

    @abstractmethod
    def flush_db(self):
        pass

    @abstractmethod
    def delete_json(self, id: Union[UUID, str]):
        pass


class SingletonCreationError(Exception):
    """
    Exception to be raised when trying to create another singleton instance.
    """
    pass


class NoIdWarning(UserWarning):
    """
    Warning to be raised when there is no id in Linker.
    """
    pass


class NoDataWarning(UserWarning):
    """
    Warning to be raised when there is no data in Linker.
    """
    pass
