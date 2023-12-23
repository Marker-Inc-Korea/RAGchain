from abc import ABC, abstractmethod
from typing import Union
from uuid import UUID


class BaseLinker(ABC):
    @abstractmethod
    def put_json(self, id: Union[UUID, str], json: dict):
        pass

    @abstractmethod
    def get_json(self, ids: list[Union[UUID, str]]):
        pass

    @abstractmethod
    def flush_db(self):
        pass
