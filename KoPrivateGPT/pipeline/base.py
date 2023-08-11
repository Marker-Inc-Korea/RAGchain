from abc import ABC, abstractmethod

from pydantic import BaseModel


class BasePipeline(ABC, BaseModel):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass
