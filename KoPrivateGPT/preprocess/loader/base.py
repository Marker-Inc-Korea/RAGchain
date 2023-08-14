from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document


class BaseLoader(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self) -> List[Document]:
        pass
