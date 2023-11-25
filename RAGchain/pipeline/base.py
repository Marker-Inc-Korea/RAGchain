from abc import ABC, abstractmethod
from typing import Optional, List

from langchain.schema.runnable import Runnable

from RAGchain.schema import Passage


class BasePipeline(ABC):
    """
    Base class for all pipelines
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the pipeline
        """
        pass


class BaseRunPipeline(ABC):
    def __init__(self):
        self.run: Optional[Runnable] = None
        self._make_runnable()
        if self.run is None:
            raise NotImplementedError("You should implement __make_runnable method")

    @abstractmethod
    def _make_runnable(self):
        """initialize runnable"""
        pass

    @abstractmethod
    def get_passages_and_run(self, questions: List[str]) -> tuple[List[str], List[List[Passage]], List[List[float]]]:
        """
        Run the pipeline for evaluator, and get retrieved passages and rel scores.
        It is same with pipeline.run.batch, but returns passages and rel scores.
        Return List of answer, List of passages, Relevance score of passages.
        """
        pass
