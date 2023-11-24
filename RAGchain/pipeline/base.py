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
        self.runnable: Optional[Runnable] = None
        self._make_runnable()
        if self.runnable is None:
            raise NotImplementedError("You should implement __make_runnable method")

    @abstractmethod
    def _make_runnable(self):
        """initialize runnable"""
        pass

    @abstractmethod
    def run(self, questions: List[str]) -> tuple[List[str], List[List[Passage]], List[List[float]]]:
        """
        Run the pipeline for evaluator.
        Return List of answer, List of passages, Relevance score of passages.
        """
        pass

    def invoke(self, *args, **kwargs):
        return self.runnable.invoke(*args, **kwargs)

    def batch(self, *args, **kwargs):
        return self.runnable.batch(*args, **kwargs)

    def stream(self, *args, **kwargs):
        yield self.runnable.stream(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.runnable.ainvoke(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self.runnable.abatch(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        async for result in self.runnable.astream(*args, **kwargs):
            yield result

    async def astream_log(self, *args, **kwargs):
        yield self.runnable.astream_log(*args, **kwargs)
