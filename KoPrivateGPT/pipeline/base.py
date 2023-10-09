from abc import ABC, abstractmethod


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
