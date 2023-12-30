from abc import abstractmethod
from typing import Dict, Optional, List

from pydantic import utils as __pydantic_utils

from RAGchain.schema import Passage


class BaseWebSearch(__pydantic_utils.Representation):
    """
    Abstract class for using a web search engine for passage contents.
    """
    @abstractmethod
    def results(self, query: str, num_results: int, search_params: Optional[Dict[str, str]] = None) -> List[Passage]:
        """
        Abstract method for searching passages from the web search engine.
        """
        pass
