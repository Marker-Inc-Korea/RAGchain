from typing import Dict, List, Optional

from langchain.tools import Tool
from langchain.utilities import BingSearchAPIWrapper

from RAGchain.utils.websearch import BaseWebSearch
from RAGchain.schema import Passage


class BingSearch(BaseWebSearch):
    """
    Langchain's 'BingSearchAPIWrapper' returns a List[Dict[str, str]] as the return value.
    This BingSearch class wraps this return value in a Passage.
    """
    def __init__(self):
        self.search = BingSearchAPIWrapper()

    def get_search_data(self, query, num_results=5,) -> List[Passage]:
        search_results = self.search.results(query, num_results)
        passages = Passage.from_search(search_results)
        return passages
