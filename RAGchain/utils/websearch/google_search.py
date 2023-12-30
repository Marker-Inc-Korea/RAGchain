from typing import Dict, List, Optional


from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from RAGchain.utils.websearch import BaseWebSearch
from RAGchain.schema import Passage


class GoogleSearch(BaseWebSearch):
    """
    Langchain's 'GoogleSearchAPIWrapper' returns a List[Dict[str, str]] as the return value.
    This GoogleSearch class wraps this return value in a Passage.
    """
    def __init__(self):
        self.search = GoogleSearchAPIWrapper()
        self.tool = Tool(
            name="GoogleSearch",
            description="Search Google for recent results",
            func=self.get_search_data,
        )

    def get_search_data(self, query, num_results=5, search_params: Optional[Dict[str, str]] = None,) -> List[Passage]:
        search_results = self.search.results(query, num_results, search_params)
        passages = Passage.from_search(search_results)
        return passages
