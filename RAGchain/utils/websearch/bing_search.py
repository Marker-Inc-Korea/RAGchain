from typing import List

from langchain.utilities import BingSearchAPIWrapper

from RAGchain.utils.websearch import BaseWebSearch
from RAGchain.schema import Passage


class BingSearch(BaseWebSearch):
    """
    Langchain's 'BingSearchAPIWrapper' returns a List[Dict[str, str]] as the return value.
    This BingSearch class wraps this return value in a Passage.

    First, you need to set up the proper API keys and environment variables.
    To set it up, create the BING_SUBSCRIPTION_KEY in the Bing Search API
    (https://portal.azure.com/#home) and a BING_SEARCH_URL using the Bing Search API
    """
    def __init__(self):
        self.search = BingSearchAPIWrapper()

    def get_search_data(self, query, num_results=5,) -> List[Passage]:
        search_results = self.search.results(query, num_results)
        passages = Passage.from_search(search_results)
        return passages
