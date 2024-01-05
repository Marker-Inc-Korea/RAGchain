from typing import Dict, List, Optional, Union, Any

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.schema import Passage
from RAGchain.utils.websearch import BaseWebSearch


class GoogleSearch(BaseWebSearch):
    """
    Langchain's 'GoogleSearchAPIWrapper' returns a List[Dict[str, str]] as the return value.
    This GoogleSearch class wraps this return value in a Passage.

    First, you need to set up the proper API keys and environment variables.
    To set it up, create the GOOGLE_API_KEY in the Google Cloud credential console
    (https://console.cloud.google.com/apis/credentials) and a GOOGLE_CSE_ID using the Programmable Search Engine
    (https://programmablesearchengine.google.com/controlpanel/create).
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

    def batch(
            self,
            inputs: List[Input],
            config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
            *,
            return_exceptions: bool = False,
            **kwargs: Optional[Any],
    ) -> List[Output]:
        outputs = []
        for _input, _config in zip(inputs, config):
            if not isinstance(_input, str):
                raise TypeError(f"Input type must be str, but {type(_input)}")
            outputs.append(self.invoke(_input, _config))
        return outputs
