from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.utils import get_from_dict_or_env
from langchain_core.pydantic_v1 import Extra, root_validator

from RAGchain.utils.websearch import BaseWebSearch
from RAGchain.schema import Passage


class GoogleSearchAPIWrapper(BaseWebSearch):
    """Google Search API Wrapper."""
    search_engine: Any  #: :meta private:
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    siterestrict: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        cse = self.search_engine.cse()
        if self.siterestrict:
            cse = cse.siterestrict()
        res = cse.list(q=search_term, cx=self.google_cse_id, **kwargs).execute()
        return res.get("items", [])

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(values, "google_cse_id", "GOOGLE_CSE_ID")
        values["google_cse_id"] = google_cse_id

        try:
            from googleapiclient.discovery import build

        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Please install it with `pip install google-api-python-client"
                ">=2.100.0`"
            )

        service = build("customsearch", "v1", developerKey=google_api_key)
        values["search_engine"] = service

        return values

    def results(
            self,
            query: str,
            num_results: int = 5,
            search_params: Optional[Dict[str, str]] = None,
    ) -> List[Passage]:
        """Run query through GoogleSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.
            search_params: Parameters to be passed on search

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """

        results = self._google_search_results(
            query, num=num_results, **(search_params or {})
        )
        if len(results) == 0:
            return []
        passages = []
        ids = [uuid4() for _ in range(len(results))]
        for i, (results, uuid) in enumerate(zip(results, ids)):
            metadata_etc = results["title"]
            filepath = results["link"]
            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(results) - 1 else None
            passage = Passage(id=uuid,
                              content=results["snippet"],
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)
            passages.append(passage)
        return passages

import os

from typing import List
from operator import itemgetter
from dotenv import load_dotenv
from uuid import uuid4

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms import BaseLLM
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.schema import RAGchainPromptTemplate, Passage

load_dotenv()


class GoogleSearchRunPipeline(BaseRunPipeline):
    def __init__(self,
                 llm: BaseLLM,
                 prompt: RAGchainPromptTemplate = None,
                 use_search_count: int = 3,
                 ):
        """
        Initializes an instance of the GoogleSearchRunPipeline class.
        :param llm: An instance of the Langchain LLM module used for generating answers.
        :param prompt: RAGchainPromptTemplate used for generating prompts based on passages and user query.
        :param use_search_count: The number of Google search result to be used for llm question answering. Default is 3.
        """
        self.llm = llm
        self.prompt = prompt if prompt is not None else self.default_prompt
        self.use_search_count = use_search_count
        super().__init__()

    def _make_runnable(self):
        self.run = {
                       "passages": itemgetter("question") | RunnableLambda(lambda question: Passage.make_prompts(
                           self.__search_passages(question))),
                       "question": itemgetter("question"),
                   } | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str]) -> tuple[List[str], List[List[Passage]], List[List[float]]]:
        pass

    def __search_passages(self, query: str):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        search = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id,
                                        k=self.use_search_count)
        tool = Tool(
            name="GoogleSearch",
            description="Search Google for recent results",
            func=search.results,
        )
        search_results = tool.run(query)
        # TODO: If no search results were found, search_results returns
        #  [{"Result": "No good Google Search Result was found"}].
        #  When we have the ability to 'say we don't know', deal with the case of no search results.
        passages = []
        ids = [uuid4() for _ in range(len(search_results))]
        for i, (search_results, uuid) in enumerate(zip(search_results, ids)):
            metadata_etc = search_results["title"]
            filepath = search_results["link"]
            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(search_results) - 1 else None
            passage = Passage(id=uuid,
                              content=search_results["snippet"],
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)
            passages.append(passage)
        return passages, [i / len(passages) for i in range(len(passages), 0, -1)]
