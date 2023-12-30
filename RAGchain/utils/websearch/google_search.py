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

