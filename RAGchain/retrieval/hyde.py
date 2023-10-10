import logging
from typing import List, Union, Optional
from uuid import UUID

import openai

from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage
from RAGchain.utils.util import set_api_base

logger = logging.getLogger(__name__)


class HyDERetrieval(BaseRetrieval):
    """
    HyDE Retrieval, which inspired by "Precise Zero-shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496.pdf)
    At retrieval, LLM model creates hypothetical passage.
    And then, retrieve passages using hypothetical passage as query.
    """
    BASIC_SYSTEM_PROMPT = "Please write a passage to answer the question"

    def __init__(self, retrieval: BaseRetrieval, system_prompt: str = None, model_name: str = "gpt-3.5-turbo",
                 api_base: str = None, *args, **kwargs):
        """
        :param retrieval: retrieval instance to use
        :param system_prompt: system prompt to use when generating hypothetical passage
        :param model_name: model name to use when generating hypothetical passage
        :param api_base: api base url to use when generating hypothetical passage
        """
        super().__init__()
        self.retrieval = retrieval
        self.system_prompt = self.BASIC_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.model_name = model_name
        set_api_base(api_base)

    def retrieve(self, query: str, top_k: int = 5, model_kwargs: Optional[dict] = {}, *args, **kwargs) -> \
            List[Passage]:
        ids = self.retrieve_id(query, top_k, model_kwargs, *args, **kwargs)
        result = self.retrieval.fetch_data(ids)
        return result

    def ingest(self, passages: List[Passage]):
        self.retrieval.ingest(passages)

    def retrieve_id(self, query: str, top_k: int = 5, model_kwargs: Optional[dict] = {}, *args, **kwargs) -> List[
        Union[str, UUID]]:
        ids, scores = self.retrieve_id_with_scores(query, top_k, model_kwargs, *args, **kwargs)
        return ids

    def retrieve_id_with_scores(self, query: str, top_k: int = 5, model_kwargs: Optional[dict] = {}, *args, **kwargs) -> \
    tuple[List[Union[str, UUID]], List[float]]:
        user_prompt = f"Question: {query}\nPassage:"
        completion = openai.ChatCompletion.create(model=self.model_name, messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ], **model_kwargs)
        hyde_answer = completion["choices"][0]["message"]["content"]
        # logging
        logger.info(f"HyDE answer : {hyde_answer}")
        return self.retrieval.retrieve_id_with_scores(query=hyde_answer, top_k=top_k, *args, **kwargs)

    @staticmethod
    def make_prompt(prompt: str):
        prompt += "\nQuestion: {0}\nPassage:"
        return prompt
