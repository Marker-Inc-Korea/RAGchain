import logging
from typing import List, Union, Optional
from uuid import UUID

import openai

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import set_api_base

logger = logging.getLogger(__name__)


class HyDERetrieval(BaseRetrieval):
    BASIC_SYSTEM_PROMPT = "Please write a passage to answer the question"

    def __init__(self,
                 retrieval: BaseRetrieval,
                 system_prompt: str = None,
                 model_name: str = "gpt-3.5-turbo", api_base: str = None, *args, **kwargs):
        self.retrieval = retrieval
        self.system_prompt = self.BASIC_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.model_name = model_name
        set_api_base(api_base)

    def retrieve(self, query: str, db: BaseDB, top_k: int = 5, model_kwargs: Optional[dict] = {}, *args, **kwargs) -> \
    List[Passage]:
        # TODO : use linker for this!
        ids = self.retrieve_id(query, top_k, model_kwargs, *args, **kwargs)
        result = db.fetch(ids)
        return result

    def ingest(self, passages: List[Passage]):
        self.retrieval.ingest(passages)

    def retrieve_id(self, query: str, top_k: int = 5, model_kwargs: Optional[dict] = {}, *args, **kwargs) -> List[
        Union[str, UUID]]:
        user_prompt = f"Question: {query}\nPassage:"
        completion = openai.ChatCompletion.create(model=self.model_name, messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.7, **model_kwargs)
        hyde_answer = completion["choices"][0]["message"]["content"]
        # logging
        logger.info(f"HyDE answer : {hyde_answer}")
        return self.retrieval.retrieve_id(query=hyde_answer, top_k=top_k, *args, **kwargs)

    @staticmethod
    def make_prompt(prompt: str):
        prompt += "\nQuestion: {0}\nPassage:"
        return prompt
