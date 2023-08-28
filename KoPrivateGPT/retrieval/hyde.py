import logging
from typing import List, Union
from uuid import UUID

import openai

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage

logger = logging.getLogger(__name__)


class HyDERetrieval(BaseRetrieval):
    BASIC_PROMPT = "Please write a passage to answer the question"

    def __init__(self,
                 retrieval: BaseRetrieval,
                 prompt: str = None,
                 model_name: str = "gpt-3.5-turbo", api_base: str = None, *args, **kwargs):
        self.retrieval = retrieval
        self.prompt = self.make_prompt(self.BASIC_PROMPT) if prompt is None else self.make_prompt(prompt)
        self.model_name = model_name
        BasicLLM.set_model(api_base)

    def retrieve(self, query: str, db: BaseDB, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        # TODO : use linker for this!
        ids = self.retrieve_id(query, top_k)
        result = db.fetch(ids)
        return result

    def ingest(self, passages: List[Passage]):
        self.retrieval.ingest(passages)

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        prompt = self.prompt.format(query)
        completion = openai.ChatCompletion.create(model=self.model_name, prompt=prompt, temperature=0.7)
        hyde_answer = completion["choices"][0]["message"]["content"]
        # logging
        logger.info(f"HyDE answer : {hyde_answer}")
        return self.retrieval.retrieve_id(query=hyde_answer, top_k=top_k, *args, **kwargs)

    @staticmethod
    def make_prompt(prompt: str):
        prompt += "\nQuestion: {0}\nPassage:"
        return prompt
