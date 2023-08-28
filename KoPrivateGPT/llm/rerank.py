from typing import List

import openai

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.utils.reranker.base import BaseReranker
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class RerankLLM(BaseLLM):
    def __init__(self, retrieval: BaseRetrieval, reranker: BaseReranker, model_name: str = "gpt-3.5-turbo",
                 api_base: str = None, retrieve_size: int = 10, use_passage_count: int = 3,
                 window_size: int = 10, *args, **kwargs):
        self.retrieval = retrieval
        self.reranker = reranker
        self.model_name = model_name
        BasicLLM.set_model(api_base)
        assert (retrieve_size > use_passage_count)
        assert (retrieve_size > 0)
        assert (use_passage_count > 0)
        assert (window_size > 0)
        self.retrieve_size = retrieve_size
        self.use_passage_count = use_passage_count
        self.window_size = window_size

    def ask(self, query: str) -> tuple[str, List[Passage]]:
        passages = self.retrieval.retrieve(query, top_k=self.retrieve_size)

        if self.retrieve_size <= self.window_size:
            reranked_passages = self.reranker.rerank(query, passages)
        else:
            reranked_passages = self.reranker.rerank_sliding_window(query, passages, self.window_size)
        final_passages = reranked_passages[:self.use_passage_count]
        contents = "\n\n".join([passage.content for passage in final_passages])
        completion = openai.ChatCompletion.create(model=self.model_name,
                                                  messages=BasicLLM.get_messages(contents, query),
                                                  temperature=0.5)
        answer = completion["choices"][0]["message"]["content"]
        return answer, final_passages
