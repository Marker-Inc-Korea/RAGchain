from typing import List, Callable

from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.reranker.base import BaseReranker
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import set_api_base


class RerankLLM(BaseLLM):
    def __init__(self, retrieval: BaseRetrieval, reranker: BaseReranker, model_name: str = "gpt-3.5-turbo",
                 api_base: str = None, retrieve_size: int = 10, use_passage_count: int = 3,
                 window_size: int = 10,
                 prompt_func: Callable[[List[Passage], str], List[dict]] = None,
                 stream_func: Callable[[str], None] = None,
                 *args, **kwargs):
        super().__init__(retrieval)
        self.stream_func = stream_func
        self.reranker = reranker
        self.model_name = model_name
        set_api_base(api_base)
        assert (retrieve_size > use_passage_count)
        assert (retrieve_size > 0)
        assert (use_passage_count > 0)
        assert (window_size > 0)
        self.retrieve_size = retrieve_size
        self.use_passage_count = use_passage_count
        self.window_size = window_size
        self.get_message = BasicLLM.get_messages if prompt_func is None else prompt_func

    def ask(self, query: str, stream: bool = False, run_retrieve: bool = True, *args, **kwargs) -> tuple[
        str, List[Passage]]:
        passages = self.retrieved_passages if len(
            self.retrieved_passages) > 0 and not run_retrieve else self.retrieval.retrieve(query,
                                                                                           top_k=self.retrieve_size)
        if self.retrieve_size <= self.window_size:
            reranked_passages = self.reranker.rerank(query, passages)
        else:
            reranked_passages = self.reranker.rerank_sliding_window(query, passages, self.window_size)
        final_passages = reranked_passages[:self.use_passage_count]
        answer = self.generate_chat(messages=self.get_message(final_passages, query),
                                    model=self.model_name,
                                    stream=stream,
                                    stream_func=self.stream_func,
                                    *args, **kwargs)
        return answer, final_passages
