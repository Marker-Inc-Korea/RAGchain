from typing import List, Callable

from RAGchain.llm.base import BaseLLM
from RAGchain.llm.basic import BasicLLM
from RAGchain.reranker.base import BaseReranker
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage
from RAGchain.utils.util import set_api_base


class RerankLLM(BaseLLM):
    """
    Rerank LLM module for question answering with retrieved passages using reranker.
    Af first, retrieval module will retrieve retrieve_size passages for reranking.
    Then, reranker rerank passages and use use_passage_count passages for llm question.

    :example:
    >>> from RAGchain.llm.rerank import RerankLLM
    >>> from RAGchain.retrieval import BM25Retrieval
    >>> from RAGchain.reranker import MonoT5Reranker

    >>> retrieval = BM25Retrieval(save_path="./bm25.pkl")
    >>> reranker = MonoT5Reranker()
    >>> llm = RerankLLM(retrieval=retrieval, reranker=reranker)
    >>> answer, passages = llm.ask("What is the purpose of this framework based on the document?")
    >>> print(answer)

    """
    def __init__(self, retrieval: BaseRetrieval, reranker: BaseReranker, model_name: str = "gpt-3.5-turbo",
                 api_base: str = None, retrieve_size: int = 10, use_passage_count: int = 3,
                 window_size: int = 10,
                 prompt_func: Callable[[List[Passage], str], List[dict]] = None,
                 stream_func: Callable[[str], None] = None,
                 *args, **kwargs):
        """
        Initializes an instance of the RerankLLM class.

        :param retrieval: An instance of the Retrieval module used for retrieving passages.
        :param reranker: An instance of the Reranker module used for reranking passages.
        :param model_name: The name or identifier of the llm model to be used. Default is "gpt-3.5-turbo".
        :param api_base: The base URL of the llm API endpoint. Default is None.
        :param retrieve_size: The number of passages to reranking. Default is 10.
        :param use_passage_count: The number of passages to use for llm question after reranking. Default is 3.
        :param window_size: The window size for sliding window reranking. The sliding window is used for LLMReranker.
        If retrieve_size is bigger than window_size, reranking will be done multiple times and do reranking with sliding window.
        If you don't want to use sliding window, set window_size to retrieve_size.
        Default is 10.
        :param prompt_func: A callable function used for generating prompts based on passages and user query. The input of prompt_func will be the list of retrieved passages and user query. The output of prompt_func should be a list of dictionaries with "role" and "content" keys, which is openai style chat prompts. Default is BasicLLM.get_messages.
        :param stream_func: A callable function used for streaming generated responses. You have to implement if you want to use stream. This stream_func will be called when the stream is received. Default is None.
        """

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
        """
        Ask a question to the LLM model and get answer and used passages.
        :param query: question
        :param stream: if stream is true, use stream feature. Default is False.
        :param run_retrieve: if run_retrieve is true, run retrieval module. If False, don't run retrieval module and use retrieved_passages instead. Default is True.
        :param args: optional parameter for llm api (openai style)
        :param kwargs: optional parameter for llm api (openai style)

        :return answer: The answer to the question that llm generated.
        :return passages: The list of passages used to generate the answer.
        """
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
