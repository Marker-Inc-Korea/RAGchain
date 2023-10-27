from typing import List, Optional

from RAGchain.llm.base import BaseLLM
from RAGchain.llm.basic import BasicLLM
from RAGchain.pipeline.base import BasePipeline
from RAGchain.reranker.base import BaseReranker
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage


class RerankRunPipeline(BasePipeline):
    """
    Rerank pipeline is for question answering with retrieved passages using reranker.
    Af first, retrieval module will retrieve retrieve_size passages for reranking.
    Then, reranker rerank passages and use use_passage_count passages for llm question.


    :example:
    >>> from RAGchain.pipeline.rerank import RerankRunPipeline
    >>> from RAGchain.retrieval import BM25Retrieval
    >>> from RAGchain.reranker import MonoT5Reranker
    >>> from RAGchain.llm.basic import BasicLLM

    >>> retrieval = BM25Retrieval(save_path="./bm25.pkl")
    >>> reranker = MonoT5Reranker()
    >>> llm = BasicLLM()
    >>> pipeline = RerankRunPipeline(retrieval, reranker, llm)
    >>> answer, passages = pipeline.run(query="What is the purpose of this framework based on the document?")
    >>> print(answer)

    """

    def __init__(self, retrieval: BaseRetrieval, reranker: BaseReranker, llm: Optional[BaseLLM] = None):
        """
        Initializes an instance of the RerankRunPipeline class.

        :param retrieval: An instance of the Retrieval module used for retrieving passages.
        :param reranker: An instance of the Reranker module used for reranking passages.
        :param llm: An instance of the LLM module used for generating answers. Default is BasicLLM.
        """
        self.retrieval = retrieval
        self.reranker = reranker
        self.llm = llm if llm is not None else BasicLLM()

    def run(self,
            query: str,
            retrieve_size: int = 50,
            use_passage_count: int = 4,
            *args, **kwargs) -> tuple[str, List[Passage]]:
        """
        :param query: question
        :param retrieve_size: The number of passages to reranking. Default is 50.
        :param use_passage_count: The number of passages to use for llm question after reranking. Default is 4.
        :param args: optional parameter for llm.ask()
        :param kwargs: optional parameter for llm.ask()

        :return answer: The answer to the question that llm generated.
        :return passages: The list of passages used to generate the answer.
        """

        retrieved_passages = self.retrieval.retrieve(query, top_k=retrieve_size)
        reranked_passages = self.reranker.rerank(query, retrieved_passages)
        final_passages = reranked_passages[:use_passage_count]
        return self.llm.ask(query, final_passages, *args, **kwargs)
