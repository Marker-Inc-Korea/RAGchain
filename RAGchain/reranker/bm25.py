from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage, RetrievalResult


class BM25Reranker(BaseReranker):
    """
    BM25Reranker class for reranker based on BM25.
    You can rerank the passages with BM25 scores .
    """

    def __init__(self, tokenizer_name: str = "gpt2", *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        retrieval_result = RetrievalResult(query=query, passages=passages, scores=[])
        result = self.invoke(retrieval_result)
        return result.passages

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        contents: List[str] = list(map(lambda x: x.content, input.passages))
        # tokenize content for bm25 instance
        tokenized_content = self.__tokenize(contents)
        # tokenize query
        tokenized_query = self.__tokenize([input.query])[0]
        bm25 = BM25Okapi(tokenized_content)
        scores = bm25.get_scores(tokenized_query)
        sorted_pairs = sorted(zip(input.passages, scores), key=lambda x: x[1], reverse=True)
        sorted_passages, sorted_scores = list(zip(*sorted_pairs))
        input.passages = sorted_passages
        input.scores = sorted_scores
        return input

    def __tokenize(self, values: List[str]):
        tokenized = self.tokenizer(values)
        return tokenized.input_ids
