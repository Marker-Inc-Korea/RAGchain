from typing import List

from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class BM25Reranker(BaseReranker):
    """
    BM25Reranker class for reranker based on BM25.
    You can rerank the passages with BM25 scores .
    """

    def __init__(self, tokenizer_name: str = "gpt2", *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        contents: List[str] = [passage.content for passage in passages]

        tokenized_content: List[str] = []
        # tokenize content for bm25 instance
        for content in tqdm(contents):
            tokenized_content.append(self.__tokenize([content])[0])

        # tokenize query
        tokenized_query = self.__tokenize([query])[0]

        bm25 = BM25Okapi(tokenized_content)

        scores = bm25.get_scores(tokenized_query)

        sorted_pairs = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        sorted_passages = [passage for passage, _ in sorted_pairs]

        return sorted_passages

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("BM25Reranker doesn't support sliding window reranking.")

    def __tokenize(self, values: List[str]):
        tokenized = self.tokenizer(values)
        return tokenized.input_ids
