from typing import List

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class BM25Reranker(BaseReranker):
    '''
    BM25Reranker is a reranker based on BM25.
    You can rerank the passages with the instruction using BM25Reranker.
    '''

    def __init__(self, save_path: str, tokenizer_name: str = "gpt2", *args, **kwargs):
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        contents: List[str] = [passage.content for passage in passages]

        bm25 = BM25Okapi(contents)
        tokenized = self.tokenizer([query]).input_ids
        tokenized_query = tokenized[0]

        scores = bm25.get_scores(tokenized_query)
        sorted_scores = sorted(scores, reverse=True)

        sorted_pairs = sorted(zip(contents, sorted_scores), key=lambda x: x[1], reverse=True)
        sorted_passages = [passage for passage, score in sorted_pairs]
        return sorted_passages

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("BM25Reranker doesn't support sliding window reranking.")
