from typing import List

from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class BM25Reranker(BaseReranker):
    """
    BM25Reranker is a reranker based on BM25.
    You can rerank the passages with the instruction using BM25Reranker.
    """

    def __init__(self, tokenizer_name: str = "gpt2", *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        contents: List[str] = [passage.content for passage in passages]

        # tokenize content for bm25 instance
        tokenized_content: List[str] = [self.__tokenize([content])[0] for content in tqdm(contents)]

        # tokenize query
        tokenized_query = self.__tokenize([query])[0]

        bm25 = BM25Okapi(tokenized_content)

        scores = bm25.get_scores(tokenized_query)
        sorted_pairs = sorted(zip(contents, scores), key=lambda x: x[1], reverse=True)
        sorted_content = [content for content, score in sorted_pairs]

        sorted_passages = []
        # Convert List[str] to List[Passage]
        for content in sorted_content:
            for passage in passages:
                if content == passage.content:
                    sorted_passages.append(passage)

        return sorted_passages

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("BM25Reranker doesn't support sliding window reranking.")

    def __tokenize(self, values: List[str]):
        tokenized = self.tokenizer(values)
        return tokenized.input_ids
