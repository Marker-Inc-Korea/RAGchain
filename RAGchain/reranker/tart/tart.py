from typing import List

import torch
import torch.nn.functional as F

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage
from .modeling_enc_t5 import EncT5ForSequenceClassification
from .tokenization_enc_t5 import EncT5Tokenizer


class TARTReranker(BaseReranker):
    """
    TARTReranker is a reranker based on TART (https://github.com/facebookresearch/tart).
    You can rerank the passages with the instruction using TARTReranker.
    """
    def __init__(self, instruction: str):
        """
        The default model is facebook/tart-full-flan-t5-xl.
        :param instruction: The instruction for reranking.
        """
        self.instruction = instruction
        model_name = "facebook/tart-full-flan-t5-xl"
        self.model = EncT5ForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = EncT5Tokenizer.from_pretrained(model_name)

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        contents: List[str] = [passage.content for passage in passages]
        instruction_queries: List[str] = ['{0} [SEP] {1}'.format(self.instruction, query) for _ in range(len(contents))]

        features = self.tokenizer(instruction_queries, contents, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**features).logits
            normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]

        sorted_pairs = sorted(zip(passages, normalized_scores), key=lambda x: x[1], reverse=True)
        sorted_passages = [passage for passage, score in sorted_pairs]
        return sorted_passages

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("TARTReranker does not support sliding window reranking.")
