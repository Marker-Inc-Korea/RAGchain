from typing import List, Callable

from pydantic import BaseModel, Field

from RAGchain.schema import Passage


class RetrievalResult(BaseModel):
    """class for storing retrieval result"""
    query: str
    """query string used for retrieval"""
    passages: List[Passage]
    """list of passage retrieved"""
    scores: List[float]
    """list of scores for each passage"""
    metadata: dict = Field(default_factory=dict)
    """metadata that you can store anything you want"""

    def to_prompt_input(self, passage_convert_func: Callable[[List[Passage]], str] = Passage.make_prompts) -> dict:
        return {
            "question": self.query,
            "passages": passage_convert_func(self.passages)
        }

    def to_dict(self):
        return {
            "query": self.query,
            "passages": self.passages,
            "scores": self.scores,
            "metadata": self.metadata
        }

    def slice(self, start: int = 0, end: int = None):
        """
        Slice passages and scores.
        :param start: int, start index of slice. Default is 0.
        :param end: int, end index of slice. Default is the length of passages.
        """
        if end is None:
            end = len(self.passages)
        self.passages = self.passages[start:end]
        self.scores = self.scores[start:end]
        return self
