from abc import ABC, abstractmethod
from typing import List

import sacrebleu


class BaseAnswerMetric(ABC):
    def __init__(self):
        self._metric_name = None

    @property
    def metric_name(self):
        return str(self._metric_name)

    def eval(self, solutions: List[str],
             pred: str) -> float:
        """
        :param solutions: list of solutions. If you have only one ground truth answer, you can use [answer].
        :param pred: predicted answer
        """
        metric = self.retrieval_metric_function(solutions, pred)

        return metric

    def _normalizer_str(self, s: str) -> str:
        return s.lower().strip()

    @abstractmethod
    def retrieval_metric_function(self, solutions: List[str],
                                  pred: str) -> float:
        pass


class BLEU(BaseAnswerMetric):
    def __init__(self):
        super().__init__()
        self._metric_name = "BLEU"
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("Please install sacrebleu. pip install sacrebleu")

    def retrieval_metric_function(self, solutions: List[str], pred: str) -> float:
        score = sacrebleu.sentence_bleu(pred, solutions)
        return score.score
