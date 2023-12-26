import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

import evaluate
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

    def _normalize_answer(self, s):
        """
        Taken from the official evaluation script for v1.1 of the SQuAD dataset.
        Lower text and remove punctuation, articles and extra whitespace.
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _token_f1_score(self, prediction, ground_truth):
        """
        Taken from the official evaluation script for v1.1 of the SQuAD dataset.
        """
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


class BasePassageAnswerMetric(BaseAnswerMetric, ABC):
    def eval(self, knowledge: List[str], pred: str) -> float:
        """
        :param knowledge: list of knowledge. Generally it is ground truth passages for a question.
        :param pred: predicted answer
        """
        metric = self.retrieval_metric_function(knowledge, pred)
        return metric

    @abstractmethod
    def retrieval_metric_function(self, knowledge: List[str], pred: str) -> float:
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


class KF1(BasePassageAnswerMetric):
    def __init__(self):
        super().__init__()
        self._metric_name = "KF1"

    def retrieval_metric_function(self, knowledge: List[str], pred: str) -> float:
        score = self._token_f1_score(pred, "\n".join(knowledge))
        return score

class METEOR(BaseAnswerMetric):
    def __init__(self):
        super().__init__()
        self._metric_name = "METEOR"

    def retrieval_metric_function(self, solutions: List[str], pred: str) -> float:
        meteor = evaluate.load("meteor")
        score = 0.0
        for solution in solutions:
            score = max(meteor.compute(predictions=[pred], references=[solution])['meteor'], score)
        return score

class ROUGE(BaseAnswerMetric):
    def __init__(self):
        super().__init__()
        self._metric_name = "ROUGE"
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError("Please install rouge_scorer. pip install rouge_score")

    def retrieval_metric_function(self, solutions: List[str], pred: str) -> float:
        rouge = evaluate.load("rouge")
        score = 0.0
        for solution in solutions:
            score = max(rouge.compute(predictions=[pred], references=[solution])['rougeL'], score)
        print(score)
        return score
