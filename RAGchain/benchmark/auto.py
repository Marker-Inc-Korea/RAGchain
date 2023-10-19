from typing import List

from RAGchain.benchmark.base import BaseEvaluator
from RAGchain.pipeline.base import BasePipeline
from RAGchain.schema import EvaluateResult


class AutoEvaluator(BaseEvaluator):
    """
    Evaluate metrics without ground truths. You only need to pass questions and your pipeline.
    You have to ingest properly to retrievals and DBs. Recommend to use IngestPipeline to ingest.
    """

    def __init__(self, pipeline: BasePipeline, questions: List[str], metrics=None):
        usable_metrics = ['context_precision', 'answer_relevancy', 'faithfulness']
        if metrics is None:
            metrics = usable_metrics
        metrics = [metric for metric in metrics if metric in usable_metrics]

        super().__init__(run_all=False, metrics=metrics)
        self.pipeline = pipeline
        self.questions = questions

    def evaluate(self) -> EvaluateResult:
        return self._calculate_metrics(questions=self.questions, pipeline=self.pipeline)
