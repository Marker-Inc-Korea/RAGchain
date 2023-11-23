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
        """
        :param pipeline: pipeline to evaluate
        :param questions: questions to evaluate
        :param metrics: metrics to evaluate. Default is None. If None, evaluate all supporting metrics.
        Supported metrics are 'context_precision', 'answer_relevancy', 'faithfulness'.
        """
        usable_metrics = self.retrieval_no_gt_metrics + self.answer_no_gt_metrics
        if metrics is None:
            metrics = usable_metrics
        metrics = [metric for metric in metrics if metric in usable_metrics]

        super().__init__(run_all=False, metrics=metrics)
        self.pipeline = pipeline
        self.questions = questions

    def evaluate(self, **kwargs) -> EvaluateResult:
        return self._calculate_metrics(questions=self.questions, pipeline=self.pipeline,
                                       validate_passages=False, **kwargs)
