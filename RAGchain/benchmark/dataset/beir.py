from typing import List, Optional

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseBeirEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult


class BeirEvaluator(BaseBeirEvaluator):
    """
    BeirEvaluator is a class for evaluating pipeline performance on beir datasets.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 file_name: str = None,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        # Create support metrics
        super().__init__(run_pipeline=run_pipeline, evaluate_size=evaluate_size, file_name=file_name, metrics=metrics)

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None, random_state=None):
        super().ingest_data(retrievals=retrievals, db=db, ingest_size=ingest_size, random_state=random_state)

    def evaluate(self, **kwargs) -> EvaluateResult:
        """
        Evaluate pipeline performance on fever dataset.
        This method always validate passages.
        """

        return self._calculate_metrics(
            questions=self.questions,
            pipeline=self.run_pipeline,
            retrieval_gt=self.retrieval_gt
        )
