import uuid
from copy import deepcopy
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class DUReaderRobustEvaluator(BaseDatasetEvaluator):
    """
    DUReaderRobustEvaluator is a class for evaluating pipeline performance on dureader robust dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None,
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        natural qa dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports natural qa dataset.
        Supporting metrics are 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'Recall', 'Precision'
        'context_recall', 'context_precision', 'BLEU', 'answer_relevancy', 'faithfulness', 'KF1'.
        You must ingest all data for using context_recall and context_precision metrics.

        Notice:
        Default metrics are essentially the metrics run when executing a test file.
        Support metrics refer to the available metrics.
        This distinction arises due to the prolonged evaluation time required for Ragas metrics.

        cf)
        Dataset link is below.
        huggingface: https://huggingface.co/datasets/PaddlePaddle/dureader_robust
        github: https://github.com/baidu/DuReader#dureader-robust-model-robustness
        paper: https://arxiv.org/abs/2004.11142
        """

        self.file_path = "PaddlePaddle/dureader_robust"
        self.dataset = load_dataset(self.file_path)['validation'].to_pandas()

        default_metrics = self.retrieval_gt_metrics + self.answer_gt_metrics \
                          + self.answer_no_gt_ragas_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_gt_ragas_metrics + self.retrieval_no_gt_ragas_metrics

        if metrics is not None:
            # Check if your metrics are available in evaluation datasets.
            for metric in metrics:
                if metric not in support_metrics:
                    raise ValueError(f"You input {metric} that this dataset evaluator not support.")
            using_metrics = list(set(metrics))
        else:
            using_metrics = default_metrics

        super().__init__(run_all=False, metrics=using_metrics)

        self.eval_size = evaluate_size
        self.run_pipeline = run_pipeline

        self.context = deepcopy(self.dataset[['id', 'context']])

        if evaluate_size is not None and len(self.dataset) > evaluate_size:
            self.qa_data = self.dataset[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If you want to use context_recall and context_precision metrics, you should ingest all data.
        """

        ingest_data = self.context
        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data[:ingest_size]
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Create passages.
        result = ingest_data.apply(self.__make_passages, axis=1)
        passages = result.tolist()

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        question = self.qa_data['question'].tolist()
        retrieval_gt = list([uuid.UUID(gt)] for gt in self.qa_data['id'])
        answer_gt = [[answer for answer in answer_arr['text']] for answer_arr in self.qa_data['answers']]

        return self._calculate_metrics(
            questions=question,
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt,
            answer_gt=answer_gt,
            **kwargs
        )

    def __make_passages(self, row):
        passage = Passage(
            id=row['id'],
            content=row['context'],
            filepath=self.file_path,
            metadata_etc={
                'dataset url': 'https://huggingface.co/datasets/PaddlePaddle/dureader_robust/viewer/plain_text/validation'
            })
        return passage
