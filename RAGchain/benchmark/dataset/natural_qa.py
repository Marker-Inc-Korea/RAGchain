import itertools
import uuid
from copy import deepcopy
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class NATURALQAEvaluator(BaseDatasetEvaluator):
    """
    NATURALQAEvaluator is a class for evaluating pipeline performance on natural qa dataset.
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
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
        'context_precision', 'answer_relevancy', 'faithfulness'.
        Rank aware metrics are 'NDCG', 'AP', 'CG', 'IndDCG', 'DCG', 'IndIDCG', 'IDCG', 'RR'.
        You must ingest all data for using context_recall and context_precision metrics.

        Notice:
        Raw natural qa dataset is too large, so we use natural questions short qa by lucadiliello generated at mrqa 2021 in huggingface.
        """

        self.file_path = "lucadiliello/naturalquestionsshortqa"
        self.dataset = load_dataset(self.file_path)['validation'].to_pandas()

        default_metrics = (self.retrieval_gt_metrics + self.retrieval_gt_metrics_rank_aware
                           + self.answer_gt_metrics
                           + self.answer_no_gt_ragas_metrics + self.answer_passage_metrics)
        support_metrics = (self.retrieval_gt_metrics
                           + self.retrieval_gt_ragas_metrics + self.retrieval_no_gt_ragas_metrics
                           + self.answer_gt_metrics + self.answer_no_gt_ragas_metrics + self.answer_passage_metrics)

        if metrics is not None:
            # Check if your metrics are available in evaluation datasets.
            for metric in metrics:
                if metric not in support_metrics:
                    raise ValueError("You input metrics that this dataset evaluator not support.")
            using_metrics = list(set(metrics))
        else:
            using_metrics = default_metrics

        super().__init__(run_all=False, metrics=using_metrics)

        self.eval_size = evaluate_size
        self.run_pipeline = run_pipeline

        # Delete duplicated question - answer - retrieval gt
        self.dataset = self.dataset.groupby('question', as_index=False).agg(
            {'context': lambda x: list(x), 'key': lambda x: list(x),
             'answers': lambda x: list(x.tolist()), 'labels': lambda x: list(x)})

        # Convert key to uuid to retrieve passages' uuid
        self.context = deepcopy(self.dataset[['key', 'context', 'labels']])

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

        ingest_data = deepcopy(self.context)
        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data[:ingest_size]
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Create passages.
        result = ingest_data.apply(self.__make_passages, axis=1)
        passages = list(itertools.chain.from_iterable(result))

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        question = self.qa_data['question'].tolist()
        retrieval_gt = [[uuid.UUID(key) for key in key_lst] for key_lst in self.qa_data['key']]
        answer_gt = [[answer[0] for answer in answer_lst] for answer_lst in self.qa_data['answers']]

        return self._calculate_metrics(
            questions=question,
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt,
            answer_gt=answer_gt,
            **kwargs
        )

    def __make_passages(self, row):
        passages = []

        for idx in range(len(row['key'])):
            passages.append(Passage(
                id=row['key'][idx],
                content=row['context'][idx],
                filepath=self.file_path,
                metadata_etc={
                    'url': row['labels'][idx]
                }
            ))
        return passages
