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


class SearchQAEvaluator(BaseDatasetEvaluator):
    """
    SearchQAEvaluator is a class for evaluating pipeline performance on search qa dataset.
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
        You must ingest all data for using context_recall metrics.

        Notice:
        Default metrics are essentially the metrics run when executing a test file.
        Support metrics refer to the available metrics.
        This distinction arises due to the prolonged evaluation time required for Ragas metrics.
        """

        self.file_path = "NomaDamas/search_qa_split"
        self.qa_data = load_dataset(self.file_path, 'qa_data')['test'].to_pandas()
        self.corpus = load_dataset(self.file_path, 'corpus')['train'].to_pandas()

        default_metrics = self.retrieval_gt_metrics + self.answer_gt_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_gt_ragas_metrics + self.retrieval_no_gt_ragas_metrics \
                          + self.answer_no_gt_ragas_metrics

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

        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None, random_state=None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If the ingest size is excessively large, it results in prolonged processing times.
        To address this, we shuffle the corpus and slice it according to the ingest size for testing purposes.
        The reason for transforming the retrieval ground truth corpus into passages and ingesting it is to enable
        retrieval to retrieve the retrieval ground truth within the database.
        This dataset has many retrieval ground truths per query, so it is recommended to set the ingest size to a small value.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        """

        ingest_data = deepcopy(self.corpus)
        gt_ingestion = list(itertools.chain.from_iterable(deepcopy([gt for gt in self.qa_data['retrieval_gt']])))

        # Retrieval ground truth ingestion
        gt_df = ingest_data[ingest_data['doc_id'].isin(gt_ingestion)]
        gt_passages = gt_df.apply(self.__make_passages, axis=1).tolist()

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data.sample(n=ingest_size, replace=False, random_state=random_state,
                                                 axis=0)
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = ingest_data.isin(gt_ingestion)
        # Remove duplicated passages
        ingest_data = ingest_data[~mask.any(axis=1)]

        # Create passages.
        passages = ingest_data.apply(self.__make_passages, axis=1).tolist()
        passages += gt_passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        question = self.qa_data['question'].tolist()
        retrieval_gt = [[uuid.UUID(gt) for gt in gt_lst] for gt_lst in self.qa_data['retrieval_gt']]
        answer_gt = [[answer] for answer in self.qa_data['answer']]

        return self._calculate_metrics(
            questions=question,
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt,
            answer_gt=answer_gt,
            **kwargs
        )

    def __make_passages(self, row):

        return Passage(
            id=row['doc_id'],
            content=row['snippets'],
                filepath=self.file_path,
                metadata_etc={
                    'air_date': row['air_date'],
                    'category': row['category'],
                    'value': row['value'],
                    'round': row['round'],
                    'show_number': row['show_number']
                }
        )

    def __make_doc_id(self, row):

        return str(uuid.uuid4())
