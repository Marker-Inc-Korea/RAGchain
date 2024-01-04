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

        self.file_path = "NomaDamas/split_search_qa"
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

        check_isnull_gt = self.qa_data[('doc_id')].isnull().sum()
        assert check_isnull_gt == 0, "There are null values in retrieval_gt column."

        corpus = deepcopy(self.corpus)
        gt_ingestion = list(itertools.chain.from_iterable(self.qa_data['doc_id'].tolist()))

        self._validate_eval_size_and_ingest_size(ingest_size, eval_size=len(self.qa_data))

        # Convert retrieval ground truth dataframe to passages.
        gt_df = corpus[corpus['doc_id'].isin(gt_ingestion)]
        gt_passages = gt_df.apply(self.__make_passages, axis=1).tolist()

        # Shuffle corpus and slice it according to the ingest size.
        if ingest_size is not None:
            corpus = corpus.sample(n=ingest_size, replace=False, random_state=random_state,
                                   axis=0)

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = corpus.isin(gt_ingestion)
        # Remove duplicated passages
        ingest_data = corpus[~mask.any(axis=1)]

        # Create passages.
        passages = ingest_data.apply(self.__make_passages, axis=1).tolist()
        passages += gt_passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        question = self.qa_data['question'].tolist()
        retrieval_gt = self.qa_data.apply(lambda row: list(map(lambda x: uuid.UUID(x), row['doc_id'])),
                                          axis=1).tolist()
        answer_gt = self.qa_data['answer'].apply(lambda row: [row]).tolist()

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
