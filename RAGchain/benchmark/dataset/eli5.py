from copy import deepcopy
from typing import List, Optional

from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class Eli5Evaluator(BaseDatasetEvaluator):
    """
    Eli5Evaluator is a class for evaluating pipeline performance on Eli5 dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        Eli5 dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports Eli5 dataset.
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score',
        'context_recall', 'context_precision', 'BLEU', 'answer_relevancy', 'faithfulness', 'KF1'.

        The dataset is collected from https://huggingface.co/datasets/Pakulski/ELI5-test.
        Because of the size of the dataset, we use the dataset that someone else has already processed.

        Notice:

        Default metrics is basically running metrics if you run test file.
        Support metrics is the metrics you are available.
        This separation is because Ragas metrics take a long time in evaluation.
        """

        self.file_path = "Pakulski/ELI5-test"

        datasets = load_dataset(self.file_path)['train'].to_pandas()

        default_metrics = self.retrieval_gt_metrics + self.answer_gt_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_gt_ragas_metrics, self.retrieval_no_gt_ragas_metrics \
                          + self.answer_no_gt_ragas_metrics

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

        # Preprocess dataset
        self.ingest_data = deepcopy(datasets[['id', 'document']])
        self.qa_data = datasets[['id', 'question', 'goldenAnswer']]

        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        """
        ingest_data = self.ingest_data
        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data[:ingest_size]
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Create passages.
        passages = ingest_data.apply(self.__make_passages, axis=1).tolist()

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:

        return self._calculate_metrics(
            questions=self.qa_data['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=[[gt] for gt in self.qa_data['id'].tolist()],
            answer_gt=[[answer] for answer in self.qa_data['goldenAnswer'].tolist()],
            **kwargs
        )

    def __make_passages(self, row):

        return Passage(
            id=row['id'],
            content=row['document'],
            filepath=self.file_path,
            metadata_etc={
                'dataset_creator': 'Pakulski',
                'dataset_url': 'https://huggingface.co/datasets/Pakulski/ELI5-test',
                'dataset_name': 'ELI5-test',
                'dataset_size': '1.3GB',
            }
        )
