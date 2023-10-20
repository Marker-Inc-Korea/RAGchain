import json
from itertools import islice
from typing import Optional, List

import pandas as pd
from huggingface_hub import hf_hub_download

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator, BaseStrategyQA
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class KoStrategyQAEvaluator(BaseDatasetEvaluator, BaseStrategyQA):
    """
    Ko-StrategyQA dataset evaluator
    """
    dataset_name = "NomaDamas/Ko-StrategyQA"

    def __init__(self, run_pipeline: BasePipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        :param metrics: The list of metrics to use. If None, use all metrics that supports KoStrategyQA.
        Supporting metrics is Recall, Precision, Hole, TopK_Accuracy, EM, F1_score, context_recall, context_precision
        """
        support_metrics = ['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
                           'context_precision']
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)
        self.run_pipeline = run_pipeline
        paragraph_path = hf_hub_download(repo_id=self.dataset_name,
                                         filename="ko-strategy-qa_paragraphs.parquet",
                                         repo_type="dataset")
        self.paragraphs = pd.read_parquet(paragraph_path)
        dev_data_path = hf_hub_download(repo_id=self.dataset_name,
                                        filename="ko-strategy-qa_dev.json",
                                        repo_type="dataset")
        with open(dev_data_path, "r", encoding="utf-8") as f:
            self.dev_data = json.load(f)
        if evaluate_size is not None:
            self.dev_data = self.__slice_data(self.dev_data, evaluate_size)

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB,
               ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If you want to use context_recall and context_precision metrics, you should ingest all data.
        """
        passages = self.paragraphs.apply(lambda x: Passage(
            id=x['key'],
            content=x['ko-content'],
            filepath=self.dataset_name,
            metadata_etc={'title': x['title']}
        ), axis=1).tolist()
        if ingest_size is not None:
            passages = passages[:ingest_size]
        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.save(passages)

    def evaluate(self) -> EvaluateResult:
        df = self.convert_qa_to_pd(self.dev_data)
        return self._calculate_metrics(
            questions=df['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=df['evidence'].tolist(),
        )

    def __slice_data(self, data, size):
        if len(data) <= size:
            return data
        return dict(islice(data.items(), size))
