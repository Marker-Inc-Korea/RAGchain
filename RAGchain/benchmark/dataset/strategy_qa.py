import json
from typing import List, Optional

from huggingface_hub import hf_hub_download

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator, BaseStrategyQA
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class StrategyQAEvaluator(BaseDatasetEvaluator, BaseStrategyQA):
    dataset_name = "voidful/StrategyQA"

    def __init__(self, run_pipeline: BasePipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        We are using train set for evaluating in this class, so it is huge. Recommend to set proper size for evaluation.
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
                                         filename="strategyqa_train_paragraphs.json",
                                         repo_type="dataset")
        with open(paragraph_path, "r", encoding="utf-8") as f:
            self.paragraphs = json.load(f)

        test_data_path = hf_hub_download(repo_id=self.dataset_name,
                                         filename="strategyqa_train.json",
                                         repo_type="dataset")
        with open(test_data_path, "r", encoding="utf-8") as f:
            self.qa_data = json.load(f)
        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If you want to use context_recall and context_precision metrics, you should ingest all data.
        """
        passages = self.__make_paragraph_passages()
        if ingest_size is not None:
            passages = passages[:ingest_size]
        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self) -> EvaluateResult:
        qa_data_dict = {x['qid']: {'answer': x['answer'], 'question': x['question'], 'evidence': x['evidence']}
                        for x in self.qa_data}
        df = self.convert_qa_to_pd(qa_data_dict)
        return self._calculate_metrics(
            questions=df['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=df['evidence'].tolist(),
        )

    def __make_paragraph_passages(self):
        passages = []
        for key, value in self.paragraphs.items():
            passages.append(Passage(
                id=key,
                content=value['content'],
                filepath=self.dataset_name,
                metadata_etc={'title': value['title']}
            ))
        return passages
