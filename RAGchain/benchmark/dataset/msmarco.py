from typing import List, Optional

import pandas as pd
from datasets import load_dataset
from pandas import json_normalize

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class MSMARCOEvaluator(BaseDatasetEvaluator):
    """
    StrategyQAEvaluator is a class for evaluating pipeline performance on StrategyQA dataset.
    """

    def __init__(self, run_pipeline: BasePipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        We are using train set for evaluating in this class, so it is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports KoStrategyQA.
        Supporting metrics are Recall, Precision, Hole, TopK_Accuracy, EM, F1_score, context_recall, context_precision
        You must ingest all data for using context_recall and context_precision metrics.
        """

        self.file_path = "ms_marco"
        self.dataset = load_dataset(self.file_path, 'v1.1')

        support_metrics = ['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
                           'context_precision', 'answer_relevancy', 'faithfulness', 'NDCG']
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)

        self.eval_size = evaluate_size  # To slice retrieval gt
        self.run_pipeline = run_pipeline
        self.retrieval_gt_lst = []
        self.retrieval_gt_ord_lst = []
        self.data = self.dataset['test']
        self.for_passage = pd.DataFrame(
            {'query_id': self.data['query_id'], 'question': self.data['query'], 'passages': self.data['passages'],
             'answers': self.data['answers']})

        # Remove non answers and is_selected list elements are all 0 and non answer rows.
        self.for_passage = self.for_passage.loc[(self.for_passage['answers'].map(lambda x: len(x)) != 0) & (
                self.for_passage['passages'].map(lambda x: sum(x['is_selected'])) != 0)].reset_index(drop=True)


        self.msmarco = pd.DataFrame(self.for_passage, columns=['query_id', 'question', 'passages', 'answer'])
        if evaluate_size is not None and len(self.msmarco) > evaluate_size:
            self.qa_data = self.msmarco[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If you want to use context_recall and context_precision metrics, you should ingest all data.
        """

        make_passages = pd.concat([self.for_passage['query_id'], json_normalize(self.for_passage['passages'].tolist())],
                                  axis=1)
        make_passages['passages'] = make_passages.apply(self.__make_passages_and_retrieval_gt, axis=1)

        if ingest_size is not None:
            passages = [passage for lst_passage in make_passages['passages'][:ingest_size] for passage in lst_passage]

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:

        df = self.qa_data
        retrieval_gt_lst = self.retrieval_gt_lst[:self.eval_size]
        retrieval_gt_ord = self.retrieval_gt_ord_lst[:self.eval_size]
        return self._calculate_metrics(
            questions=df['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt_lst,
            retrieval_gt_order=retrieval_gt_ord,
            **kwargs
        )

    def __make_passages_and_retrieval_gt(self, row):
        passages = []
        tmp_gt = []
        tmp_ord = []
        ord_rank = 0
        for passage_idx in range(len(row['url'])):
            passages.append(Passage(
                id=str(row['query_id']) + '_' + str(passage_idx),
                content=row['passage_text'][passage_idx],
                filepath=self.file_path,
                metadata_etc={
                    'url': row['url'][passage_idx]}
            ))
            if row['is_selected'][passage_idx] == 1:
                tmp_gt.append(str(row['query_id']) + '_' + str(passage_idx))
                tmp_ord.append(len(row['url']) - ord_rank)
                ord_rank += 1
        self.retrieval_gt_lst.append(tmp_gt)
        self.retrieval_gt_ord_lst.append(tmp_ord)
        return passages
