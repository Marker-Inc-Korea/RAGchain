from copy import deepcopy
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
    MSMARCO is a class for evaluating pipeline performance on MSMARCO dataset.
    """

    def __init__(self, run_pipeline: BasePipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        We are using train set for evaluating in this class, so it is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports MSMARCO dataset.
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
        'context_precision', 'answer_relevancy', 'faithfulness'.
        Rank aware metrics are 'NDCG', 'AP', 'CG', 'IndDCG', 'DCG', 'IndIDCG', 'IDCG', 'RR'.
        You must ingest all data for using context_recall and context_precision metrics.
        """

        self.file_path = "ms_marco"
        # You can available MSMARCO dataset versions v1.1 and v2.1
        self.dataset = load_dataset(self.file_path, 'v1.1')

        support_metrics = ['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
                           'context_precision', 'answer_relevancy', 'faithfulness',
                           'NDCG', 'AP', 'CG', 'IndDCG', 'DCG', 'IndIDCG', 'IDCG', 'RR']
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)

        self.eval_size = evaluate_size
        self.run_pipeline = run_pipeline
        self.retrieval_gt_lst = []
        self.retrieval_gt_ord_lst = []
        self.data = self.dataset['test']
        self.qa_data = pd.DataFrame(
            {'query_id': self.data['query_id'], 'question': self.data['query'], 'passages': self.data['passages'],
             'answers': self.data['answers']})

        # Remove none answers and is_selected list elements are all 0 and none answer rows.
        self.qa_data = self.qa_data.loc[
            (self.qa_data['answers'].map(lambda x: len(x)) != 0) & (
                    self.qa_data['passages'].map(lambda x: sum(x['is_selected'])) != 0)
            ].reset_index(drop=True)

        self.for_passages = deepcopy(self.qa_data)

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If you want to use context_recall and context_precision metrics, you should ingest all data.
        """

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                make_passages = pd.concat(
                    [self.for_passages['query_id'], json_normalize(self.for_passages['passages'].tolist())],
                    axis=1)[:ingest_size]

                # Slice for_passages for test code.
                self.for_passages = self.for_passages[:ingest_size]

            else:
                raise ValueError("ingest size must be same or larger than evaluate size")
        else:
            make_passages = pd.concat(
                [self.for_passages['query_id'], json_normalize(self.for_passages['passages'].tolist())],
                axis=1)

        # Evaluating passages must be in the ingesting passages.
        # self.qa_data['passages'][:ingest_size] is evaluating passages.
        # Assertion test if evaluating passages are in the ingesting passages.
        self.qa_data = self.for_passages[:ingest_size]
        for idx, passages in enumerate(self.for_passages['passages']):
            assert passages['passage_text'] == make_passages['passage_text'][idx]

        # Make passages.
        make_passages['passages'] = make_passages.apply(self.__make_passages_and_retrieval_gt, axis=1)
        passages = [passage for lst_passage in make_passages['passages'] for passage in lst_passage]

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        if self.eval_size is not None and len(self.qa_data) > self.eval_size:
            self.qa_data = self.qa_data[:self.eval_size]
        # else case is qa_data sliced by ingest size.

        return self._calculate_metrics(
            questions=self.qa_data['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=self.retrieval_gt_lst[:self.eval_size],
            retrieval_gt_order=self.retrieval_gt_ord_lst[:self.eval_size],
            **kwargs
        )

    def __make_passages_and_retrieval_gt(self, row):
        passages = []
        tmp_gt = []
        tmp_ord = []
        ord_rank = 0
        for passage_idx, passage_text in enumerate(row['passage_text']):
            passages.append(Passage(
                id=str(row['query_id']) + '_' + str(passage_idx),
                content=passage_text,
                filepath=self.file_path,
                metadata_etc={
                    'url': row['url'][passage_idx]
                }
            ))
            # Make retrieval gt and retrieval gt order.(27 is max count of passage texts in v2.1)
            if row['is_selected'][passage_idx] == 1:
                tmp_gt.append(str(row['query_id']) + '_' + str(passage_idx))
                tmp_ord.append(27 - ord_rank)
                ord_rank += 1
        self.retrieval_gt_lst.append(tmp_gt)
        self.retrieval_gt_ord_lst.append(tmp_ord)
        return passages
