from copy import deepcopy
from typing import List, Optional

import pandas as pd
from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class MsMarcoEvaluator(BaseDatasetEvaluator):
    """
    StrategyQAEvaluator is a class for evaluating pipeline performance on StrategyQA dataset.
    """

    file_path = "ms_marco"
    dataset = load_dataset(file_path, 'v1.1')

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
        support_metrics = ['Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall',
                           'context_precision', 'answer_relevancy', 'faithfulness']
        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics
        super().__init__(run_all=False, metrics=using_metrics)

        self.eval_size = evaluate_size  # To slice retrieval gt
        self.run_pipeline = run_pipeline

        self.data = self.dataset['test']
        query_id = self.data['query_id']
        question = self.data['query']
        passages = self.data['passages']
        answer = self.data['answers']
        self.for_passage = {'query_id': query_id, 'question': question, 'passages': passages, 'answer': answer}
        tmp = deepcopy(self.for_passage)

        # Remove non answer.
        idx_cnt = 0
        for detector_non_answer_idx in range(len(answer)):
            if len(self.for_passage['answer'][detector_non_answer_idx]) == 0:
                for key in tmp:
                    tmp[key].pop(detector_non_answer_idx - idx_cnt)
                idx_cnt += 1

        # Remove is_selected list elements are all 0.
        idx_cnt = 0
        cnt = 0
        for detector_is_selected_zero_idx in range(len(tmp['passages'])):
            for element in tmp['passages'][detector_is_selected_zero_idx - idx_cnt]['is_selected']:
                if element == 1:
                    cnt += 1
                else:
                    continue

            if cnt == 0:
                for key in tmp:
                    tmp[key].pop(detector_is_selected_zero_idx - idx_cnt)
                idx_cnt += 1
            else:
                cnt = 0

        self.for_passage = tmp

        self.retrieval_gt_lst = []

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
        passages = self.__make_passages_and_retrieval_gt()
        if ingest_size is not None:
            passages = passages[:ingest_size]
        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:

        df = self.qa_data
        retrieval_gt_lst = self.retrieval_gt_lst[:self.eval_size]

        return self._calculate_metrics(
            questions=df['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt_lst,
            **kwargs
        )

    def __make_passages_and_retrieval_gt(self):

        passages = []
        for idx in range(len(self.for_passage['passages'])):
            for passage_idx in range(len(self.for_passage['passages'][idx]['url'])):
                passages.append(Passage(
                    id=str(self.for_passage['query_id'][idx]) + '_' + str(passage_idx),
                    content=self.for_passage['passages'][idx]['passage_text'][passage_idx],
                    filepath=self.file_path,
                    metadata_etc={'url': self.for_passage['passages'][idx]['url'][passage_idx]}
                ))

                # Create retrieval gt list with 'is_selected' based on 'answer'.('is_selected' is list if passages were used to formulate and answer(is_selected:1))
                # It is appended id that is 'is_selected' status 1
                if self.for_passage['passages'][idx]['is_selected'][passage_idx] == 1:
                    self.retrieval_gt_lst.append([str(self.for_passage['query_id'][idx]) + '_' + str(passage_idx)])

        return passages
