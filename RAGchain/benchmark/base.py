from abc import ABC, abstractmethod
from typing import List, Optional, Union
from uuid import UUID

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from tqdm import tqdm

from RAGchain.benchmark.answer.metrics import BaseAnswerMetric
from RAGchain.benchmark.retrieval.metrics import BaseRetrievalMetric, AP, NDCG, CG, IndDCG, DCG, IndIDCG, IDCG, \
    Recall, Precision, RR, Hole, TopKAccuracy, ExactlyMatch, F1
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage
from RAGchain.utils.util import text_modifier


class BaseEvaluator(ABC):
    def __init__(self, run_all: bool = True, metrics: Optional[List[str]] = None):
        if run_all:
            self.metrics = ['AP', 'NDCG', 'CG', 'Ind_DCG', 'DCG', 'Ind_IDCG', 'IDCG', 'Recall', 'Precision', 'RR',
                            'Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall', 'context_precision',
                            'answer_relevancy', 'faithfulness']
        else:
            if metrics is None:
                raise ValueError("If run_all is False, metrics should be given")
            self.metrics = metrics

    @abstractmethod
    def evaluate(self) -> EvaluateResult:
        pass

    def _calculate_metrics(self,
                           questions: List[str],
                           pipeline: BasePipeline,
                           retrieval_gt: Optional[List[List[Union[str, UUID]]]] = None,
                           retrieval_gt_order: Optional[List[List[int]]] = None,
                           answer_gt: Optional[List[str]] = None,
                           ) -> EvaluateResult:
        """
        Calculate metrics for a list of questions and return their results
        """
        answers, passages = self._run_pipeline(questions, pipeline)
        # TODO: Replace this to real rel scores Issue/#279
        scores = [[1.0 for _ in range(len(passage_group))] for passage_group in passages]
        k = len(passages[0])

        df_temp = [
            [question, answer] +
            [passage.id for passage in passage_group] +
            [passage.content for passage in passage_group] +
            [score for score in score_group]
            for question, answer, passage_group, score_group in zip(questions, answers, passages, scores)
        ]

        passage_id_columns = [f'passage_id_{i + 1}' for i in range(k)]
        passage_content_columns = [f'passage_content_{i + 1}' for i in range(k)]
        passage_scores_columns = [f'passage_scores_{i + 1}' for i in range(k)]
        columns = ['question', 'answer'] + passage_id_columns + passage_content_columns + passage_scores_columns
        result_df = pd.DataFrame(df_temp, columns=columns)
        use_metrics = []

        # without gt - retrieval & answer
        ragas_metrics = self.__ragas_metrics()
        # You can't use context_recall when retrieval_gt is None
        if retrieval_gt is None:
            ragas_metrics = [metric for metric in ragas_metrics if isinstance(metric, type(context_recall)) is False]
        use_metrics += [metric.name for metric in ragas_metrics]

        dataset_dict = {
            'question': questions,
            'answer': answers,
            'contexts': result_df[passage_content_columns].values.tolist()
        }
        if retrieval_gt is not None:
            dataset_dict['ground_truths'] = self.__fetch_contents(retrieval_gt)

        ragas_result = evaluate(
            Dataset.from_dict(dataset_dict),
            metrics=ragas_metrics
        )
        ragas_result_df = ragas_result.to_pandas()
        assert ragas_result_df.iloc[0]['question'] == result_df.iloc[0]['question']
        assert ragas_result_df.iloc[0]['answer'] == result_df.iloc[0]['answer']

        result_df = pd.concat([result_df, ragas_result_df[[metric.name for metric in ragas_metrics]]], axis=1)

        # with gt - retrieval
        def calculate_retrieval_metrics_pd(row, index, metric: BaseRetrievalMetric):
            retrieved_ids = row[passage_id_columns].tolist()
            retrieved_scores = row[passage_scores_columns].tolist()
            pred = {str(_id): score for _id, score in zip(retrieved_ids, retrieved_scores)}

            gt_ids = self.uuid_to_str(retrieval_gt[index])
            if retrieval_gt_order is None:
                solution = {str(_id): len(gt_ids) - i for i, _id in enumerate(gt_ids)}
            else:
                solution = {str(_id): rank for _id, rank in zip(gt_ids, retrieval_gt_order[index])}
            return metric.eval(solution, pred, k=len(pred))

        if retrieval_gt is not None:
            retrieval_metrics_with_gt = self.__retrieval_metrics_with_gt(rank_aware=(retrieval_gt_order is not None))
            use_metrics += [metric.metric_name for metric in retrieval_metrics_with_gt]
            # column name이 metric.metric_name
            for metric in retrieval_metrics_with_gt:
                result_df[metric.metric_name] = result_df.apply(
                    lambda row: calculate_retrieval_metrics_pd(row, row.name, metric), axis=1)

        # with gt - answer
        # TODO: add this after making answer evaluation Issue/#123

        return EvaluateResult(
            results=result_df[use_metrics].mean().to_dict(),
            use_metrics=use_metrics,
            each_results=result_df
        )

    def _run_pipeline(self, questions: List[str], pipeline: BasePipeline, **kwargs) \
            -> tuple[List[str], List[List[Passage]]]:
        """
        Run the pipeline for a list of questions and return the results (answers, retrieval results)
        :param questions: List of questions
        :param pipeline: Pipeline to run
        :param kwargs: Arguments for pipeline.run()
        :return: Tuple of answers and retrieved passages
        """
        answers = []
        passages_result = []

        for question in tqdm(questions):
            answer, passages = pipeline.run(question, **kwargs)
            answers.append(answer)
            passages_result.append(passages)

        return answers, passages_result

    def __retrieval_metrics_with_gt(self, rank_aware: bool = False) -> List[BaseRetrievalMetric]:
        """
        Make a list of retrieval metrics from a list of metric names
        """
        binary_metrics = [TopKAccuracy(), ExactlyMatch(), F1(), Hole(), Recall(), Precision()]
        rank_aware_metrics = [AP(), NDCG(), CG(), IndDCG(), DCG(), IndIDCG(), IDCG(), RR(), ]
        result = []
        for metric_name in self.metrics:
            for rm in binary_metrics:
                if metric_name in text_modifier(rm.metric_name):
                    result.append(rm)
                    break
            if rank_aware:
                for rm in rank_aware_metrics:
                    if metric_name in text_modifier(rm.metric_name):
                        result.append(rm)
                        break

        return result

    def __ragas_metrics(self):
        ragas_metrics = [context_precision, context_recall, answer_relevancy, faithfulness]
        result = []
        for metric_name in self.metrics:
            for metric in ragas_metrics:
                if metric_name in text_modifier(metric.name):
                    result.append(metric)
                    break

        return result

    def __fetch_contents(self, ids: List[List[Union[str, UUID]]]) -> List[List[str]]:
        class DummyRetrieval(BaseRetrieval):
            def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
                pass

            def ingest(self, passages: List[Passage]):
                pass

            def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
                pass

            def retrieve_id_with_scores(self, query: str, top_k: int = 5, *args, **kwargs) -> tuple[
                List[Union[str, UUID]], List[float]]:
                pass

        dummy_retrieval = DummyRetrieval()
        retrieval_gt_contents = []
        for passage_ids in ids:
            retrieval_gt_contents.append([passage.content for passage in dummy_retrieval.fetch_data(passage_ids)])
        return retrieval_gt_contents

    def __answer_metrics_with_gt(self) -> List[BaseAnswerMetric]:
        # TODO: Implement this
        return []

    @staticmethod
    def uuid_to_str(id_list: List[Union[UUID, str]]) -> List[str]:
        return [str(_id) for _id in id_list]
