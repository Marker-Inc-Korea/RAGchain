from typing import Optional, Union
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

from RAGchain.benchmark.answer.metrics import *
from RAGchain.benchmark.retrieval.metrics import BaseRetrievalMetric, AP, NDCG, CG, IndDCG, DCG, IndIDCG, IDCG, \
    Recall, Precision, RR, Hole, TopKAccuracy, ExactlyMatch, F1
from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage
from RAGchain.utils.util import text_modifier


class BaseEvaluator(ABC):
    retrieval_gt_metrics = ['Hole', 'TopK_Accuracy', 'EM', 'F1_score', 'context_recall', 'Recall', 'Precision']
    retrieval_gt_metrics_rank_aware = ['AP', 'NDCG', 'CG', 'Ind_DCG', 'DCG', 'Ind_IDCG', 'IDCG', 'RR']
    retrieval_no_gt_metrics = ['context_precision']
    answer_gt_metrics = ['BLEU']
    answer_no_gt_metrics = ['answer_relevancy', 'faithfulness']

    def __init__(self, run_all: bool = True, metrics: Optional[List[str]] = None):
        if run_all:
            self.metrics = self.retrieval_gt_metrics + self.retrieval_gt_metrics_rank_aware + \
                           self.retrieval_no_gt_metrics + self.answer_gt_metrics + self.answer_no_gt_metrics
        else:
            if metrics is None:
                raise ValueError("If run_all is False, metrics should be given")
            self.metrics = metrics

    @abstractmethod
    def evaluate(self, **kwargs) -> EvaluateResult:
        """
        Evaluate metrics and return the results
        :param kwargs: Arguments for running pipeline.run()
        :return: EvaluateResult
        """
        pass

    def _calculate_metrics(self,
                           questions: List[str],
                           pipeline: BasePipeline,
                           retrieval_gt: Optional[List[List[Union[str, UUID]]]] = None,
                           retrieval_gt_order: Optional[List[List[int]]] = None,
                           answer_gt: Optional[List[List[str]]] = None,
                           **kwargs
                           ) -> EvaluateResult:
        """
        Calculate metrics for a list of questions and return their results
        :param questions: List of questions
        :param pipeline: Pipeline to run
        :param retrieval_gt: Ground truth for retrieval
        :param retrieval_gt_order: Ground truth for retrieval rates
        :param answer_gt: Ground truth for answer. 2d list because it can evaluate multiple ground truth answers.
        :param kwargs: Arguments for pipeline.run()
        """
        result_df = {'question': questions}
        if retrieval_gt is not None:
            result_df['retrieval_gt'] = retrieval_gt
        if retrieval_gt_order is not None:
            result_df['retrieval_gt_order'] = retrieval_gt_order
        if answer_gt is not None:
            result_df['answer_gt'] = answer_gt
        result_df = pd.DataFrame(result_df)

        answers, passages = self._run_pipeline(result_df['question'].tolist(), pipeline, **kwargs)
        # TODO: Replace this to real rel scores Issue/#279
        scores = [[1.0 for _ in range(len(passage_group))] for passage_group in passages]
        k = len(passages[0])

        result_df['answer_pred'] = answers
        result_df['passage_ids'] = [[passage.id for passage in passage_group] for passage_group in passages]
        result_df['passage_contents'] = [[passage.content for passage in passage_group] for passage_group in passages]
        result_df['passage_scores'] = scores

        use_metrics = []

        # without gt - retrieval & answer
        ragas_metrics = self.__ragas_metrics()
        if len(ragas_metrics) > 0:
            # You can't use context_recall when retrieval_gt is None
            if retrieval_gt is None:
                ragas_metrics = [metric for metric in ragas_metrics if
                                 isinstance(metric, type(context_recall)) is False]
            use_metrics += [metric.name for metric in ragas_metrics]

            dataset_dict = {
                'question': result_df['question'].tolist(),
                'answer': result_df['answer_pred'].tolist(),
                'contexts': result_df['passage_contents'].tolist()
            }
            if retrieval_gt is not None:
                dataset_dict['ground_truths'] = self.__fetch_contents(retrieval_gt)

            ragas_result = evaluate(
                Dataset.from_dict(dataset_dict),
                metrics=ragas_metrics
            )
            ragas_result_df = ragas_result.to_pandas()
            assert ragas_result_df.iloc[0]['question'] == result_df.iloc[0]['question']
            assert ragas_result_df.iloc[0]['answer'] == result_df.iloc[0]['answer_pred']

            result_df = pd.concat([result_df, ragas_result_df[[metric.name for metric in ragas_metrics]]], axis=1)

        # with gt - retrieval
        def calculate_retrieval_metrics_pd(row, metric: BaseRetrievalMetric):
            pred = {str(_id): score for _id, score in zip(row['passage_ids'], row['passage_scores'])}
            gt_ids = self.uuid_to_str(row['passage_ids'])
            if 'retrieval_gt_order' not in row.axes[0]:
                solution = {str(_id): len(gt_ids) - i for i, _id in enumerate(gt_ids)}
            else:
                solution = {str(_id): rank for _id, rank in zip(gt_ids, row['retrieval_gt_order'])}
            result = metric.eval(solution, pred, k=len(pred))
            return result

        if retrieval_gt is not None:
            retrieval_metrics_with_gt = self.__retrieval_metrics_with_gt(rank_aware=(retrieval_gt_order is not None))
            use_metrics += [metric.metric_name for metric in retrieval_metrics_with_gt]
            # column name이 metric.metric_name
            for metric in retrieval_metrics_with_gt:
                result_df[metric.metric_name] = result_df.apply(
                    lambda row: calculate_retrieval_metrics_pd(row, metric), axis=1)

        # with gt - answer
        if answer_gt is not None:
            answer_gt_metrics = self.__answer_metrics_with_gt()
            use_metrics += [metric.metric_name for metric in answer_gt_metrics]
            # column name이 metric.metric_name
            for metric in answer_gt_metrics:
                result_df[metric.metric_name] = result_df.apply(
                    lambda row: metric.eval(row['answer_gt'], row['answer_pred']), axis=1)

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
        binary_metrics = {metric_names: metric for metric in
                          [TopKAccuracy(), ExactlyMatch(), F1(), Hole(), Recall(), Precision()]
                          for metric_names in text_modifier(metric.metric_name)}
        rank_aware_metrics = {metric_names: metric for metric in
                              [AP(), NDCG(), CG(), IndDCG(), DCG(), IndIDCG(), IDCG(), RR()]
                              for metric_names in text_modifier(metric.metric_name)}

        result = [binary_metrics[metric_name] for metric_name in self.metrics if metric_name in binary_metrics]
        if rank_aware:
            result += [rank_aware_metrics[metric_name] for metric_name in self.metrics if
                       metric_name in rank_aware_metrics]

        return result

    def __ragas_metrics(self):
        ragas_metrics = {metric_names: metric for metric in [context_recall, context_precision, answer_relevancy,
                                                             faithfulness]
                         for metric_names in text_modifier(metric.name)}
        result = [ragas_metrics[metric_name] for metric_name in self.metrics if metric_name in ragas_metrics]
        return result

    def __answer_metrics_with_gt(self) -> List[BaseAnswerMetric]:
        answer_metrics = {metric_names: metric for metric in [BLEU()]
                          for metric_names in text_modifier(metric.metric_name)}
        result = [answer_metrics[metric_name] for metric_name in self.metrics if metric_name in answer_metrics]
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

            def delete(self, passages: List[Passage]):
                pass

        dummy_retrieval = DummyRetrieval()
        retrieval_gt_contents = []
        for passage_ids in ids:
            retrieval_gt_contents.append([passage.content for passage in dummy_retrieval.fetch_data(passage_ids)])
        return retrieval_gt_contents

    @staticmethod
    def uuid_to_str(id_list: List[Union[UUID, str]]) -> List[str]:
        return [str(_id) for _id in id_list]
