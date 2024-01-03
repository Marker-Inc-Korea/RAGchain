import warnings
from typing import Optional, Union
from uuid import UUID

import pandas as pd
from datasets import Dataset

from RAGchain.benchmark.answer.metrics import *
from RAGchain.benchmark.retrieval.metrics import BaseRetrievalMetric, AP, NDCG, CG, IndDCG, DCG, IndIDCG, IDCG, \
    Recall, Precision, RR, Hole, TopKAccuracy, EM_retrieval, F1
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage
from RAGchain.utils.util import text_modifier


class BaseEvaluator(ABC):
    # retrieval_gt_ragas_metrics is retrieval gt metrics that use ragas evaluation.
    retrieval_gt_metrics = ['Hole', 'TopK_Accuracy', 'EM_retrieval', 'F1_score', 'Recall', 'Precision']
    retrieval_gt_ragas_metrics = ['context_recall']
    retrieval_gt_metrics_rank_aware = ['AP', 'NDCG', 'CG', 'Ind_DCG', 'DCG', 'Ind_IDCG', 'IDCG', 'RR']
    retrieval_no_gt_ragas_metrics = ['context_precision']
    answer_gt_metrics = ['BLEU', 'METEOR', 'ROUGE', 'EM_answer']
    answer_no_gt_ragas_metrics = ['answer_relevancy', 'faithfulness']
    answer_passage_metrics = ['KF1']

    def __init__(self, run_all: bool = True, metrics: Optional[List[str]] = None):
        if run_all:
            self.metrics = self.retrieval_gt_metrics + self.retrieval_gt_ragas_metrics + \
                           self.retrieval_gt_metrics_rank_aware + self.retrieval_no_gt_ragas_metrics + \
                           self.answer_gt_metrics + self.answer_no_gt_ragas_metrics + \
                           self.answer_passage_metrics
        else:
            if metrics is None:
                raise ValueError("If run_all is False, metrics should be given")
            self.metrics = metrics
        self.dummy_retrieval = DummyRetrieval()

    @abstractmethod
    def evaluate(self, validate_passages: bool = True) -> EvaluateResult:
        """
        Evaluate metrics and return the results
        :param validate_passages: If True, validate passages in retrieval_gt already ingested.
        If False, you can't use context_recall and KF1 metrics.
        We recommend to set True for robust evaluation.
        :return: EvaluateResult
        """
        pass

    def _calculate_metrics(self,
                           questions: List[str],
                           pipeline: BaseRunPipeline,
                           retrieval_gt: Optional[List[List[Union[str, UUID]]]] = None,
                           retrieval_gt_order: Optional[List[List[int]]] = None,
                           answer_gt: Optional[List[List[str]]] = None,
                           validate_passages: bool = True
                           ) -> EvaluateResult:
        """
        Calculate metrics for a list of questions and return their results
        :param questions: List of questions
        :param pipeline: Pipeline to run. Must be BaseRunPipeline
        :param retrieval_gt: Ground truth for retrieval
        :param retrieval_gt_order: Ground truth for retrieval rates
        :param answer_gt: Ground truth for answer. 2d list because it can evaluate multiple ground truth answers.
        :param validate_passages: If True, validate passages in retrieval_gt already ingested.
        You can't use KF1 and context_recall when this parameter is False.
        """
        result_df = {'question': questions}
        if retrieval_gt is not None:
            result_df['retrieval_gt'] = retrieval_gt

        if retrieval_gt_order is not None:
            result_df['retrieval_gt_order'] = retrieval_gt_order
        if answer_gt is not None:
            result_df['answer_gt'] = answer_gt
        result_df = pd.DataFrame(result_df)

        if validate_passages and retrieval_gt is not None:
            result_df = self.__validate_passages(result_df)

        answers, passages, rel_scores = pipeline.get_passages_and_run(result_df['question'].tolist())

        result_df['answer_pred'] = answers
        result_df['passage_ids'] = [[passage.id for passage in passage_group] for passage_group in passages]
        result_df['passage_contents'] = [[passage.content for passage in passage_group] for passage_group in passages]
        result_df['passage_scores'] = rel_scores

        use_metrics = []

        # without gt - retrieval & answer
        ragas_metrics = self.__ragas_metrics()
        if len(ragas_metrics) > 0:
            from ragas.metrics import context_recall
            # You can't use context_recall when retrieval_gt is None or don't validate passages.
            if retrieval_gt is None or 'retrieval_gt_contents' not in result_df.columns:
                ragas_metrics = [metric for metric in ragas_metrics if
                                 isinstance(metric, type(context_recall)) is False]
            else:
                warnings.warn("You can't use context_recall when retrieval_gt is None or don't validate passages.")
        if len(ragas_metrics) > 0:
            from ragas import evaluate
            use_metrics += [metric.name for metric in ragas_metrics]

            dataset_dict = {
                'question': result_df['question'].tolist(),
                'answer': result_df['answer_pred'].tolist(),
                'contexts': result_df['passage_contents'].tolist()
            }
            if retrieval_gt is not None and 'retrieval_gt_contents' in result_df.columns:
                dataset_dict['ground_truths'] = result_df['retrieval_gt_contents'].tolist()

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
            gt_ids = self.uuid_to_str(row['retrieval_gt'])
            if 'retrieval_gt_order' not in row.axes[0]:
                solution = {str(_id): len(gt_ids) - i for i, _id in enumerate(gt_ids)}
            else:
                solution = {str(_id): rank for _id, rank in zip(gt_ids, row['retrieval_gt_order'])}
            result = metric.eval(solution, pred, k=len(pred))
            return result

        if retrieval_gt is not None:
            retrieval_metrics_with_gt = self.__retrieval_metrics_with_gt(rank_aware=(retrieval_gt_order is not None))
            use_metrics += [metric.metric_name for metric in retrieval_metrics_with_gt]
            for metric in retrieval_metrics_with_gt:
                result_df[metric.metric_name] = result_df.apply(
                    lambda row: calculate_retrieval_metrics_pd(row, metric), axis=1)

            # answer metric compare with retrieval ground truth knowledge
            answer_passage_metrics = self.__answer_passage_metrics()
            if len(answer_passage_metrics) > 0 and 'retrieval_gt_contents' in result_df.columns:
                use_metrics += [metric.metric_name for metric in answer_passage_metrics]
                for metric in answer_passage_metrics:
                    result_df[metric.metric_name] = result_df.apply(
                        lambda row: metric.eval(row['retrieval_gt_contents'], row['answer_pred']), axis=1)
            else:
                warnings.warn("You can't use answer metric with retrieval gt knowledge when retrieval_gt is None."
                              "Skip this metric.")

        # with gt - answer
        if answer_gt is not None:
            answer_gt_metrics = self.__answer_metrics_with_gt()
            use_metrics += [metric.metric_name for metric in answer_gt_metrics]
            for metric in answer_gt_metrics:
                result_df[metric.metric_name] = result_df.apply(
                    lambda row: metric.eval(row['answer_gt'], row['answer_pred']), axis=1)

        return EvaluateResult(
            results=result_df[use_metrics].mean().to_dict(),
            use_metrics=use_metrics,
            each_results=result_df
        )

    def __validate_passages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        It returns passage contents from retrieval_gt with column name 'retrieval_gt_contents'.
        """

        def fetch(row):
            fetch_passages = self.dummy_retrieval.fetch_data(row['retrieval_gt'])
            if len(fetch_passages) != len(row['retrieval_gt']):
                raise ValueError(f"Passages with id {row['retrieval_gt']} is not exist in retrieval.")
            return [passage.content for passage in fetch_passages]

        df['retrieval_gt_contents'] = df.apply(fetch, axis=1)
        return df

    def __retrieval_metrics_with_gt(self, rank_aware: bool = False) -> List[BaseRetrievalMetric]:
        """
        Make a list of retrieval metrics from a list of metric names
        """
        binary_metrics = {metric_names: metric for metric in
                          [TopKAccuracy(), EM_retrieval(), F1(), Hole(), Recall(), Precision()]
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
        ragas_metric_names = ["context_recall", "answer_relevancy", "faithfulness", "context_precision"]
        exist_metrics = [modified_metric_name
                         for metric_name in ragas_metric_names
                         for modified_metric_name in text_modifier(metric_name)
                         if modified_metric_name in self.metrics]
        if len(exist_metrics) <= 0:
            return []
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        )
        ragas_metrics = {metric_names: metric for metric in [context_recall, context_precision, answer_relevancy,
                                                             faithfulness]
                         for metric_names in text_modifier(metric.name)}
        result = [ragas_metrics[metric_name] for metric_name in self.metrics if metric_name in ragas_metrics]
        return result

    def __answer_metrics_with_gt(self) -> List[BaseAnswerMetric]:
        answer_metrics = {metric_names: metric for metric in [BLEU(), METEOR(), ROUGE(), EM_answer()]
                          for metric_names in text_modifier(metric.metric_name)}
        result = [answer_metrics[metric_name] for metric_name in self.metrics if metric_name in answer_metrics]
        return result

    def __answer_passage_metrics(self) -> List[BasePassageAnswerMetric]:
        metrics = {metric_names: metric for metric in [KF1()]
                   for metric_names in text_modifier(metric.metric_name)}
        result = [metrics[metric_name] for metric_name in self.metrics if metric_name in metrics]
        return result

    @staticmethod
    def uuid_to_str(id_list: List[Union[UUID, str]]) -> List[str]:
        return [str(_id) for _id in id_list]


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
