from copy import deepcopy
from typing import List, Optional

import pandas as pd
from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class DuoRCEvaluator(BaseDatasetEvaluator):
    """
    DuoRCEvaluator is a class for evaluating pipeline performance on DuoRC dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None,
                 sub_dataset_name: str = None
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        TriviaQA dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports TriviaQA dataset.
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score',
        'context_precision', 'BLEU', 'answer_relevancy', 'faithfulness', 'KF1'.
        param sub_dataset_name: sub_name is duorc sub dataset name, ParaphraseRC, SelfRC.
        SelfRC dataset is built on Wikipedia movie plots solely.
        ParaphraseRC has questions written from Wikipedia movie plots and the answers are given based
        on corresponding IMDb movie plots.

        Notice:
        The reason context_recall does not accommodate this benchmark is due to the excessive number
        of retrieval ground truths that exceed the context length in ragas metrics.
        """

        # TODO: task자체가 영화의 내용에 대해서 물어보는건데 영화가 어떤 영화인지 단서를 줘야하지 않을까?
        #  그렇다면 title을 question과 plot에 붙여야하는가? 그럼 데이터 손상인데 괜찮나? 그냥해버려? 다른 예제 코드는 안보임.
        # TODO:doc에 sub dataset 차이 적을것

        self.file_path = "duorc"
        # Check if sub dataset name valid.
        if sub_dataset_name is None:
            raise ValueError("You didn't input sub_dataset_name")

        if sub_dataset_name not in ['ParaphraseRC', 'SelfRC']:
            raise ValueError(f"You have input invalid sub_dataset_name, {sub_dataset_name}.\n"
                             f"You can input 'ParaphraseRC' or 'SelfRC' as sub_dataset_name")

        dataset = load_dataset(self.file_path, sub_dataset_name)['test'].to_pandas()
        # Delete no answer row.
        dataset = dataset[dataset['no_answer'] == False]
        # Shuffle dataset because same ground truths are grouped in dataset.
        dataset = dataset.sample(n=len(dataset), replace=False, axis=0).reset_index(drop=True)

        default_metrics = self.retrieval_gt_metrics + self.answer_gt_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_no_gt_ragas_metrics + self.answer_no_gt_ragas_metrics

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
        self.ingest_data = deepcopy(dataset[['plot_id', 'plot', 'question_id']])
        self.qa_data = dataset

        if evaluate_size is not None and len(self.qa_data) > evaluate_size:
            self.qa_data = self.qa_data[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        If you want to use context_precision metrics, you should ingest all data.
        """
        plot_id = self.ingest_data['plot_id'].drop_duplicates().reset_index(drop=True)
        remain = self.ingest_data[['plot', 'question_id']].groupby(['plot'], as_index=False).agg(
            {'question_id': lambda x: list(x)})

        ingest_data = pd.concat([plot_id, remain], axis=1)

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data[:ingest_size]
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        # Create passages.
        result = ingest_data.apply(self.__make_passages, axis=1)
        passages = result.tolist()

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        retrieval_gt = [[str(gt)] for gt in self.qa_data['plot_id']]
        answers = self.qa_data['answers'].apply(self.__make_answer_gt)

        return self._calculate_metrics(
            questions=self.qa_data['question'].tolist(),
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt,
            answer_gt=answers.tolist(),
            **kwargs
        )

    def __make_passages(self, row):

        passages = Passage(
            id=str(row['plot_id']),
            content=row['plot'],
            filepath=self.file_path,
            metadata_etc={
                'matching_question_id': row['question_id'],
            }
        )
        return passages

    def __make_answer_gt(self, row):

        return [answer for answer in row]
