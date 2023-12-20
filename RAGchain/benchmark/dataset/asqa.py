import itertools
from copy import deepcopy
from typing import List, Optional

import numpy as np
from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.dataset.base import BaseDatasetEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import EvaluateResult, Passage


class ASQAEvaluator(BaseDatasetEvaluator):
    """
    ASQAEvaluator is a class for evaluating pipeline performance on NFCorpus dataset.
    """

    def __init__(self, run_pipeline: BaseRunPipeline,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        NFCorpus dataset we use is huge. Recommend to set proper size for evaluation.
        :param metrics: The list of metrics to use. If None, use all metrics that supports NFCorpus dataset.
        Supporting metrics are 'Recall', 'Precision', 'Hole', 'TopK_Accuracy', 'EM', 'F1_score',
        'context_recall', 'BLEU', 'context_precision', 'answer_relevancy', 'faithfulness', 'KF1'.

        Notice:
        Default metrics is basically running metrics if you run test file.
        Support metrics is the metrics you are available.
        This separation is because Ragas metrics take a long time in evaluation.
        """

        self.file_path = "din0s/asqa"
        dataset = load_dataset(self.file_path)['dev'].to_pandas().reset_index(drop=True)

        dataset['retrieval_gt'], dataset['answer_gt'] = zip(*dataset.apply(self.__split_content_answer, axis=1))
        dataset = dataset.dropna(ignore_index=True)

        self.question = dataset[['sample_id', 'ambiguous_question']]
        self.content = dataset[['sample_id', 'retrieval_gt', 'wikipages']]
        self.answers = dataset[['sample_id', 'answer_gt']]

        default_metrics = self.retrieval_gt_metrics + self.answer_gt_metrics + self.answer_passage_metrics
        support_metrics = default_metrics + self.retrieval_gt_ragas_metrics \
                          + self.retrieval_no_gt_ragas_metrics + self.answer_no_gt_ragas_metrics

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

        if evaluate_size is not None and len(self.question) > evaluate_size:
            self.question = self.question[:evaluate_size]
            self.answers = self.answers[:evaluate_size]
            self.retrieval_gt = self.content[:evaluate_size]

    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB, ingest_size: Optional[int] = None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        """
        ingest_data = deepcopy(self.content)

        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= self.eval_size:
                ingest_data = ingest_data[:ingest_size]
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

        passages = ingest_data.apply(self.__make_passages, axis=1).tolist()
        passages = list(itertools.chain.from_iterable(passages))

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        question = self.question['ambiguous_question']
        answer_gt = [answers for answers in self.answers['answer_gt']]
        retrieval_gt = self.retrieval_gt.apply(self.__make_retrieval_gt, axis=1).tolist()

        return self._calculate_metrics(
            questions=question,
            pipeline=self.run_pipeline,
            retrieval_gt=retrieval_gt,
            answer_gt=answer_gt,
            **kwargs
        )

    def __make_passages(self, row):
        passage = []
        for idx, content_dict in enumerate(row['retrieval_gt']):
            passage.append(Passage(
                id=str(row['sample_id']) + '_' + str(idx),
                content=content_dict['content'],
                filepath=self.file_path,
                metadata_etc={
                    'title': content_dict['wikipage'],
                    'url': ", ".join([wikipages['url'] for wikipages in row['wikipages']]),
                }
            ))

        return passage

    def __split_content_answer(self, row):
        content_lst = []
        answer_lst = []
        for element in row['annotations']:
            if len(element['knowledge']) != 0:
                content_lst += [content for content in element['knowledge']]
            answer_lst += [element['long_answer']]

        if len(content_lst) == 0 or len(answer_lst) == 0:
            row['retrieval_gt'], row['answers'] = np.NAN, np.NAN
        else:
            row['retrieval_gt'], row['answers'] = content_lst, answer_lst

        return row['retrieval_gt'], row['answers']

    def __make_retrieval_gt(self, row):
        gt = [str(row['sample_id']) + '_' + str(idx) for idx, content in enumerate(row['retrieval_gt'])]

        return gt
