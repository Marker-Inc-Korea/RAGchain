import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional

import pandas as pd
from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.base import BaseEvaluator
from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage, EvaluateResult


class BaseDatasetEvaluator(BaseEvaluator, ABC):
    @abstractmethod
    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB,
               ingest_size: Optional[int] = None):
        pass

    def _validate_eval_size_and_ingest_size(self, ingest_size, eval_size):
        if ingest_size is not None and ingest_size < eval_size:
            # ingest size must be larger than evaluate size.
            raise ValueError(f"ingest size({ingest_size}) must be same or larger than evaluate size({eval_size})")


class BaseStrategyQA:
    def convert_qa_to_pd(self, data):
        result = []
        for key, value in data.items():
            result.append([
                value['question'],
                value['answer'],
                self.__unpack_evidence(value['evidence'])
            ])
        return pd.DataFrame(result, columns=['question', 'answer', 'evidence'])

    def __unpack_evidence(self, evidence) -> List[str]:
        evidence_per_annotator = []
        for annotator in evidence:
            evidence_per_annotator.extend(
                list(set(
                    evidence_id
                    for step in annotator
                    for x in step
                    if isinstance(x, list)
                    for evidence_id in x
                ))
            )
        return evidence_per_annotator


class BaseBeirEvaluator(BaseDatasetEvaluator):
    def __init__(self, run_pipeline: BaseRunPipeline,
                 file_path: str = None,
                 evaluate_size: Optional[int] = None,
                 metrics: Optional[List[str]] = None,
                 ):
        """
        :param run_pipeline: The pipeline that you want to benchmark.
        :param evaluate_size: The number of data to evaluate. If None, evaluate all data.
        We are using train set for evaluating in this class, so it is huge. Recommend to set proper size for evaluation.

        :param metrics: The list of metrics to use. If None, use all metrics that supports Beir dataset.
        Supporting metrics are Recall, Precision, Hole, TopK_Accuracy, EM, F1_score, context_precision, MRR.
        You must ingest all data for using context_recall  metrics.
        Notice: We except context_recall metric that is ragas metric.
                It takes a long time in evaluation because beir token is too big.
                So if you want to use context_recall metric, You can add self.retrieval_gt_ragas_metrics.

        The default metric refers to the metric that is essentially executed when you run the test file.
        Support metrics refer to those that are available for use.
        This distinction exists because the evaluation process for Ragas metrics is time-consuming.

        Additionally, you can preprocess datasets in this class constructor to benchmark your own pipeline.
        You can modify utils methods by overriding it for your dataset.
        """
        if file_path is None:
            raise ValueError("file_path is not input.")

        self.questions = load_dataset(file_path, 'queries')['queries'].to_pandas()
        self.corpus = load_dataset(file_path, 'corpus')['corpus'].to_pandas()
        self.qrels = load_dataset(f"{file_path}-qrels")['test'].to_pandas()

        self.run_pipeline = run_pipeline
        self.eval_size = evaluate_size
        self.file_path = file_path

        # Remove row that contains none relevant passages.
        self.qrels = self.qrels[self.qrels['score'] >= 1]

        # BeIR datasets consisted rank dataset and none rank dataset.
        # Create porper metrics.
        using_metrics = self.__call_metrics(metrics)
        super().__init__(run_all=False, metrics=using_metrics)

        # Preprocess qrels proper form and slice by evaluate size.
        self.qrels = BaseBeirEvaluator.__preprocess_qrels(self.qrels, evaluate_size)

        # Create retrieval_gt. If retrieval gt exist order, create retrieval gt order too.
        self.retrieval_gt = self.qrels['corpus-id'].tolist()

        # Create question
        q_id = self.qrels['query-id'].tolist()
        self.questions = self.questions.loc[self.questions['_id'].isin(q_id)]['text'].tolist()

    def ingest(self, retrievals: List[BaseRetrieval],
               db: BaseDB,
               ingest_size: Optional[int] = None,
               random_state=None):
        """
        Ingest dataset to retrievals and db.
        :param retrievals: The retrievals that you want to ingest.
        :param db: The db that you want to ingest.
        :param ingest_size: The number of data to ingest. If None, ingest all data.
        You must ingest all data for using context_recall metrics.
        If the ingest size is excessively large, it results in prolonged processing times.
        To address this, we shuffle the corpus and slice it according to the ingest size for testing purposes.
        The reason for transforming the retrieval ground truth corpus into passages and ingesting it is to enable
        retrieval to retrieve the retrieval ground truth within the database.
        :param random_state: A random state to fix the shuffled corpus to ingest.
        Types are like these. int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        """
        gt_ids = deepcopy(self.retrieval_gt)
        corpus = deepcopy(self.corpus)

        self._validate_eval_size_and_ingest_size(ingest_size, len(self.questions))

        gt_before_passages, id_for_remove_duplicated_corpus = self.make_gt_passages_and_duplicated_id(gt_ids, corpus)

        # Slice corpus by ingest_size and remove duplicate passages.
        corpus_before_passages = self.remove_duplicate_passages(ingest_size=ingest_size,
                                                                corpus=corpus,
                                                                random_state=random_state,
                                                                id_for_remove_duplicated_corpus=id_for_remove_duplicated_corpus,
                                                                )
        gt_passages = gt_before_passages.apply(self.make_corpus_passages, axis=1).tolist()

        passages = corpus_before_passages.apply(self.make_corpus_passages, axis=1).tolist()
        passages += gt_passages

        for retrieval in retrievals:
            retrieval.ingest(passages)
        db.create_or_load()
        db.save(passages)

    def evaluate(self, **kwargs) -> EvaluateResult:
        """
        Evaluate pipeline performance on beir dataset.
        This method always validate passages.
        """

        return self._calculate_metrics(
            questions=self.questions,
            pipeline=self.run_pipeline,
            retrieval_gt=self.retrieval_gt
        )

    # Default metrics is basically running metrics if you run test file.
    # Support metrics is the metrics you are available.
    # This separation is because Ragas metrics take a long time in evaluation.
    def __call_metrics(self, metrics):
        default_metrics = (self.retrieval_gt_metrics)
        support_metrics = (default_metrics + self.retrieval_gt_ragas_metrics
                           + self.retrieval_no_gt_ragas_metrics + self.answer_no_gt_ragas_metrics)

        if metrics is not None:
            # Check if your metrics are available in evaluation datasets.
            for metric in metrics:
                if metric not in support_metrics:
                    raise ValueError(f"You input {metric} that this dataset evaluator not support.")
            using_metrics = list(set(metrics))
        else:
            using_metrics = default_metrics

        return using_metrics

    @staticmethod
    def __preprocess_qrels(qrels, evaluate_size):
        # Convert integer type to string type of qrels' query-id and corpus-id.
        qrels[['query-id', 'corpus-id']] = qrels[['query-id', 'corpus-id']].astype(str)

        # Preprocess qrels. Some query ids duplicated and were appended different corpus id.
        preprocessed_qrels = qrels.groupby('query-id', as_index=False).agg(
            {'corpus-id': lambda x: list(x), 'score': lambda x: list(x)})

        if evaluate_size is not None and len(qrels) > evaluate_size:
            preprocessed_qrels = preprocessed_qrels[:evaluate_size]

        return preprocessed_qrels

    def make_gt_passages_and_duplicated_id(self, gt_ids, corpus):
        # Flatten retrieval ground truth ids and convert string type.
        gt_ids_lst = [str(id) for id in list(itertools.chain.from_iterable(gt_ids))]
        id_for_remove_duplicated_corpus = deepcopy(gt_ids_lst)

        # gt_passages is retrieval_gt passages to ingest.
        gt_passages = corpus.loc[corpus['_id'].isin(id_for_remove_duplicated_corpus)]

        return gt_passages, id_for_remove_duplicated_corpus

    def remove_duplicate_passages(self,
                                  ingest_size: int,
                                  corpus,
                                  random_state,
                                  id_for_remove_duplicated_corpus: List[str],
                                  ):
        """
        Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        Marking duplicated values in the corpus using retrieval_gt id.
        """
        corpus_passages = corpus
        if ingest_size is not None:
            corpus_passages = corpus.sample(n=ingest_size, replace=False, random_state=random_state,
                                            axis=0)

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the dataframe using values list and Remove duplicated values.
        mask = corpus_passages.isin(id_for_remove_duplicated_corpus)
        corpus_passages = corpus_passages[~mask.any(axis=1)]

        # Assert whether duplicated passages were removed in corpus_passages
        if corpus_passages['_id'].isin(id_for_remove_duplicated_corpus).any().any() == True:
            raise ValueError(
                "There are duplicated values in corpus_passages. Please remove duplicated values to ingest efficiently")

        return corpus_passages

    def make_corpus_passages(self, row):
        # Corpus to passages
        passage = Passage(
            id=row['_id'],
            content=row['text'],
            filepath=self.file_path,
            metadata_etc={'title': row['title']}
        )
        return passage
