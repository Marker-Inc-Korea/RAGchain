import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional

import pandas as pd
from datasets import load_dataset

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.base import BaseEvaluator
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage


class BaseDatasetEvaluator(BaseEvaluator, ABC):
    @abstractmethod
    def ingest(self, retrievals: List[BaseRetrieval], db: BaseDB,
               ingest_size: Optional[int] = None):
        pass


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
    def __init__(self, evaluate_size: Optional[int] = None,
                 file_path: str = None,
                 metrics: Optional[List[str]] = None,
                 ):
        if file_path is None:
            raise ValueError("Please input file_path to call the metrics.")

        queries = load_dataset(file_path, 'queries')['queries']
        corpus = load_dataset(file_path, 'corpus')['corpus']
        qrels = load_dataset(f"{file_path}-qrels")['test']

        self.queries = queries.to_pandas()
        self.corpus = corpus.to_pandas()
        self.qrels = qrels.to_pandas()
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
        self.questions = self.queries.loc[self.queries['_id'].isin(q_id)]['text'].tolist()

    def __call_metrics(self, metrics):
        if (self.qrels['score'] > 1).any():
            # TODO: rank aware인경우 잘 나오는지 체크하기
            support_metrics = (self.retrieval_gt_metrics + self.retrieval_no_gt_metrics +
                               self.retrieval_gt_metrics_rank_aware + self.answer_no_gt_metrics)
        else:
            support_metrics = (self.retrieval_gt_metrics + self.retrieval_no_gt_metrics)

        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics

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

    @staticmethod
    def make_gt_passages(gt_ids, corpus):
        # Flatten retrieval ground truth ids and convert string type.
        gt_ids_lst = [str(id) for id in list(itertools.chain.from_iterable(gt_ids))]
        id_for_remove_duplicated_corpus = deepcopy(gt_ids_lst)

        # gt_passages is retrieval_gt passages to ingest.
        gt_passages = corpus.loc[corpus['_id'].isin(id_for_remove_duplicated_corpus)]

        return gt_passages

    def remove_duplicate_passages(self,
                                  ingest_size: int,
                                  eval_size,
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
            # ingest size must be larger than evaluate size.
            if ingest_size >= eval_size:
                corpus_passages = corpus.sample(n=ingest_size, replace=False, random_state=random_state,
                                                axis=0)
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

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

