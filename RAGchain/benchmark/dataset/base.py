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

    # TODO: support metric dataset에 따라서 조정하기 -> score가 binary인지 연속인지
    # TODO: score가 binary이므로 rank aware metric은 없으며 answer gt 또한 없다. 전처리를 여기서 할수 있고, ingest나 evaluate같은경우에는 다른곳에서 할수 있게?
    # TODO: 지금 만드는건 쿼리 아이디로 qrels를 보고 판단하는것 score들에 order나 relevant한게 아닌 0값도 나중에 한번에 부모클래스 처리
    # TODO: gt는 한 쿼리당 여러개 corpus일수 있음 -> query id가 여러번 반복되고 각각 다른 corpus id가 담김.

    # TODO: Run all param은 metric 다 돌리는것
    def __init__(self, evaluate_size: Optional[int] = None,
                 file_path: str = None,
                 metrics: Optional[List[str]] = None,
                 ):
        # TODO: IF case 추가
        # TODO: support metric 잘 불러와지는지 예외처리도 모두 체크하기

        if file_path is None:
            raise ValueError("Please input file_path to call the metrics.")

        queries = load_dataset(file_path, 'queries')['queries']
        corpus = load_dataset(file_path, 'corpus')['corpus']
        qrels = load_dataset(f"{file_path}-qrels")['test']

        self.queries = queries.to_pandas()
        self.corpus = corpus.to_pandas()
        self.qrels = qrels.to_pandas()
        self.file_path = file_path

        # TODO: 여기에 0과 -1인 score는 non relevant이므로 날려버리기 그렇게 되면 아래 metric 불러오는 코드도 수정하면 됌
        # TODO: rank_order코드도 case 만들어서 만들기
        # BeIR datasets consisted rank dataset and none rank dataset.
        # Create porper metrics.
        using_metrics = self.__call_metrics(metrics)
        super().__init__(run_all=False, metrics=using_metrics)

        # Preprocess qrels proper form.
        self.qrels = self.__preprocess_qrels(self.qrels, evaluate_size)

        # Create retrieval_gt.
        self.retrieval_gt = self.qrels['corpus-id'].tolist()

        # Create question
        q_id = self.qrels['query-id'].tolist()
        self.questions = self.queries.loc[self.queries['_id'].isin(q_id)]['text'].tolist()

    def __call_metrics(self, metrics):
        if (self.qrels['score'] > 1).any():
            support_metrics = (self.retrieval_gt_metrics + self.retrieval_no_gt_metrics +
                               self.retrieval_gt_metrics_rank_aware + self.answer_no_gt_metrics)
        else:
            support_metrics = (self.retrieval_gt_metrics + self.retrieval_no_gt_metrics)

        if metrics is not None:
            using_metrics = list(set(metrics))
        else:
            using_metrics = support_metrics

        return using_metrics

    def __preprocess_qrels(self, qrels, evaluate_size):
        # Convert integer type to string type of qrels' query-id and corpus-id.
        qrels[['query-id', 'corpus-id']] = qrels[['query-id', 'corpus-id']].astype(str)

        # Preprocess qrels. Some query ids duplicated and were appended different corpus id.
        preprocessed_qrels = qrels.groupby('query-id', as_index=False).agg(
            {'corpus-id': lambda x: list(x), 'score': lambda x: list(x)})

        if evaluate_size is not None and len(qrels) > evaluate_size:
            preprocessed_qrels = preprocessed_qrels[:evaluate_size]

        return preprocessed_qrels

    def make_gt_passages(self, gt_ids, corpus):
        # Flatten retrieval ground truth ids and convert string type.
        gt_ids_lst = [str(id) for id in list(itertools.chain.from_iterable(gt_ids))]
        id_for_remove_duplicated_corpus = deepcopy(gt_ids_lst)

        # gt_passages is retrieval_gt passages to ingest.
        gt_passages = corpus.loc[corpus['_id'].isin(id_for_remove_duplicated_corpus)]

        return gt_passages

    def remove_duplicate_passages(self, ingest_size: int,
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

        # TODO: 방금 eval_size가 ingest_size보다 컸는데 작동이 안됐음

        # Remove duplicated passages between corpus and retrieval gt for ingesting passages faster.
        # Marking duplicated values in the corpus using retrieval_gt id.
        mask = corpus_passages.isin(id_for_remove_duplicated_corpus)
        # Remove duplicated passages
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
