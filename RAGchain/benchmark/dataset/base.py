import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional

import pandas as pd

from RAGchain.DB.base import BaseDB
from RAGchain.benchmark.base import BaseEvaluator
from RAGchain.retrieval.base import BaseRetrieval


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
    # TODO: retrieval gt를 만드는건 함수로 따로 부모 클래스에 만들기
    # TODO: fiqa는 relevance judgement를 binary로 만들어놓음. 숫자가 연속인지 바이너리인지 체크후 부모클래에서 고려해서 preprocess
    # TODO: gt는 한 쿼리당 여러개 corpus일수 있음 -> query id가 여러번 반복되고 각각 다른 corpus id가 담김.

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
        if ingest_size is not None:
            # ingest size must be larger than evaluate size.
            if ingest_size >= eval_size:
                corpus_passages = corpus.sample(n=ingest_size, replace=False, random_state=random_state,
                                                axis=0)
            else:
                raise ValueError("ingest size must be same or larger than evaluate size")

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
