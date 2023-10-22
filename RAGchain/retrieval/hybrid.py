import concurrent.futures
from typing import List, Union, Optional
from uuid import UUID

import numpy as np
import pandas as pd

from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage


class HybridRetrieval(BaseRetrieval):
    """
    Hybrid Retrieval class for retrieve passages from multiple retrievals.
    You can combine retrieval scores with rrf algorithm or convex combination algorithm.
    """
    def __init__(self, retrievals: List[BaseRetrieval],
                 weights: Optional[List[float]] = None,
                 p: int = 500,
                 method: str = 'cc',
                 rrf_k: int = 60,
                 *args, **kwargs):
        """

        Initializes a HybridRetrieval object.

        :param retrievals: A list of BaseRetrieval objects. Must be more than 1.
        :param weights: A list of weights corresponding to each retrieval method.
        The weights should sum up to 1.0.
        :param p: The number of passages to retrieve from each retrieval method. Smaller p will result in
        faster process time, but may result lack of retrieved passages. Default is 500.
        :param method: The method used to combine the retrieval results. Choose between cc and rrf, which is
        convex combination and reciprocal rank fusion respectively. Default is 'cc'.
        :param rrf_k: k parameter for reciprocal rank fusion. Default is 60.
        """
        super().__init__()
        self.retrievals = retrievals
        self.weights = weights
        self.rrf_k = rrf_k
        if method == 'cc':
            assert sum(weights) == 1.0, "weights should be sum to 1.0"
            assert len(weights) > 1, "weights should be more than 1"
        elif method == 'rrf':
            pass
        else:
            raise ValueError("method should be either 'cc' or 'rrf'")
        self.p = p
        self.method = method

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        ids = self.retrieve_id(query, top_k, *args, **kwargs)
        passages = self.fetch_data(ids)
        return passages

    def ingest(self, passages: List[Passage]):
        for retrieval in self.retrievals:
            retrieval.ingest(passages)

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        ids, scores = self.retrieve_id_with_scores(query, top_k=top_k, *args, **kwargs)
        return ids

    def retrieve_id_with_scores(self, query: str, top_k: int = 5, *args, **kwargs) -> tuple[
        List[Union[str, UUID]], List[float]]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.retrieve_id_with_scores_parallel, retrieval, query, self.p, *args, **kwargs)
                       for retrieval in self.retrievals]

        if self.method == 'cc':
            scores_df = pd.concat([future.result() for future in futures], axis=1, join="inner")
            normalized_scores = (scores_df - scores_df.min()) / (scores_df.max() - scores_df.min())
            normalized_scores['weighted_sum'] = normalized_scores.mul(self.weights).sum(axis=1)
            normalized_scores = normalized_scores.sort_values(by='weighted_sum', ascending=False)
            return (list(map(self.__str_to_uuid, normalized_scores.index[:top_k].tolist())),
                    normalized_scores['weighted_sum'][:top_k].tolist())
        elif self.method == 'rrf':
            scores_df = pd.concat([future.result() for future in futures], axis=1)
            rank_df = scores_df.rank(ascending=False, method='min')
            rank_df = rank_df.fillna(0)
            rank_df['rrf'] = rank_df.apply(self.__rrf_calculate, axis=1)
            rank_df = rank_df.sort_values(by='rrf', ascending=False)
            return (list(map(self.__str_to_uuid, rank_df.index[:top_k].tolist())),
                    rank_df['rrf'][:top_k].tolist())
        else:
            raise ValueError("method should be either 'cc' or 'rrf'")

    def delete(self, ids: List[Union[str, UUID]]):
        for retrieval in self.retrievals:
            retrieval.delete(ids)

    def retrieve_id_with_scores_parallel(self, retrieval: BaseRetrieval, query: str, top_k: int, *args,
                                         **kwargs) -> pd.Series:
        ids, scores = retrieval.retrieve_id_with_scores(query, top_k=top_k, *args, **kwargs)
        return pd.Series(dict(zip(list(map(str, ids)), scores)))

    @staticmethod
    def min_max_normalization(arr: np.ndarray):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    @staticmethod
    def __str_to_uuid(input_str: str) -> Union[str, UUID]:
        try:
            return UUID(input_str)
        except:
            return input_str

    def __rrf_calculate(self, row):
        result = 0
        for r in row:
            if r == 0:
                continue
            result += 1 / (r + self.rrf_k)
        return result
