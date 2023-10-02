import concurrent.futures
from typing import List, Union
from uuid import UUID

import numpy as np
import pandas as pd

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class HybridRetrieval(BaseRetrieval):
    def __init__(self, retrievals: List[BaseRetrieval],
                 weights: List[float],
                 p: int = 500,
                 method: str = 'cc',
                 *args, **kwargs):
        """

        Initializes a HybridRetrieval object.

        Parameters:
        - retrievals (List[BaseRetrieval]): A list of BaseRetrieval objects.
        - weights (List[float]): A list of weights corresponding to each retrieval method. The weights should sum up to 1.0.
        - p (int, optional): The number of passages to retrieve from each retrieval method. Smaller p will result in
        faster process time, but may result lack of retrieved passages. Default is 500.
        - method (str, optional): The method used to combine the retrieval results. Choose between cc and rrf, which is
        convex combination and reciprocal rank fusion respectively. Default is 'cc'.
        """
        super().__init__()
        self.retrievals = retrievals
        assert sum(weights) == 1.0, "weights should be sum to 1.0"
        assert len(weights) > 1, "weights should be more than 1"
        self.weights = weights
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
        scores_df = pd.concat([future.result() for future in futures], axis=1, join="inner")

        normalized_scores = (scores_df - scores_df.min()) / (scores_df.max() - scores_df.min())
        if self.method == 'cc':
            normalized_scores['weighted_sum'] = normalized_scores.mul(self.weights).sum(axis=1)
        elif self.method == 'rrf':
            normalized_scores['weighted_sum'] = normalized_scores.apply(lambda row: self.__rrf_row(row, self.weights),
                                                                        axis=1)
        else:
            raise ValueError("method should be either 'cc' or 'rrf'")
        normalized_scores = normalized_scores.sort_values(by='weighted_sum', ascending=False)
        return (list(map(self.__str_to_uuid, normalized_scores.index[:top_k].tolist())),
                normalized_scores['weighted_sum'][:top_k].tolist())

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

    @staticmethod
    def __rrf_row(row, weights):
        assert len(row) == len(weights)
        result = 0
        for i in range(len(row)):
            result += (1 / (weights[i] + row[i]))
        return result
