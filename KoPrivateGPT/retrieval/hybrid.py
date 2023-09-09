from typing import List, Union
from uuid import UUID

import numpy as np

from KoPrivateGPT.retrieval.base import BaseRetrieval
from schema import Passage


class HybridRetrieval(BaseRetrieval):
    def __init__(self, retrievals: List[BaseRetrieval],
                 weights: List[float],
                 p: int = 500,
                 *args, **kwargs):
        super().__init__()
        self.retrievals = retrievals
        assert sum(weights) == 1.0, "weights should be sum to 1.0"
        assert len(weights) > 1, "weights should be more than 1"
        self.weights = weights
        self.p = p

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
        ids = {}
        for retrieval in self.retrievals:
            _ids, _scores = retrieval.retrieve_id_with_scores(query, top_k=self.p, *args, **kwargs)
            if len(ids) == 0:
                for i in range(len(_ids)):
                    ids[str(_ids[i])] = [_scores[i]]
            else:
                for i in range(len(_ids)):
                    try:
                        ids[str(_ids[i])].append(_scores[i])
                    except KeyError:
                        pass  # ignore no key at first retrieval
        keys = list(ids.keys())
        scores = []
        for key in keys:
            scores.append(ids[key])

        # min-max normalization
        scores = np.array(scores)
        scores = np.apply_along_axis(self.min_max_normalization, axis=0, arr=scores)

        # weighted sum
        weighted_scores = np.sum(scores * np.array(self.weights), axis=1)
        top_k_index = np.argsort(weighted_scores)[::-1][:top_k]
        return [self.__str_to_uuid(elem) for elem in np.array(keys)[top_k_index]], weighted_scores[top_k_index]

    @staticmethod
    def min_max_normalization(arr: np.ndarray):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    @staticmethod
    def __str_to_uuid(input_str: str) -> Union[str, UUID]:
        try:
            return UUID(input_str)
        except:
            return input_str
