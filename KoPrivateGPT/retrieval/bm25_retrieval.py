import pickle
from typing import List, Union
from uuid import UUID

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import FileChecker


class BM25Retrieval(BaseRetrieval):
    """
    Default data structure looks like this:
    {
        "tokens" : [], # 2d list of tokens
        "passage_id" : [], # 2d list of passage_id. Type must be UUID.
    }
    """

    def __init__(self, save_path: str, *args, **kwargs):
        super().__init__()
        self.data = self.load_data(save_path)
        assert (len(self.data["tokens"]) == len(self.data["passage_id"]))
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")

    @staticmethod
    def load_data(save_path: str):
        if FileChecker(save_path).check_type(file_types=[".pkl", ".pickle"]).is_exist():
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
            assert ('tokens' and 'passage_id' in list(data.keys()))
            return data
        else:
            if not FileChecker(save_path).check_type(file_types=[".pkl", ".pickle"]):
                raise ValueError("input save_path is not pickle file.")
            return {
                "tokens": [],
                "passage_id": [],
            }

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        ids = self.retrieve_id(query, top_k)
        passage_list = self.fetch_data(ids)
        return passage_list

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        if self.data is None:
            raise ValueError("BM25Retriever.data is None. Please save data first.")

        bm25 = BM25Okapi(self.data["tokens"])
        tokenized_query = self.__tokenize([query])[0]
        scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]  # this code is from rank_bm25.py in rank_bm25 package
        return [self.data['passage_id'][i] for i in top_n]

    def ingest(self, passages: List[Passage]):
        for passage in tqdm(passages):
            self._save_one(passage)
        self.persist(self.save_path)

    def delete(self, ids: List[Union[str, UUID]]):
        for _id in ids:
            self._delete_one(_id)

    def _delete_one(self, _id: Union[str, UUID]):
        index = self.data["passage_id"].index(_id)
        del self.data["tokens"][index]
        del self.data["passage_id"][index]

    def _save_one(self, passage: Passage):
        tokenized = self.__tokenize([passage.content])[0]
        self.data["tokens"].append(tokenized)
        self.data["passage_id"].append(passage.id)

    def persist(self, save_path: str):
        FileChecker(save_path).check_type(file_types=[".pkl", ".pickle"])
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __tokenize(self, values: List[str]):
        tokenized = self.tokenizer(values)
        return tokenized.input_ids
