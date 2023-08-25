import json
from uuid import UUID

import numpy as np
from typing import List, Union
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.pipeline.selector import ModuleSelector
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.linker import RedisDBSingleton
from KoPrivateGPT.utils.util import FileChecker
import pickle
from tqdm import tqdm


class BM25Retrieval(BaseRetrieval):
    """
    Default data structure looks like this:
    {
        "tokens" : [], # 2d list of tokens
        "passage_id" : [], # 2d list of passage_id. Type must be UUID.
    }
    """

    def __init__(self, save_path: str, *args, **kwargs):
        self.data = self.load_data(save_path)
        assert (len(self.data["tokens"]) == len(self.data["passage_id"]))
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
        self.redis_db = RedisDBSingleton()

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

    @staticmethod
    def split_ids(ids: List[UUID], db_origin_list: List[json]):
        mongo_db_ids = [ids[i] for i, db_origin in enumerate(db_origin_list) if db_origin.db_type == "mongo_db"]
        pickle_db_ids = [ids[i] for i, db_origin in enumerate(db_origin_list) if db_origin.db_type == "pickle_db"]
        return mongo_db_ids, pickle_db_ids

    def fetch_data(self, ids: List[UUID]) -> List[Passage]:
        db_origin_list = self.redis_db.get_json(ids)
        mongo_db_ids, pickle_db_ids = self.split_ids(ids, db_origin_list)
        passage_list = []
        self.fetch_mongo_data(mongo_db_ids, db_origin_list, passage_list)
        self.fetch_pickle_data(pickle_db_ids, db_origin_list, passage_list)
        return passage_list

    @staticmethod
    def fetch_mongo_data(mongo_db_ids: List[UUID], db_origin_list, passage_list):
        if mongo_db_ids:
            mongo_db = ModuleSelector("db").select("mongo_db").get(**db_origin_list[0].db_path)
            mongo_db.load()
            mongo_db_passage_list = mongo_db.fetch(mongo_db_ids)
            passage_list.append(mongo_db_passage_list)

    @staticmethod
    def fetch_pickle_data(pickle_db_ids: List[UUID], db_origin_list, passage_list):
        if pickle_db_ids:
            pickle_db = ModuleSelector("db").select("pickle_db").get(**db_origin_list[0].db_path)
            pickle_db.load()
            pickle_db_passage_list = pickle_db.fetch(pickle_db_ids)
            passage_list.append(pickle_db_passage_list)
