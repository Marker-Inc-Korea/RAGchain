import numpy as np
from typing import List
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from KoPrivateGPT.retrieve.base import BaseRetriever
from utils import FileChecker
import pickle
from tqdm import tqdm


class BM25Retriever(BaseRetriever):
    """
    Default data structure looks like this:
    {
        "tokens" : [], # 2d list of tokens
        "texts" : [], # 2d list of texts
    """

    def __init__(self, data=None):
        if data is None:
            self.data = {
                "tokens": [],
                "texts": [],
                "metadata": [],
            }
        else:
            self.data = data
        assert (len(self.data["tokens"]) == len(self.data["texts"]))
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Document]:
        if self.data is None:
            raise ValueError("BM25Retriever.data is None. Please save data first.")

        bm25 = BM25Okapi(self.data["tokens"])
        tokenized_query = self.__tokenize([query])[0]
        scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]  # this code is from rank_bm25.py in rank_bm25 package
        return [Document(page_content=self.data["texts"][i], metadata=self.data["metadata"][i]) for i in top_n]

    @classmethod
    def load(cls, save_path: str, *args, **kwargs):
        if not FileChecker(save_path).check_type(file_types=[".pkl", ".pickle"]).is_exist():
            return BM25Retriever()
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        return BM25Retriever(data)

    def save(self, documents: List[Document], *args, **kwargs):
        for document in tqdm(documents):
            self.save_one(document)

    def save_one(self, document: Document, *args, **kwargs):
        tokenized = self.__tokenize([document.page_content])[0]
        self.data["tokens"].append(tokenized)
        self.data["texts"].append(document.page_content)
        self.data["metadata"].append(document.metadata)

    def delete(self, ids: List[str], *args, **kwargs):
        raise NotImplementedError

    def delete_one(self, id: str, *args, **kwargs):
        raise NotImplementedError

    def delete_all(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, documents: List[Document], *args, **kwargs):
        raise NotImplementedError

    def update_one(self, document: Document, *args, **kwargs):
        raise NotImplementedError

    def persist(self, save_path: str):
        FileChecker(save_path).check_type(file_types=[".pkl", ".pickle"])
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __tokenize(self, values: List[str]):
        tokenized = self.tokenizer(values)
        return tokenized.input_ids
