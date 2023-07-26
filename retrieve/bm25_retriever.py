import uuid
from typing import List
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from retrieve.base import BaseRetriever
from utils import FileChecker
import pickle
from tqdm import tqdm


class BM25Retriever(BaseRetriever):
    """
    BM25Retriever data format example
    {
        "id": {"value": [3, 56, 3], "metadata": {metadata}},
        "id2": {"value": [3, 56, 3], "metadata": {metadata}}
    }
    metadata must have "content" key
    """

    def __init__(self, data=None):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
        self.data = data

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Document]:
        if self.data is None:
            raise ValueError("BM25Retriever.data is None. Please save data first.")

        values = [value["value"] for key, value in self.data.items()]
        contents = [value["metadata"]["content"] for key, value in self.data.items()]
        bm25 = BM25Okapi(values)
        tokenized_query = self.__tokenize([query])[0]
        result = bm25.get_top_n(tokenized_query, contents, n=top_k)
        return [Document(page_content=content) for content in result]

    @classmethod
    def load(cls, save_path: str, *args, **kwargs):
        if not FileChecker(save_path).check_type(file_types=[".pkl", ".pickle"]).is_exist():
            return BM25Retriever()
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        return BM25Retriever(data)

    def save(self, documents: List[Document], upsert: bool = False, *args, **kwargs):
        # TODO : deal with id
        for document in tqdm(documents):
            if not upsert:
                continue
            self.update_one(document)

    def save_one(self, document: Document, upsert: bool = False, *args, **kwargs):
        if not upsert and self.__check_id(document.metadata["id"]):
            return
        self.update_one(document)

    def delete(self, ids: List[str], *args, **kwargs):
        pass

    def delete_one(self, id: str, *args, **kwargs):
        pass

    def delete_all(self, *args, **kwargs):
        pass

    def update(self, documents: List[Document], *args, **kwargs):
        for document in tqdm(documents):
            self.update_one(document)

    def update_one(self, document: Document, *args, **kwargs):
        tokenized = self.__tokenize([document.page_content])[0]
        self.data[document.metadata["id"]] = {"value": tokenized, "metadata": document.metadata}

    def __tokenize(self, values: List[str]):
        tokenized = self.tokenizer(values)
        return tokenized.input_ids

    # TODO : add persist directory
