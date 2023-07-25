from typing import List, Dict, Tuple

from langchain.schema import Document
from embed import delete_embeddings
from vectorDB import DB
from .base import BaseRetriever
from embed import Embedding


class LangchainRetriever(BaseRetriever):
    def __init__(self, db: DB):
        self.db = db

    def save_one(self, document: Document, *args, **kwargs):
        self.db.from_documents([document])

    def delete(self, ids: List[str], *args, **kwargs):
        raise NotImplementedError("delete is not implemented yet.")

    def delete_one(self, id: str, *args, **kwargs):
        raise NotImplementedError("delete_one is not implemented yet.")

    def delete_all(self):
        db_type = self.db.db_type
        delete_embeddings(db_type)

    def update(self, documents: List[Document], *args, **kwargs):
        raise NotImplementedError("update is not implemented yet.")

    def update_one(self, document: Document, *args, **kwargs):
        raise NotImplementedError("update_one is not implemented yet.")

    def retrieve(self, query: str, top_n: int = 5, *args, **kwargs) -> List[Document]:
        result = self.db.search(query, top_n)
        return result

    @classmethod
    def load(cls, db_type: str, embedding: Embedding):
        db = DB(db_type, embedding)
        retriever = cls(db)
        return retriever

    def save(self, documents: List[Document], *args, **kwargs):
        self.db.from_documents(documents)
