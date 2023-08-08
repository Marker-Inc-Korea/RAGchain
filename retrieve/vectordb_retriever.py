from typing import List, Dict, Tuple

from langchain.schema import Document
from embed import delete_embeddings_vectordb
from options import ChromaOptions, PineconeOptions
from vectorDB import Pinecone
from vectorDB.chroma import Chroma
from vectorDB.base import BaseVectorDB
from .base import BaseRetriever
from embed import Embedding


class VectorDBRetriever(BaseRetriever):
    def __init__(self, db: BaseVectorDB):
        self.db = db

    def save_one(self, document: Document, *args, **kwargs):
        self.db.add_documents([document])

    def delete(self, ids: List[str], *args, **kwargs):
        raise NotImplementedError("delete is not implemented yet.")

    def delete_one(self, id: str, *args, **kwargs):
        raise NotImplementedError("delete_one is not implemented yet.")

    def delete_all(self):
        delete_embeddings_vectordb(self.db.get_db_type())

    def update(self, documents: List[Document], *args, **kwargs):
        raise NotImplementedError("update is not implemented yet.")

    def update_one(self, document: Document, *args, **kwargs):
        raise NotImplementedError("update_one is not implemented yet.")

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> tuple[List[Document], List[float]]:
        result, scores = self.db.similarity_search(query, top_k)
        return result

    @classmethod
    def load(cls, db_type: str, embedding: Embedding):
        if db_type in ['chroma', 'Chroma', 'CHROMA']:
            db = Chroma.load(ChromaOptions.persist_dir, ChromaOptions.collection_name, embedding)
        elif db_type in ['pinecone', 'Pinecone', 'PineCone', 'PINECONE']:
            db = Pinecone.load(PineconeOptions.namespace, embedding)
        else:
            raise ValueError(f"Unknown db type: {db_type}")
        retriever = cls(db)
        return retriever

    def save(self, documents: List[Document], *args, **kwargs):
        self.db.add_documents(documents)
