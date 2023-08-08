import os
from typing import List, Optional
import chromadb
from chromadb.types import Where, WhereDocument
from langchain.schema import Document

from embed import Embedding
from vectorDB.base import BaseVectorDB


class Chroma(BaseVectorDB):
    def __init__(self, persist_dir: str, collection_name: str, embedding: Embedding):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding = embedding.embedding()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    @classmethod
    def load(cls, persist_dir: str, collection_name: str, embedding: Embedding):
        if not os.path.isdir(persist_dir):
            raise ValueError(f"persistent_dir must be a directory, but got {persist_dir}")
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        return cls(persist_dir, collection_name, embedding)

    def add_documents(self, docs: List[Document]):
        embeddings = self.embedding.embed_documents(docs)
        self.collection.add(
            embeddings=embeddings,
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
            ids=[doc.metadata["id"] for doc in docs]
        )

    def similarity_search(self, query: str, top_k: int = 5,
                          where: Optional[Where] = None,
                          where_document: Optional[WhereDocument] = None) -> tuple[List[Document], List[float]]:
        query_embedding = self.embedding.embed_query(query)
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        result_documents = [Document(page_content=doc, metadata=metadata) for doc, metadata in
                            zip(result["documents"][0],
                                result["metadatas"][0])]
        return result_documents, result["distances"][0]

    def delete_all(self):
        self.collection.delete()

    def get_db_type(self) -> str:
        return "chroma"
