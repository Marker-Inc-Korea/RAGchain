import os
from typing import List, Optional
import chromadb
from chromadb.types import Where, WhereDocument
from uuid import UUID
from KoPrivateGPT.schema.vector import Vector
from KoPrivateGPT.utils.vectorDB.base import BaseVectorDB


class Chroma(BaseVectorDB):
    def __init__(self, persist_dir: str, collection_name: str):
        if not os.path.isdir(persist_dir):
            raise ValueError(f"persistent_dir must be a directory, but got {persist_dir}")
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_vectors(self, vectors: List[Vector]):
        self.collection.add(
            embeddings=[v.vector for v in vectors],
            metadatas=[{"passage_id": str(v.passage_id)} for v in vectors],
            ids=[str(v.passage_id) for v in vectors]
        )

    def similarity_search(self, query_vectors: List[float], top_k: int = 5,
                          where: Optional[Where] = None,
                          where_document: Optional[WhereDocument] = None) -> tuple[List[UUID], List[float]]:
        result = self.collection.query(
            query_embeddings=query_vectors,
            n_results=top_k
        )
        return [UUID(metadata['passage_id']) for metadata in result["metadatas"][0]], result["distances"][0]

    def delete_all(self):
        self.collection.delete()

    def get_db_type(self) -> str:
        return "chroma"
