from typing import List
from typing import Optional, Dict, Union
from uuid import UUID

import pinecone
from langchain.embeddings.base import Embeddings

from KoPrivateGPT.schema.vector import Vector
from KoPrivateGPT.utils.vectorDB.base import BaseVectorDB


class Pinecone(BaseVectorDB):
    def __init__(self, api_key: str, environment: str,
                 index_name: str, namespace: str, dimension: Optional[int] = None, *args, **kwargs):
        pinecone.init(
            api_key=api_key,
            environment=environment
        )

        active_indexes = pinecone.list_indexes()
        if index_name not in active_indexes:
            if dimension is None:
                raise ValueError("Dimension must be set when creating a new index.")
            pinecone.create_index(index_name, dimension=dimension, *args, **kwargs)
        self.index = pinecone.Index(index_name)
        self.namespace = namespace

    def add_vectors(self, vectors: List[Vector]):
        pinecone_vectors = []
        for v in vectors:
            pinecone_vectors.append(
                {'id': str(v.passage_id),
                 'values': v.vector,
                 'metadata': {"passage_id": str(v.passage_id)}}
            )
        self.index.upsert(vectors=pinecone_vectors, namespace=self.namespace)

    def similarity_search(self, query_vectors: List[float], top_k: int = 5,
                          filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None) -> tuple[
        List[Union[UUID, str]], List[float]]:
        response = self.index.query(
            vector=query_vectors,
            namespace=self.namespace,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
        )
        ids = []
        scores = []
        for res in response["matches"]:
            metadata = res["metadata"]
            ids.append(self._str_to_uuid(metadata["passage_id"]))
            scores.append(res["score"])
        return ids, scores

    def delete_all(self):
        self.index.delete(namespace=self.namespace)

    def get_db_type(self) -> str:
        return "pinecone"

    def __test_embed(self, embedding: Embeddings) -> List[float]:
        test_query = "test"
        return embedding.embed_query(test_query)
