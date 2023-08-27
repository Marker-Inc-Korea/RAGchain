from typing import List, Union
from uuid import UUID

from langchain.embeddings.base import Embeddings

from KoPrivateGPT.utils.embed import delete_embeddings_vectordb
from .base import BaseRetrieval
from ..DB.base import BaseDB
from ..schema import Passage
from ..schema.vector import Vector
from ..utils.vectorDB.base import BaseVectorDB


class VectorDBRetrieval(BaseRetrieval):
    def __init__(self, vectordb: BaseVectorDB, embedding: Embeddings, *args, **kwargs):
        self.vectordb = vectordb
        self.embedding = embedding

    def ingest(self, passages: List[Passage]):
        embeds = self.embedding.embed_documents([passage.content for passage in passages])
        self.vectordb.add_vectors(
            [Vector(passage_id=passage.id, vector=embed) for passage, embed in zip(passages, embeds)])

    def delete_all(self):
        delete_embeddings_vectordb(self.vectordb.get_db_type())

    def retrieve(self, query: str, db: BaseDB, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        ids = self.retrieve_id(query, top_k)
        result = db.fetch(ids)
        return result

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        query_vector = self.embedding.embed_query(query)
        ids, scores = self.vectordb.similarity_search(query_vector, top_k)
        return ids
