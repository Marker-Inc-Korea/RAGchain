from typing import List, Union
from uuid import UUID

from langchain.embeddings.base import Embeddings

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage, Vector
from KoPrivateGPT.utils.embed import delete_embeddings_vectordb
from KoPrivateGPT.utils.vectorDB.base import BaseVectorDB


class VectorDBRetrieval(BaseRetrieval):
    def __init__(self, vectordb: BaseVectorDB, embedding: Embeddings, *args, **kwargs):
        super().__init__()
        self.vectordb = vectordb
        self.embedding = embedding

    def ingest(self, passages: List[Passage]):
        embeds = self.embedding.embed_documents([passage.content for passage in passages])
        self.vectordb.add_vectors(
            [Vector(passage_id=passage.id, vector=embed) for passage, embed in zip(passages, embeds)])

    def delete_all(self):
        delete_embeddings_vectordb(self.vectordb.get_db_type())

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        ids = self.retrieve_id(query, top_k)
        passage_list = self.fetch_data(ids)
        return passage_list

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        query_vector = self.embedding.embed_query(query)
        ids, scores = self.vectordb.similarity_search(query_vector, top_k)
        return ids
