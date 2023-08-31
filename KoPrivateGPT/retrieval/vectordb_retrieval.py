import os
from typing import List, Union
from uuid import UUID

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings

from KoPrivateGPT.options import ChromaOptions, PineconeOptions
from KoPrivateGPT.utils.embed import delete_embeddings_vectordb
from KoPrivateGPT.utils.util import text_modifier
from KoPrivateGPT.utils.vectorDB import Chroma
from KoPrivateGPT.utils.vectorDB import Pinecone
from .base import BaseRetrieval
from ..DB.base import BaseDB
from ..schema import Passage
from ..schema.vector import Vector


class VectorDBRetrieval(BaseRetrieval):
    def __init__(self, vectordb_type: str, embedding: Embeddings, *args, **kwargs):
        if vectordb_type in text_modifier('chroma'):
            self.vectordb = Chroma(ChromaOptions.persist_dir, ChromaOptions.collection_name)
        elif vectordb_type in text_modifier('pinecone', modify_words=['PineCone']):
            load_dotenv()
            self.vectordb = Pinecone(api_key=os.getenv('PINECONE_API_KEY'),
                                     environment=os.getenv('PINECONE_ENV'),
                                     index_name=PineconeOptions.index_name,
                                     namespace=PineconeOptions.namespace,
                                     dimension=PineconeOptions.dimension)
        else:
            raise ValueError(f"Unknown db type: {vectordb_type}")
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
