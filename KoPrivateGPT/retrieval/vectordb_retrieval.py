from typing import List

from langchain.schema import Document
from KoPrivateGPT.embed import delete_embeddings_vectordb
from KoPrivateGPT.options import ChromaOptions, PineconeOptions
from KoPrivateGPT.vectorDB import Pinecone
from KoPrivateGPT.vectorDB.chroma import Chroma
from KoPrivateGPT.vectorDB.base import BaseVectorDB
from .base import BaseRetrieval
from KoPrivateGPT.embed import Embedding
from ..DB.base import BaseDB
from ..schema import Passage
from ..schema.vector import Vector


class VectorDBRetrieval(BaseRetrieval):
    def __init__(self, vectordb_type: str, embedding: Embedding, db: BaseDB):
        if vectordb_type in ['chroma', 'Chroma', 'CHROMA']:
            self.vectordb = Chroma.load(ChromaOptions.persist_dir, ChromaOptions.collection_name)
        elif vectordb_type in ['pinecone', 'Pinecone', 'PineCone', 'PINECONE']:
            self.vectordb = Pinecone.load(PineconeOptions.index_name, PineconeOptions.namespace,
                                          PineconeOptions.dimension)
        else:
            raise ValueError(f"Unknown db type: {vectordb_type}")
        self.embedding = embedding.embedding()
        self.db = db

    def ingest(self, passages: List[Passage]):
        embeds = self.embedding.embed_documents([Document(page_content=passage.content) for passage in passages])
        self.vectordb.add_vectors(
            [Vector(passage_id=passage.id, vector=embed) for passage, embed in zip(passages, embeds)])

    def delete_all(self):
        delete_embeddings_vectordb(self.vectordb.get_db_type())

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        query_vector = self.embedding.embed_query(query)
        ids, scores = self.vectordb.similarity_search(query_vector, top_k)
        result = self.db.fetch(ids)
        return result
