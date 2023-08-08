from typing import List
import pinecone
from langchain.schema import Document
from dotenv import load_dotenv
from typing import Optional, Dict, Union
from embed import Embedding
from vectorDB.base import BaseVectorDB
import os


class Pinecone(BaseVectorDB):
    def __init__(self, namespace: str, embedding: Embedding, *args, **kwargs):
        active_indexes = pinecone.list_indexes()
        index_name = str(embedding.embed_type)
        if index_name not in active_indexes:
            pinecone.create_index(index_name, dimension=len(self.__test_embed(embedding)), *args, **kwargs)
        if namespace not in pinecone.list_collections():
            pinecone.create_collection(namespace, index_name)

        self.index = pinecone.Index(index_name)
        self.namespace = namespace
        self.embedding = embedding.embedding()

    @classmethod
    def load(cls, namespace: str, embedding: Embedding):
        load_dotenv()
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        return cls(namespace, embedding)

    def add_documents(self, docs: List[Document]):
        self.index.upsert(
            vectors=[
                {'id': doc.metadata["id"],
                 'values': self.embedding.embed_query(doc.page_content),
                 'metadata': {
                     "page_content": doc.page_content,
                     "metadata": doc.metadata
                 }} for doc in docs
            ],
            namespace=self.namespace
        )

    def similarity_search(self, query: str, top_k: int = 5,
                          filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None) -> tuple[
        List[Document], List[float]]:

        embedded_query = self.embedding.embed_query(query)
        response = self.index.query(
            vector=embedded_query,
            namespace=self.namespace,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
        )
        docs = []
        scores = []
        for res in response["matches"]:
            metadata = res["metadata"]
            page_content = metadata.pop("page_content")
            docs.append(Document(page_content=page_content, metadata=metadata["metadata"]))
            scores.append(res["score"])
        return docs, scores

    def delete_all(self):
        self.index.delete(namespace=self.namespace)

    def get_db_type(self) -> str:
        return "pinecone"

    def __test_embed(self, embedding: Embedding) -> List[float]:
        test_query = "test"
        return embedding.embedding().embed_query(test_query)
