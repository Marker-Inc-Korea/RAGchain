import uuid
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
        index_name = embedding.embed_type.value
        if index_name not in active_indexes:
            pinecone.create_index(index_name, dimension=len(self.__test_embed(embedding)), *args, **kwargs)
        self.index = pinecone.Index(index_name)
        self.namespace = namespace
        self.embedding = embedding.embedding()

    @classmethod
    def load(cls, namespace: str, embedding: Embedding):
        load_dotenv()
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENV"]
        )
        return cls(namespace, embedding)

    def add_documents(self, docs: List[Document]):
        vectors = []
        for doc in docs:
            if 'id' not in list(doc.metadata.keys()):
                id = str(uuid.uuid4())
            else:
                id = doc.metadata["id"]
            _metadata = doc.metadata
            _metadata['page_content'] = doc.page_content
            vectors.append(
                {'id': id,
                 'values': self.embedding.embed_query(doc.page_content),
                 'metadata': _metadata}
            )
        self.index.upsert(vectors=vectors, namespace=self.namespace)

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
            docs.append(Document(page_content=page_content, metadata=metadata))
            scores.append(res["score"])
        return docs, scores

    def delete_all(self):
        self.index.delete(namespace=self.namespace)

    def get_db_type(self) -> str:
        return "pinecone"

    def __test_embed(self, embedding: Embedding) -> List[float]:
        test_query = "test"
        return embedding.embedding().embed_query(test_query)
