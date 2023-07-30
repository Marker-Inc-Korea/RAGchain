from langchain.vectorstores import Chroma, Pinecone
from enum import Enum
from langchain.schema import Document
import pinecone
from typing import List
from dotenv import load_dotenv
import os
from options import ChromaOptions, PineconeOptions



class DBType(Enum):
    CHROMA = 'chroma'
    PINECONE = 'pinecone'


class DB:
    def __init__(self, db_type: str, embeddings):
        load_dotenv()
        if db_type in ['chroma', 'Chroma', 'CHROMA']:
            self.db_type = DBType.CHROMA
        elif db_type in ['pinecone', 'Pinecone', 'PineCone', 'PINECONE']:
            self.db_type = DBType.PINECONE
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV")
            )
        else:
            raise ValueError(f"Unknown db type: {db_type}")

        self.embeddings = embeddings
        self.db = self.load()

    def load(self):
        if self.db_type == DBType.CHROMA:
            return Chroma(persist_directory=ChromaOptions.persist_dir, embedding_function=self.embeddings,
                          client_settings=ChromaOptions.settings)
        elif self.db_type == DBType.PINECONE:
            return Pinecone(pinecone.Index(PineconeOptions.index_name), self.embeddings.embed_query, "text")

    def from_documents(self, docs: List[Document]):
        if self.db_type == DBType.CHROMA:
            result = Chroma.from_documents(docs, self.embeddings, persist_directory=ChromaOptions.persist_dir,
                                           client_settings=ChromaOptions.settings)
            result.persist()
            return result
        elif self.db_type == DBType.PINECONE:
            return Pinecone.from_documents(docs, self.embeddings, index_name=PineconeOptions.index_name)

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        return self.db.similarity_search(query=query, k=top_k)
