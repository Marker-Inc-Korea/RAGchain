from langchain.vectorstores import Chroma, Pinecone
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY, PINECONE_INDEX_NAME
from enum import Enum
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import pinecone
from typing import List
from dotenv import load_dotenv
import os

class DBType(Enum):
    CHROMA = 'chroma'
    PINECONE = 'pinecone'


class DB():
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

    def load(self):
        if self.db_type == DBType.CHROMA:
            return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=self.embeddings, client_settings=CHROMA_SETTINGS)
        elif self.db_type == DBType.PINECONE:
            return Pinecone(pinecone.Index(PINECONE_INDEX_NAME), self.embeddings.embed_query, "text")

    def from_documents(self, docs: List[Document]):
        if self.db_type == DBType.CHROMA:
            result = Chroma.from_documents(docs, self.embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
            result.persist()
            return result
        elif self.db_type == DBType.PINECONE:
            return Pinecone.from_documents(docs, self.embeddings, index_name=PINECONE_INDEX_NAME)
