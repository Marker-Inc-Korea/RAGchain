from typing import List, Union
from uuid import UUID

from langchain.schema import Document
from langchain.vectorstores import VectorStore

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class VectorDBRetrieval(BaseRetrieval):
    def __init__(self, vectordb: VectorStore, *args, **kwargs):
        super().__init__()
        self.vectordb = vectordb

    def ingest(self, passages: List[Passage]):
        self.vectordb.add_documents(
            [Document(page_content=passage.content, metadata={'passage_id': str(passage.id)}) for passage in passages])

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        ids = self.retrieve_id(query, top_k)
        passage_list = self.fetch_data(ids)
        return passage_list

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        docs = self.vectordb.similarity_search(query=query, k=top_k)
        return [self.__str_to_uuid(doc.metadata.get('passage_id')) for doc in docs]

    @staticmethod
    def __str_to_uuid(input_str: str) -> Union[str, UUID]:
        try:
            return UUID(input_str)
        except:
            return input_str
