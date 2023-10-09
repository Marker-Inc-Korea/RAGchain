from typing import List, Union
from uuid import UUID

from langchain.schema import Document
from langchain.vectorstores import VectorStore

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.vectorstore.base import SlimVectorStore


class VectorDBRetrieval(BaseRetrieval):
    """
    VectorDBRetrieval is a retrieval class that uses VectorDB as a backend.
    First, embed the passage content using embedding model.
    Then, store the embedded vector in VectorDB.
    When retrieving, embed the query and search the most similar vectors in VectorDB.
    Lastly, return the passages that have the most similar vectors.
    """
    def __init__(self, vectordb: VectorStore, *args, **kwargs):
        """
        :param vectordb: VectorStore instance. You can all langchain VectorStore classes, also you can use SlimVectorStore for better storage efficiency.
        """
        super().__init__()
        self.vectordb = vectordb

    def ingest(self, passages: List[Passage]):
        if isinstance(self.vectordb, SlimVectorStore):
            self.vectordb.add_passages(passages)
        else:
            self.vectordb.add_documents(
                [Document(page_content=passage.content, metadata={'passage_id': str(passage.id)}) for passage in
                 passages])

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        ids = self.retrieve_id(query, top_k)
        passage_list = self.fetch_data(ids)
        return passage_list

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        docs = self.vectordb.similarity_search(query=query, k=top_k)
        return [self.__str_to_uuid(doc.metadata.get('passage_id')) for doc in docs]

    def retrieve_id_with_scores(self, query: str, top_k: int = 5, *args, **kwargs) -> tuple[
        List[Union[str, UUID]], List[float]]:
        results = self.vectordb.similarity_search_with_score(query=query, k=top_k)
        results = results[::-1]
        docs = [result[0] for result in results]
        scores = [result[1] for result in results]
        return [self.__str_to_uuid(doc.metadata.get('passage_id')) for doc in docs], scores

    @staticmethod
    def __str_to_uuid(input_str: str) -> Union[str, UUID]:
        try:
            return UUID(input_str)
        except:
            return input_str
