from typing import List

from langchain.schema import Document

from RAGchain.DB.base import BaseDB


class FileCache:
    """
    This class is used to delete duplicate documents from given DB.
    You can use this after you load your file to Document using file loader.
    It will automatically check duplicate documents using source metadata and return non-duplicate documents.

    :example:
    >>> from RAGchain.utils.file_cache import FileCache
    >>> from RAGchain.DB import PickleDB
    >>> from langchain.document_loaders import TextLoader
    >>>
    >>> db = PickleDB(save_path='./pickle_db.pkl')
    >>> file_cache = FileCache(db)
    >>> documents = TextLoader('./data.txt').load()
    >>> documents = file_cache.delete_duplicate(documents)
    """
    def __init__(self, db: BaseDB):
        self.db = db
        self.db.create_or_load()

    def delete_duplicate(self, documents: List[Document]) -> List[Document]:
        for document in documents.copy():
            result = self.db.search(filepath=[document.metadata['source']])
            if len(result) > 0:
                documents.remove(document)
        return documents
