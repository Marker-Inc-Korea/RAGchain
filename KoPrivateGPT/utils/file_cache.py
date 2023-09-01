from typing import List

from langchain.schema import Document

from KoPrivateGPT.DB.base import BaseDB


class FileCache:
    def __init__(self, db: BaseDB):
        self.db = db
        self.db.create_or_load()

    def delete_duplicate(self, documents: List[Document]) -> List[Document]:
        for document in documents.copy():
            result = self.db.search({'filepath': document.metadata['source']})
            if len(result) > 0:
                documents.remove(document)
        return documents
