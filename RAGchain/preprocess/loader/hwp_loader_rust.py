import os
from typing import List
from libhwp import HWPReader

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class HwpLoaderRust(BaseLoader):
    # It works any OS
    def __init__(self, path: str, *args, **kwargs):
        self.file_path = path
        self.result = []
        self.only_table = []

    def load(self) -> List[Document]:
        hwp = HWPReader(self.file_path)

        for paragraph in hwp.find_all('paragraph'):
            print(str(paragraph))
            self.result.append(str(paragraph))

        page = " ".join(self.result)
        print(page)
        document_list = [Document(page_content=page, metadata={"source": self.file_path})]
        return document_list

    def load_table(self) -> List[Document]:
        # just return paragraph in table

        hwp = HWPReader(self.file_path)

        for table in hwp.find_all('table'):
            for cell in table.cells:
                for paragraph in cell.paragraphs:
                    self.only_table.append(str(paragraph))

        table = " ".join(self.only_table)

        document_list = [Document(page_content=table, metadata={"source": self.file_path})]
        return document_list
