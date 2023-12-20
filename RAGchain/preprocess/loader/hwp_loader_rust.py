from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class HwpLoaderRust(BaseLoader):
    # It works any OS
    def __init__(self, path: str, *args, **kwargs):
        try:
            from libhwp import HWPReader
        except ImportError:
            raise ImportError("Please install libhwp."
                              "pip install libhwp")
        self.file_path = path
        self.result = []
        self.only_table = []
        self.hwp = HWPReader(self.file_path)

    def load(self) -> List[Document]:
        for paragraph in self.hwp.find_all('paragraph'):
            print(str(paragraph))
            self.result.append(str(paragraph))

        page = " ".join(self.result)
        print(page)
        document_list = [Document(page_content=page, metadata={"source": self.file_path})]
        return document_list

    def load_table(self) -> List[Document]:
        # just return paragraph in table
        for table in self.hwp.find_all('table'):
            for cell in table.cells:
                for paragraph in cell.paragraphs:
                    self.only_table.append(str(paragraph))

        table = " ".join(self.only_table)

        document_list = [Document(page_content=table, metadata={"source": self.file_path})]
        return document_list
