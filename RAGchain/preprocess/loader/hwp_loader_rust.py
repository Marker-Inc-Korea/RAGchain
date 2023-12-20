from typing import List, Iterator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class HwpLoaderRust(BaseLoader):
    # It works any OS
    def __init__(self, path: str):
        try:
            from libhwp import HWPReader
        except ImportError:
            raise ImportError("Please install libhwp."
                              "pip install libhwp")
        self.file_path = path
        self.result = []
        self.only_table = []
        self.hwp = HWPReader(self.file_path)

    def lazy_load(self) -> Iterator[Document]:
        for paragraph in self.hwp.find_all('paragraph'):
            self.result.append(str(paragraph))

        page = "\n\n".join(self.result)
        yield Document(page_content=page, metadata={"source": self.file_path, 'page_type': 'text'})
        yield self.__load_table()

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def __load_table(self) -> Document:
        # just return paragraph in table
        for table in self.hwp.find_all('table'):
            for cell in table.cells:
                for paragraph in cell.paragraphs:
                    self.only_table.append(str(paragraph))

        table = ",".join(self.only_table)

        return Document(page_content=table, metadata={"source": self.file_path, 'page_type': 'table'})
