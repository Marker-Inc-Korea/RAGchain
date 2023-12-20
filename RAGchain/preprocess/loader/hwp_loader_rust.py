from typing import List, Iterator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class HwpLoaderRust(BaseLoader):
    """
    Load HWP file using libhwp. It works for any os.
    Using load or lazy_load, you can get list of Documents from hwp file.
    This loader loads all paragraphs and tables from hwp file.
    At the first Document, there are all paragraphs from hwp file, including texts in each table.
    Next, there are separated Documents for each table paragraphs.
    Unfortunately, You can't distinguish row and columns in table.

    In the metadata, there are filepath at key 'source' and page_type, which is 'text' or 'table'.

    Recommend to use other hwp loader, but it is great option to use this loader at mac and linux.
    It is no need to use external hwp loader server, or hwp program only available at windows.
    """
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
        paragraph = " ".join([str(paragraph) for paragraph in self.hwp.find_all('paragraph')])
        yield Document(page_content=paragraph, metadata={"source": self.file_path, 'page_type': 'text'})

        for table in self.hwp.find_all('table'):
            table_contents = []
            for cell in table.cells:
                for paragraph in cell.paragraphs:
                    table_contents.append(str(paragraph))
            yield Document(page_content=",".join(table_contents),
                           metadata={"source": self.file_path, 'page_type': 'table'})

    def load(self) -> List[Document]:
        return list(self.lazy_load())
