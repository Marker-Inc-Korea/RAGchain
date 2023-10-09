import os
from typing import List, Iterator

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from tqdm import tqdm


class FileLoader(BaseLoader):
    """
    Loads documents from a directory.
    You can load .txt, .pdf, .csv, .xlsx, .hwp files.
    """
    def __init__(self, target_dir: str, hwp_host_url: str, *args, **kwargs):
        """
        :param target_dir: directory path to load documents from
        :param hwp_host_url: hwp-converter-api host url
        """
        # add more extensions when if you want to add more extensions loader
        self.hwp_host_url = hwp_host_url
        self.ingestable_extensions = ['.txt', '.pdf', '.csv', '.xlsx', '.hwp']
        if not os.path.exists(target_dir):
            raise ValueError(f"Target directory {target_dir} does not exist.")
        self.target_dir = target_dir

    def load(self, filter_ext: List[str] = None) -> List[Document]:
        """
        Load all files in the target directory.
        :param filter_ext: If not None, only files with the given extensions will be loaded. filter_ext elements must contain the dot (.) prefix.
        """
        docs = list(self.lazy_load(filter_ext=filter_ext))
        if len(docs) <= 0:
            print(f"Could not find any new documents in {self.target_dir}")
        else:
            print(f"Loaded {len(docs)} documents from {self.target_dir}")
        return docs

    def lazy_load(self, filter_ext: List[str] = None) -> Iterator[Document]:
        """
        Lazily load all files in the target directory.
        :param filter_ext: If not None, only files with the given extensions will be loaded. filter_ext elements must contain the dot (.) prefix.
        """
        valid_ext = self.ingestable_extensions if filter_ext is None else filter_ext
        for (path, dir, files) in tqdm(os.walk(self.target_dir)):
            for file_name in files:
                ext = os.path.splitext(file_name)[-1].lower()  # this function contain dot (.) prefix
                if filter_ext is not None and ext not in filter_ext:
                    continue
                full_file_path = os.path.join(path, file_name)
                if ext in valid_ext:
                    yield self._load_single_document(full_file_path)
                else:
                    print(f"Not Support file type {ext} yet.")

    def _load_single_document(self, file_path: str) -> Document:
        from KoPrivateGPT.preprocess.loader import ExcelLoader, HwpLoader
        # Loads a single document from a file path
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf8")
        elif file_path.endswith(".pdf"):
            loader = PDFMinerLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(".xlsx"):
            loader = ExcelLoader(file_path)
        elif file_path.endswith(".hwp"):
            loader = HwpLoader(file_path, hwp_host_url=self.hwp_host_url)

        return loader.load()[0]
