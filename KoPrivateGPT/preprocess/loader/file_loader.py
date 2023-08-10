import os
from typing import List
from tqdm import tqdm
from langchain.schema import Document
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader

from KoPrivateGPT.options import Options


class FileLoader:
    def __init__(self, target_dir: str):
        # add more extensions when if you want to add more extensions loader
        self.ingestable_extensions = ['.txt', '.pdf', '.csv', '.xlsx', '.hwp']
        if not os.path.exists(target_dir):
            raise ValueError(f"Target directory {target_dir} does not exist.")
        self.target_dir = target_dir

    def load(self, filter_ext: List[str] = None) -> List[Document]:
        """
        Load all files in the target directory.
        Parameters:
            filter_ext: List[str] = None
                If not None, only files with the given extensions will be loaded.
                filter_ext elements must contain the dot (.) prefix.
        """
        valid_ext = self.ingestable_extensions if filter_ext is None else filter_ext
        docs = []
        for (path, dir, files) in tqdm(os.walk(self.target_dir)):
            for file_name in files:
                ext = os.path.splitext(file_name)[-1].lower()  # this function contain dot (.) prefix
                if filter_ext is not None and ext not in filter_ext:
                    continue
                full_file_path = os.path.join(path, file_name)
                if ext in valid_ext:
                    docs.append(self._load_single_document(full_file_path))
                else:
                    print(f"Not Support file type {ext} yet.")
        return docs

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
            loader = HwpLoader(file_path, hwp_host_url=Options.HwpConvertHost)

        return loader.load()[0]
