import tempfile
from typing import List

import requests
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class PdfLinkLoader(BaseLoader):
    """
    Load PDF from a link
    """

    def __init__(self, link: str, *args, **kwargs):
        if not self.valid_url(link):
            raise ValueError(f"Invalid url: {link}")
        self.link = link

    def load(self) -> List[Document]:
        with tempfile.NamedTemporaryFile() as f:
            f.write(requests.get(self.link).content)
            f.seek(0)
            loader = PDFMinerLoader(f.name)
            return loader.load()

    @staticmethod
    def valid_url(url):
        return url.startswith("http://") or url.startswith("https://")
