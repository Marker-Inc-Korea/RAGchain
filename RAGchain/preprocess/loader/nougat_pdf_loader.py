import tempfile
from pathlib import Path
from typing import List, Iterator
from urllib.parse import urljoin, urlencode

import requests
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.schema import Document

from RAGchain.preprocess.loader.mathpix_markdown_loader import MathpixMarkdownLoader


class NougatPDFLoader(BasePDFLoader):
    """
    Load PDF file using Nougat API server.
    You can use Nougat API server using Dockerfile at https://github.com/facebookresearch/nougat
    """
    def __init__(self, file_path: str, nougat_host: str):
        super().__init__(file_path)
        # check connection of nougat api server
        response = requests.get(nougat_host)
        if response.status_code != 200:
            raise ValueError(f"Could not connect to Nougat server: {nougat_host}")
        self.nougat_host = nougat_host

    def load(self, split_section: bool = True, split_table: bool = True, *args, **kwargs) -> List[Document]:
        """
        :param split_section: If True, split the document by section.
        :param split_table: If True, split the document by table.
        :param start: Start page number to load. Optional.
        :param stop: Stop page number to load. Optional.
        """
        return list(self.lazy_load(split_section=split_section, split_table=split_table, *args, **kwargs))

    def lazy_load(self, split_section: bool = True, split_table: bool = True, *args, **kwargs) -> Iterator[Document]:
        """
        :param split_section: If True, split the document by section.
        :param split_table: If True, split the document by table.
        :param start: Start page number to load. Optional.
        :param stop: Stop page number to load. Optional.
        """
        request_url = urljoin(self.nougat_host, "predict/") + '?' + urlencode(kwargs)
        file = {
            'file': open(self.file_path, 'rb')
        }
        response = requests.post(request_url, files=file)
        if response.status_code != 200:
            raise ValueError(f'Nougat API server returns {response.status_code} status code.')
        result = response.text
        result = result.replace('\\n', '\n')
        result = result[1:-1]  # remove first and last double quote

        with tempfile.NamedTemporaryFile() as temp_path:
            Path(temp_path.name).write_text(result)
            loader = MathpixMarkdownLoader(temp_path.name)
            for doc in loader.lazy_load(split_section=split_section, split_table=split_table):
                yield doc
