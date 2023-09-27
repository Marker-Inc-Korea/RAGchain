import tempfile
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urlencode

import requests
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.schema import Document

from KoPrivateGPT.preprocess.loader.mathpix_markdown_loader import MathpixMarkdownLoader


class NougatPDFLoader(BasePDFLoader):
    def __init__(self, file_path: str, nougat_host: str):
        super().__init__(file_path)
        # check connection of nougat api server
        response = requests.get(nougat_host)
        if response.status_code != 200:
            raise ValueError(f"Could not connect to Nougat server: {nougat_host}")
        self.nougat_host = nougat_host

    def load(self, *args, **kwargs) -> List[Document]:
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
            return loader.load()
