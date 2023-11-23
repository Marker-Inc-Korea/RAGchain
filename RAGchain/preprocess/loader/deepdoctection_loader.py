import requests
from urllib.parse import urljoin, urlencode
from pathlib import Path
import tempfile
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.schema import Document


class DeepdoctectionPDFLoader(BasePDFLoader):
    """
    Load PDF file using Deepdoctection API server.
    You can use Deepdoctection API server using Dockerfile at
    """
    def __init__(self, file_path: str, deepdoctection_host: str):
        super().__init__(file_path)
        # check connection of deepdoctection api server
        response = requests.get(deepdoctection_host)
        if response.status_code != 200:
            raise ValueError(f"Could not connect to Deepdoctection server: {deepdoctection_host}")
        self.deepdoctection_host = deepdoctection_host

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
        request_url = urljoin(self.deepdoctection_host, "extract/") + '?' + urlencode(kwargs)
        file = {
            'file': open(self.file_path, 'rb')
        }
        response = requests.post(request_url, files=file)
        if response.status_code != 200:
            raise ValueError(f'Deepdoctection API server returns {response.status_code} status code.')
        result = response.json()  # assuming the server returns a json
        for page_result in result:
            yield Document(page_result)  # assuming the Document class can take this dict as input
