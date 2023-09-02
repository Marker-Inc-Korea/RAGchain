import logging
from typing import List

import requests
from langchain.docstore.document import Document

from KoPrivateGPT.preprocess.loader.base import BaseLoader

logger = logging.getLogger(__name__)


class HwpLoader(BaseLoader):
    """Load Hwp files.

    Hwp to text  using hwp-converter-api

    and just use textLoader

    Args:
        path: Path to the file to load.
    """

    def __init__(
            self,
            path: str,
            hwp_host_url: str,
            retry_connection: int = 4
    ):
        """Initialize with file path."""
        self.file_path = path
        self.hwp_convert_path = hwp_host_url

        assert retry_connection >= 1
        retry_cnt = 0
        while True:
            if retry_cnt >= retry_connection:
                break
            response = requests.post(hwp_host_url, files={'file': open(path, 'rb')})
            if response.status_code == 200:
                break
            retry_cnt += 1

        if response.status_code != 200:
            raise ValueError(
                "Check the url of your file; returned status code %s"
                % response.status_code
            )

        self.temp_response = response

    def load(self) -> List[Document]:
        """Load from response."""
        text = ""
        try:
            with self.temp_response as r:
                text = r.content.decode(r.apparent_encoding)
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
