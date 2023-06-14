import logging
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
#from langchain.document_loaders.helpers import detect_file_encodings
import requests

logger = logging.getLogger(__name__)


class HwpLoader(BaseLoader):
    """Load Hwp files.

    Hwp to text  using hwp-converter-api

    and just use textLoader

    Args:
        file_path: Path to the file to load.

        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.

        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
    """

    def __init__(
        self,
        file_path: str,
        hwp_convert_path: str,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,

    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
        self.hwp_convert_path = hwp_convert_path

        r = requests.post(hwp_convert_path, data = {'file': open(file_path, 'rb')})

        if r.status_code != 200:
            raise ValueError(
                "Check the url of your file; returned status code %s"
                % r.status_code
            )

        self.temp_text = r.content

    def load(self) -> List[Document]:
        """Load from file path."""
        text = ""
        try:
            with self.temp_text as f:
                text = f.read()
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]