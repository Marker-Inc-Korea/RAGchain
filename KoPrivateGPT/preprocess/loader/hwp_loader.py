import asyncio
import logging
from http.client import HTTPException
from typing import List, Iterator

import aiohttp
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class HwpLoader(BaseLoader):
    """Load Hwp files.

    Hwp to text using hwp-converter-api.
    You can use hwp-converter-api at https://github.com/NomaDamas/hwp-converter-api
    """

    def __init__(
            self,
            path: str,
            hwp_host_url: str,
            retry_connection: int = 4
    ):
        """
        :param path: Path to the file. You must use .hwp file. .hwpx file is not supported.
        :param hwp_host_url: URL of hwp-converter-api.
        :param retry_connection: Number of retries to connect to hwp-converter-api. Default is 4.
        """
        self.path = path
        self.hwp_host_url = hwp_host_url

        assert retry_connection >= 1
        self.retry_connection = retry_connection

    def load(self) -> List[Document]:
        """Load a document."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """
        Load a document lazily. This method uses asyncio requests.
        """
        response = asyncio.run(self.async_request())
        yield Document(page_content=response, metadata={"source": self.path})

    async def async_request(self):
        for _ in range(self.retry_connection):
            async with aiohttp.ClientSession() as session:
                async with session.post(self.hwp_host_url, data={'file': open(self.path, 'rb')}) as response:
                    if response.status == 200:
                        return await response.text()
        raise HTTPException(
            f"Check the url of your file; returned status code {response.status}"
        )
