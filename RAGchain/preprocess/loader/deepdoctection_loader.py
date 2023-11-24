from typing import List, Iterator

import requests
from urllib.parse import urljoin, urlencode
from pathlib import Path
import tempfile
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.schema import Document

import re


def _check_deepdoctection_server(deepdoctection_host: str):
    response = requests.get(deepdoctection_host)
    if response.status_code != 200:
        raise ValueError(f"Could not connect to Deepdoctection server: {deepdoctection_host}")


class DeepdoctectionPDFLoader(BasePDFLoader):
    """
    Load PDF file using Deepdoctection API server.
    """

    def __init__(self, file_path: str, deepdoctection_host: str):
        super().__init__(file_path)
        _check_deepdoctection_server(deepdoctection_host)
        self.deepdoctection_host = deepdoctection_host

    def load(self, split_section: bool = True, split_table: bool = True, *args, **kwargs) -> List[Document]:
        return list(self.lazy_load(*args, **kwargs))

    def lazy_load(self, *args, **kwargs) -> Iterator[Document]:
        request_url = urljoin(self.deepdoctection_host, "extract/") + '?' + urlencode(kwargs)
        with open(self.file_path, 'rb') as file:
            file_upload = {'file': file}
            response = requests.post(request_url, files=file_upload)
        if response.status_code != 200:
            raise ValueError(f'Deepdoctection API server returns {response.status_code} status code.')
        result = response.json()
        # [title + text] should be made into one document
        extracted_pages = []
        last_title = None
        for i, item in enumerate(result):
            titles = item['title']
            text = item['text']
            page_number = item['page_number']
            # Find the positions of each title in the text
            positions = [(title, pos) for title in titles for pos in self.find_positions(text, title)]
            positions.sort(key=lambda x: x[1])
            # If there are no titles in this page, use the last title from the previous page
            if not titles:
                if last_title:
                    extracted_page = {'Title': last_title, 'Text': text.strip(),
                                      'PageNumber': page_number}
                    extracted_pages.append(extracted_page)
            else:
                # If there is a last title, create a new document with the last title and the text
                # before the first title of the current page
                if last_title is not None:
                    extracted_pages.append({
                        'Title': last_title,
                        'Text': text[:positions[0][1]].strip(),
                        'PageNumber': page_number
                    })
                # Create a new extracted page for each title in the current page
                for j in range(len(positions)):
                    title, start = positions[j]
                    if j == len(positions) - 1:
                        end = len(text)
                    else:
                        end = positions[j + 1][1]
                    txt = text[start:end].replace(title, '', 1).strip()
                    extracted_page = {'Title': title, 'Text': txt,
                                      'PageNumber': page_number}
                    extracted_pages.append(extracted_page)
                # Update last_title to the last title of the current page if there are titles,
                # otherwise keep the last title
                last_title = positions[-1][0] if positions else last_title
        for extracted_page in extracted_pages:
            page_content = extracted_page['Title'] + '\n' + extracted_page['Text']
            metadata = extracted_page['PageNumber']
            yield Document(page_content=page_content, emetadata=metadata)

    @staticmethod
    def find_positions(text, substring):
        positions = [match.start() for match in re.finditer(re.escape(substring), text)]
        return positions
