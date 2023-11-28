from typing import List, Iterator, Dict, Any

import requests
from urllib.parse import urljoin, urlencode
from pathlib import Path
import tempfile
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.schema import Document

import re


class DeepdoctectionPDFLoader(BasePDFLoader):
    """
    Load PDF file using NomaDamas' Deepdoctection API server.
    You can use Deepdoctection API server using Dockerfile at https://github.com/NomaDamas/deepdoctection-api-server
    """

    def __init__(self, file_path: str, deepdoctection_host: str):
        super().__init__(file_path)
        response = requests.get(deepdoctection_host)
        if response.status_code != 200:
            raise ValueError(f"Could not connect to Deepdoctection server: {deepdoctection_host}")
        self.deepdoctection_host = deepdoctection_host

    def load(self, *args, **kwargs) -> List[Document]:
        """
        load pdf file using Deepdoctection API server
        return list of Document
        """
        return list(self.lazy_load(*args, **kwargs))

    def lazy_load(self, *args, **kwargs) -> Iterator[Document]:
        """
        lazy_load pdf file using Deepdoctection API server
        return list of Document
        """
        request_url = urljoin(self.deepdoctection_host, "extract/") + '?' + urlencode(kwargs)
        with open(self.file_path, 'rb') as file:
            file_upload = {'file': file}
            response = requests.post(request_url, files=file_upload)
        if response.status_code != 200:
            raise ValueError(f'Deepdoctection API server returns {response.status_code} status code.')
        result = response.json()
        extracted_pages = self.extract_pages(result)
        for extracted_page in extracted_pages:
            if 'table' in extracted_page:
                yield Document(page_content=extracted_page['table'],
                               metadata={'Page_number': extracted_page['page_number']})
            else:
                page_content = extracted_page['title'] + '\n' + extracted_page['text']
                metadata = {'Page_number': extracted_page['page_number']}
                yield Document(page_content=page_content, metadata=metadata)

    def extract_pages(self, result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted_pages = []
        last_title = None
        for i, item in enumerate(result):
            titles = item['title']
            text = item['text']
            page_number = item['page_number']
            table = item['table']
            # If there is a table, extract the table and add it to the extracted pages
            for tbl in table:
                extracted_pages.append({'table': tbl, 'page_number': page_number})
            # Find the positions of each title in the text
            positions = [(title, pos) for title in titles for pos in self.find_positions(text, title)]
            positions.sort(key=lambda x: x[1])
            # If there are no titles in this page, use the last title from the previous page
            if not titles:
                if last_title:
                    extracted_page = {'title': last_title, 'text': text.strip(),
                                      'page_number': page_number}
                    extracted_pages.append(extracted_page)
                else:
                    extracted_page = {'title': '', 'text': text.strip(),
                                      'page_number': page_number}
                    extracted_pages.append(extracted_page)
            else:
                # If there is a last title, create a new document with the last title and the text
                # before the first title of the current page
                if last_title is not None:
                    extracted_pages.append({
                        'title': last_title,
                        'text': text[:positions[0][1]].strip(),
                        'page_number': page_number
                    })
                # Create a new extracted page for each title in the current page
                for j in range(len(positions)):
                    title, start = positions[j]
                    if j == len(positions) - 1:
                        end = len(text)
                    else:
                        end = positions[j + 1][1]
                    txt = text[start:end].replace(title, '', 1).strip()
                    extracted_page = {'title': title, 'text': txt,
                                      'page_number': page_number}
                    extracted_pages.append(extracted_page)
                # Update last_title to the last title of the current page if there are titles,
                # otherwise keep the last title
                last_title = positions[-1][0]
        return extracted_pages

    @staticmethod
    def find_positions(text, substring):
        positions = [match.start() for match in re.finditer(re.escape(substring), text)]
        return positions
