import os
import re
from typing import List, Iterator

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class MathpixMarkdownLoader(BaseLoader):
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} does not exist.")
        self.filepath = filepath

    def load(self, split_section: bool = True, split_table: bool = True) -> List[Document]:
        with open(self.filepath, 'r') as f:
            content = f.read()
        result = [Document(page_content=content, metadata={"source": self.filepath, "content_type": "text"})]

        if split_section:
            section_result = []
            split_sections = self.split_section(content)
            for section in split_sections:
                section_result.append(Document(page_content=section, metadata={"source": self.filepath,
                                                                               "content_type": "text"}))
            result = section_result

        if split_table:
            table_result = []
            for document in result:
                contents = self.split_table(document.page_content)
                for i, content in enumerate(contents):
                    if i % 2 == 0:
                        table_result.append(Document(page_content=content, metadata={"source": self.filepath,
                                                                                     "content_type": "text"}))
                    else:
                        table_result.append(Document(page_content=content, metadata={"source": self.filepath,
                                                                                     "content_type": "table"}))
            result = table_result
        return result

    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError

    @staticmethod
    def split_section(content: str) -> List[str]:
        split_text = re.split('(#+ )', content)
        split_text.pop(0)
        result = [split_text[i] + split_text[i + 1] for i in range(0, len(split_text), 2)]
        return result

    @staticmethod
    def split_table(content: str) -> List[str]:
        """
        Split table from mathpix markdown content.
        :param content: mathpix markdown content
        :return: The odd index is the content without table, and the even index is the table.
        """
        pattern = re.compile(r'\\\\begin{table}.*?\\\\end{table}', re.DOTALL)
        matches = re.findall(pattern, content)
        texts_without_tables = re.split(pattern, content)
        result = []
        for i in range(len(texts_without_tables)):
            result.append(texts_without_tables[i])
            if i < len(matches):
                result.append(matches[i])
        return result
