import os
import re
from typing import List, Iterator

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class MathpixMarkdownLoader(BaseLoader):
    """
    Load mathpix markdown file.
    mathpix markdown is .mmd file which is a markdown file for science papers.
    This class supports to split the file into sections and tables of science papers.
    """
    def __init__(self, filepath: str):
        """
        :param filepath: The path of mathpix markdown file. The file should be .mmd file.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} does not exist.")
        self.filepath = filepath

    def load(self, split_section: bool = True, split_table: bool = True) -> List[Document]:
        """
        :param split_section: If True, split the file into sections. Default is True.
        :param split_table: If True, split the file into tables. Default is True.
        :return: List of Document. If split_section and split_table are True, the list contains multiple Documents.
        The order of each section and table are the same as the order of the file.
        """
        return list(self.lazy_load(split_section=split_section, split_table=split_table))

    def lazy_load(self, split_section: bool = True, split_table: bool = True) -> Iterator[Document]:
        """
        :param split_section: If True, split the file into sections. Default is True.
        :param split_table: If True, split the file into tables. Default is True.
        :return: Iterator of Document. If split_section and split_table are True, return contains multiple Documents.
        The order of each section and table are the same as the order of the file.
        """
        with open(self.filepath, 'r') as f:
            content = f.read()

        if not split_section and not split_table:
            yield Document(page_content=content, metadata={"source": self.filepath, "content_type": "text"})
        else:
            split_sections: List[str] = [content]
            if split_section:
                split_sections = self.split_section(content)
                if not split_table:
                    for section in split_sections:
                        yield Document(page_content=section, metadata={"source": self.filepath, "content_type": "text"})

            if split_table:
                for document in split_sections:
                    contents: List[str] = self.split_table(document)
                    for content in contents:
                        page_type = "table" if content.startswith('\\\\begin{table}') else "text"
                        yield Document(page_content=content,
                                       metadata={"source": self.filepath, "content_type": page_type})

    @staticmethod
    def split_section(content: str) -> List[str]:
        """
        Split section from mathpix markdown content by '#'.
        """
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
