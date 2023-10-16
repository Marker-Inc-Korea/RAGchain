from typing import List
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


# 마크다운 헤더로 자르는 splitter
class MarkDownHeaderSplitter(BaseTextSplitter):
    def __init__(self, headers_to_split_on: List[tuple[str, str]], return_each_line: bool = False):
        """
        :param headers_to_split_on: A list of tuples which appended  to create split standard.
        ex)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        """
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line)

    def split_document(self, documents: Document):
        split_documents = self.markdown_splitter.split_text(documents.page_content)

        # Modify meta_data's keys and values.
        test_meta = dict(documents.metadata, **split_documents[0].metadata.copy())

        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]
        filepath = documents.metadata.pop('source')  # user doc's metadata value.
        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            split = split_document.metadata.copy()
            metadata_etc = dict(documents.metadata,
                                **split_document.metadata.copy())  # metadata_etc = doc's metadata_etc + headers
            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(split_documents) - 1 else None
            passage = Passage(id=uuid,
                              content=split_document.page_content,
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)
            passages.append(passage)
        print(f"Split into {len(passages)} passages")

        return passages

