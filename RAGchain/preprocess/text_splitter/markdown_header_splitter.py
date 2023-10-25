import copy
from typing import List, Optional
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class MarkDownHeaderSplitter(BaseTextSplitter):
    def __init__(self, headers_to_split_on: Optional[List[tuple[str, str]]] = None, return_each_line: bool = False):
        """
        :param headers_to_split_on: A list of tuples which appended to create split standard.
        ex)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        """

        # Set default value headers_to_split_on.
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ]

        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line)

    def split_document(self, document: Document):
        doc_copy = copy.deepcopy(document)
        split_documents = self.markdown_splitter.split_text(document.page_content)


        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]

        filepath = doc_copy.metadata.pop('source')  # user doc's metadata value.
        doc_metadata_etc = doc_copy.metadata  # TEST_DOCUMENT's metadata etc.(Already removed file path data)

        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            # Modify meta_data's keys and values right form.
            ## metadata_etc = doc's metadata_etc + headers
            metadata_etc = dict(split_document.metadata.copy(),  # Header information
                                **doc_metadata_etc, )  # TEST_DOCUMENT's metadata etc

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
