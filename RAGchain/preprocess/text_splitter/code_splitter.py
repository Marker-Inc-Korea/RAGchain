import copy
from typing import List
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


class CodeSplitter(BaseTextSplitter):
    def __init__(
            self,
            language_name: str = 'PYTHON',
            chunk_size: int = 50,
            chunk_overlap: int = 0,
            **kwargs
    ):
        """
        :param language_name: A kind of language to split. Default is PYTHON.
        (CPP, GO, JAVA, KOTLIN, JS, TS, PHP, PROTO, PYTHON, RST, RUBY, RUST, SCALA, SWIFT, MARKDOWN, LATEX, HTML, SOL, CSHARP)
        :param chunk_size: Maximum size of chunks to return. Default is 50.
        :param chunk_overlap: Overlap in characters between chunks. Default is 0.
        :param kwargs: Additional arguments to pass to the langchain RecursiveCharacterTextSplitter.
        """

        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language[language_name], chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )

    def split_document(self, document: Document) -> List[Passage]:
        split_documents = self.code_splitter.create_documents([document.page_content])

        # Convert List[Document] to List[Passage]
        document_copy = copy.deepcopy(document)
        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]

        filepath = document_copy.metadata.pop('source')  # user doc's metadata value.

        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            # Modify meta_data's keys and values right form.
            metadata_etc = dict(split_document.metadata.copy(),
                                **document_copy.metadata, )  # metadata_etc = doc's metadata_etc + splitter's metadata.

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
