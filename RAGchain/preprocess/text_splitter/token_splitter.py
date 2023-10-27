import copy
from typing import List
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import (CharacterTextSplitter, TokenTextSplitter, SpacyTextSplitter,
                                     SentenceTransformersTokenTextSplitter, NLTKTextSplitter)
from transformers import AutoTokenizer

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage
from RAGchain.utils.util import text_modifier


class TokenSplitter(BaseTextSplitter):
    """
    Split a document into passages by recursively splitting on a list of separators.
    You can specify a window_size and overlap_size to split the document into overlapping passages.
    """

    def __init__(self, tokenizer_name: str = 'tiktoken', chunk_size: int = 100, chunk_overlap: int = 0,
                 pretrained_model_name: str = "gpt2", **kwargs):
        """
        :param tokenizer_name: A tokenizer_name name. You can choose tokenizer_name.
                        (tiktoken, spaCy, SentenceTransformers, NLTK, huggingFace)
        :param chunk_size: Maximum size of chunks to return. Default is 100.
        :param chunk_overlap: Overlap in characters between chunks. Default is 0.
        :param pretrained_model_name: A huggingface tokenizer pretrained_model_name to use huggingface token splitter.
                                      You can choose various pretrained_model_name in this parameter. Default is "gpt2".
                                      Refer to pretrained model in this link.  (https://huggingface.co/models)
        :param kwargs: Additional arguments.

        All splitters were inherited TextSplitter class in langchain text_splitter.py.
        """

        # Create token splitter according to chosen_tokenizer.
        if 'tiktoken' in text_modifier(tokenizer_name):
            self.splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif 'spaCy' in text_modifier(tokenizer_name):
            self.splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif 'SentenceTransformers' in text_modifier(tokenizer_name):
            self.splitter = SentenceTransformersTokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif 'NLTK' in text_modifier(tokenizer_name):
            self.splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif 'huggingFace' in text_modifier(tokenizer_name):
            tokenizers = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.splitter = CharacterTextSplitter.from_huggingface_tokenizer(
                tokenizers, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError("Ooops! You input invalid tokenizer name." + " Your input: " + tokenizer_name)



    def split_document(self, document: Document) -> List[Passage]:
        """
        Split a document.
        """

        split_text = self.splitter.split_text(document.page_content)

        # Convert split text to split documents.
        split_documents = self.splitter.create_documents(split_text)

        # Convert to documents to passages.
        doc_copy = copy.deepcopy(document)
        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]

        filepath = doc_copy.metadata.pop('source')  # user doc's metadata value.
        doc_metadata_etc = doc_copy.metadata  # TEST_DOCUMENT's metadata etc.(Already removed file path data)

        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            # Modify meta_data's keys and values right form.
            ## metadata_etc = doc's metadata_etc + splitter's information.
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

