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
                 huggingFace_tokenizer: str = "gpt2", **kwargs):
        """
        :param tokenizer_name: A tokenizer_name name. You can choose tokenizer_name.
                        (tiktoken, spaCy, SentenceTransformers, NLTK, huggingFace)
        :param chunk_size: Maximum size of chunks to return. Default is 0.
        :param chunk_overlap: Overlap in characters between chunks. Default is 0.
        :param huggingFace_tokenizer: A huggingface tokenizer model to use huggingface token splitter.
                                      You can choose various huggingface tokenizer in this parameter. Default is "gpt2".
                                      Refer to pretrained model in this link.  (https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)
        :param kwargs: Additional arguments.
        All splitters were inherited TextSplitter class in langchain text_splitter.py.
        """
        self.chosen_tokenizer = tokenizer_name

        # tiktoken
        self.tiktoken_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size = chunk_size, chunk_overlap =  chunk_overlap)

        # spaCy
        self.spaCy_splitter = SpacyTextSplitter(chunk_size= chunk_size, chunk_overlap= chunk_overlap)

        # SentenceTransformers (Default: chunk_overlap=0)
        self.sentence_transformer_splitter = SentenceTransformersTokenTextSplitter(chunk_size=chunk_size,
                                                                                   chunk_overlap=chunk_overlap)

        # NLTK
        self.NLTK_splitter = NLTKTextSplitter(chunk_size= chunk_size, chunk_overlap= chunk_overlap)
        # Hugging Face
        tokenizers = AutoTokenizer.from_pretrained(huggingFace_tokenizer)
        self.huggingFace_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizers, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_document(self, document: Document) -> List[Passage]:
        """
        Split a document.
        """
        global chosen_splitter

        if 'tiktoken' in text_modifier(self.chosen_tokenizer):
            chosen_splitter = self.tiktoken_splitter.split_text(document.page_content)
        elif 'spaCy' in text_modifier(self.chosen_tokenizer):
            chosen_splitter = self.spaCy_splitter.split_text(document.page_content)
        elif 'SentenceTransformers' in text_modifier(self.chosen_tokenizer):
            chosen_splitter = self.sentence_transformer_splitter.split_text(document.page_content)
        elif 'NLTK' in text_modifier(self.chosen_tokenizer):
            chosen_splitter = self.NLTK_splitter.split_text(document.page_content)
        elif 'huggingFace' in text_modifier(self.chosen_tokenizer):
            chosen_splitter = self.huggingFace_splitter.split_text(document.page_content)

        # Create split text to split documents.
        split_documents = self.tokenzier_create_documents(document, chosen_splitter)

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

    def tokenzier_create_documents(self, document: Document, chosen_splitter):
        split_documents = self.tiktoken_splitter.create_documents(chosen_splitter)
        return split_documents
