from typing import List

from langchain.schema import Document
from langchain.text_splitter import (CharacterTextSplitter, TokenTextSplitter,
                                     SentenceTransformersTokenTextSplitter)
from transformers import AutoTokenizer

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage
from RAGchain.utils.util import text_modifier


class TokenSplitter(BaseTextSplitter):
    """
    The `TokenSplitter` is used to split a document into passages by token using various tokenization methods.
    It's designed to split text from a document into smaller chunks, or "tokens", using various tokenization methods.
    The class supports tokenization with 'tiktoken', 'spaCy', 'SentenceTransformers', 'NLTK', and 'huggingFace'.
    """

    def __init__(self, tokenizer_name: str = 'tiktoken', chunk_size: int = 100, chunk_overlap: int = 0,
                 pretrained_model_name: str = "gpt2", **kwargs):
        """
        :param tokenizer_name: A tokenizer_name is a name of tokenizer. You can choose tokenizer_name.
                        (tiktoken, spaCy, SentenceTransformers, NLTK, huggingFace)
        :param chunk_size: Maximum size of chunks to return. Default is 100.
        :param chunk_overlap: Overlap in characters between chunks. Default is 0.
        :param pretrained_model_name: A huggingface tokenizer pretrained_model_name to use huggingface token splitter.
                                      You can choose various pretrained_model_name in this parameter. Default is "gpt2".
                                      Refer to pretrained model in this link.  (https://huggingface.co/models)
        :param kwargs: Additional arguments.
        """

        # Create token splitter according to chosen_tokenizer.
        if 'tiktoken' in text_modifier(tokenizer_name):
            self.splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        elif 'spaCy' in text_modifier(tokenizer_name):
            from langchain.text_splitter import SpacyTextSplitter
            self.splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        elif 'SentenceTransformers' in text_modifier(tokenizer_name):
            self.splitter = SentenceTransformersTokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        elif 'NLTK' in text_modifier(tokenizer_name):
            from langchain.text_splitter import NLTKTextSplitter
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
        split_documents = self.splitter.split_documents([document])
        passages = self.docs_to_passages(split_documents)
        return passages
