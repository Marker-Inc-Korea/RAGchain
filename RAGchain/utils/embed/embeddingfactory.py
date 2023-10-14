import os
from enum import Enum

from RAGchain.utils.util import text_modifier


class EmbeddingType(Enum):
    OPENAI = 'openai'
    KOSIMCSE = 'kosimcse'
    KO_SROBERTA_MULTITASK = 'ko-sroberta-multitask'
    MULTILINGUAL_E5 = 'multilingual-e5'
    CONTRIEVER = 'contriever'


class EmbeddingFactory:
    """
    EmbeddingFactory is a factory class that returns the embedding class according to the embedding type.
    You can create embedding class easily by using this class.
    """
    def __init__(self, embed_type: str, device_type: str = 'cuda'):
        """
        :param embed_type: Embedding type. You can choose one of the following types.
        - openai: OpenAI GPT-3
        - kosimcse: KoSimCSE
        - ko_sroberta_multitask: Ko-SRoBERTa-MultiTask
        - multilingual_e5: Multilingual-E5
        - contriever: mContriever
        :param device_type: Device type. You can choose one of the following types.
        - cuda: GPU
        - cpu: CPU
        - mps: MPS
        """
        if embed_type in text_modifier('openai'):
            self.embed_type = EmbeddingType.OPENAI

        elif embed_type in text_modifier('kosimcse',
                                         modify_words=['KoSimCSE', 'KoSimcse', 'koSimCSE', 'kosimCSE']):
            self.embed_type = EmbeddingType.KOSIMCSE
        elif embed_type in text_modifier('ko_sroberta_multitask'):
            self.embed_type = EmbeddingType.KO_SROBERTA_MULTITASK
        elif embed_type in text_modifier('multilingual_e5'):
            self.embed_type = EmbeddingType.MULTILINGUAL_E5
        elif embed_type in text_modifier('contriever'):
            self.embed_type = EmbeddingType.CONTRIEVER
        else:
            raise ValueError(f"Unknown embedding type: {embed_type}")

        if device_type in text_modifier('cpu'):
            self.device_type = 'cpu'
        elif device_type in text_modifier('mps'):
            self.device_type = 'mps'
        else:
            self.device_type = 'cuda'

    def get(self):
        """
        Returns the embedding class according to the embedding type.
        """
        if self.embed_type == EmbeddingType.OPENAI:
            openai_token = os.getenv("OPENAI_API_KEY")
            if openai_token is None:
                raise ValueError("OPENAI_API_KEY is empty.")
            try:
                from langchain.embeddings import OpenAIEmbeddings
            except ImportError:
                raise ModuleNotFoundError(
                    "Could not import OpenAIEmbeddings library. Please install OpenAI library."
                    "pip install openai"
                )
            return OpenAIEmbeddings(openai_api_key=openai_token)

        elif self.embed_type == EmbeddingType.KOSIMCSE:
            return self.__set_huggingface_embeddings("BM-K/KoSimCSE-roberta-multitask",
                                                     {"device": self.device_type})
        elif self.embed_type == EmbeddingType.KO_SROBERTA_MULTITASK:
            return self.__set_huggingface_embeddings("jhgan/ko-sroberta-multitask",
                                                     {"device": self.device_type})
        elif self.embed_type == EmbeddingType.MULTILINGUAL_E5:
            return self.__set_huggingface_embeddings("intfloat/multilingual-e5-large",
                                                     {"device": self.device_type})
        elif self.embed_type == EmbeddingType.CONTRIEVER:
            return self.__set_huggingface_embeddings("facebook/mcontriever-msmarco",
                                                     {"device": self.device_type})
        else:
            raise ValueError(f"Unknown embedding type: {self.embed_type}")

    @staticmethod
    def __set_huggingface_embeddings(model_name: str, model_kwargs: dict):
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import HuggingFaceEmbeddings library. Please install HuggingFace library."
                "pip install sentence_transformers"
            )
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        return HuggingFaceEmbeddings(model_name=model_name,
                                     model_kwargs=model_kwargs)
