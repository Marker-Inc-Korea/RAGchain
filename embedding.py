from enum import Enum
from dotenv import load_dotenv
import os


class EmbeddingType(Enum):
    OPENAI = 'openai'
    KOSIMCSE = 'kosimcse'


class Embedding:
    def __init__(self, embed_type: str, device_type: str = 'cuda'):
        load_dotenv()
        if embed_type in ['OpenAI', 'openai', 'OPENAI', 'Openai']:
            self.embed_type = EmbeddingType.OPENAI

        elif embed_type in ['KoSimCSE', 'kosimcse', 'KOSIMCSE', 'Kosimcse']:
            self.embed_type = EmbeddingType.KOSIMCSE
        else:
            raise ValueError(f"Unknown embedding type: {embed_type}")

        if device_type in ['cpu', 'CPU']:
            self.device_type = 'cpu'
        elif device_type in ['mps', 'MPS']:
            self.device_type = 'mps'
        else:
            self.device_type = 'cuda'

    def embedding(self):

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
            try:
                from langchain.embeddings import HuggingFaceEmbeddings
            except ImportError:
                raise ModuleNotFoundError(
                    "Could not import HuggingFaceEmbeddings library. Please install HuggingFace library."
                    "pip install sentence_transformers"
                )
            from langchain.embeddings import HuggingFaceInstructEmbeddings
            return HuggingFaceInstructEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask",
                                                 model_kwargs={"device": self.device_type})
        else:
            raise ValueError(f"Unknown embedding type: {self.embed_type}")
