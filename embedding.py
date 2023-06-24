from enum import Enum
from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

class EMBEDDINGType(Enum):
    OPENAI = 'openai'
    HUGGINGFACE = 'hugging_face'


class EMBEDDING():
    def __init__(self, embed_type: str):
        load_dotenv()
        if embed_type in ['OpenAI', 'openai', 'OPENAI', 'Openai']:
            self.embed_type = EMBEDDINGType.OPENAI

        elif embed_type in ['HuggingFace', 'huggingface', 'Huggingface', 'hugging_face', 'Hugging_face', 'huggingFace', 'HuggingFace', 'huggingFace', 'Huggingface']:
            self.embed_type = EMBEDDINGType.HUGGINGFACE
        else:
            raise ValueError(f"Unknown embedding type: {embed_type}")

    def embedding(self):
        if self.embed_type == EMBEDDINGType.OPENAI:
            return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif self.embed_type == EMBEDDINGType.HUGGINGFACE:
            return HuggingFaceInstructEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask",
                                          model_kwargs={"device": "cpu"})
        else:
            raise ValueError(f"Unknown embedding type: {self.embed_type}")
