from abc import ABC, abstractmethod
from typing import List

import openai

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class BaseLLM(ABC):
    def __init__(self, retrieval: BaseRetrieval):
        self.retrieval = retrieval
        self.retrieved_passages: List[Passage] = []

    @abstractmethod
    def ask(self, query: str, stream: bool = False, run_retrieve: bool = True) -> tuple[str, List[Passage]]:
        """
        Ask a question to the LLM model and get answer and used passages
        """
        pass

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        """
        Retrieve passages from the retrieval module
        """
        self.retrieved_passages = self.retrieval.retrieve(query, top_k, *args, **kwargs)
        return self.retrieved_passages

    def generate_chat(self, messages: List[dict], model: str,
                      stream: bool = False,
                      stream_func: callable = None,
                      *args, **kwargs) -> str:
        return self.__generate(openai.ChatCompletion,
                               messages, model, stream, stream_func, *args, **kwargs)

    def generate(self, messages: List[dict], model: str,
                 stream: bool = False,
                 stream_func: callable = None,
                 *args, **kwargs) -> str:
        return self.__generate(openai.Completion,
                               messages, model, stream, stream_func, *args, **kwargs)

    @staticmethod
    def __generate(completion,
                   messages: List[dict], model: str,
                   stream: bool = False,
                   stream_func: callable = None,
                   *args, **kwargs) -> str:
        response = completion.create(*args, **kwargs,
                                     model=model,
                                     messages=messages,
                                     stream=stream)
        answer: List[str] = []
        if stream:
            for chunk in response:
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    stream_func(content)
                    answer.append(content)
        else:
            answer = response["choices"][0]["message"]["content"]
        return ''.join(answer)
