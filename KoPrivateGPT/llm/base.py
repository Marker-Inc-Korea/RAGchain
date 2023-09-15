from abc import ABC, abstractmethod
from typing import List

import openai

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class BaseLLM(ABC):
    stream_end_token: str = '<|endofstream|>'

    def __init__(self, retrieval: BaseRetrieval):
        self.retrieval = retrieval
        self.retrieved_passages: List[Passage] = []

    @abstractmethod
    def ask(self, query: str, stream: bool = False, run_retrieve: bool = True, *args, **kwargs) -> tuple[
        str, List[Passage]]:
        """
        Ask a question to the LLM model and get answer and used passages.
        *args, **kwargs is optional parameter for openai api llm
        """
        pass

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        """
        Retrieve passages from the retrieval module
        """
        self.retrieved_passages = self.retrieval.retrieve(query, top_k, *args, **kwargs)
        return self.retrieved_passages

    @classmethod
    def generate_chat(cls, messages: List[dict], model: str,
                      stream: bool = False,
                      stream_func: callable = None,
                      *args, **kwargs) -> str:
        """
        If stream is true, run stream_func for each response.
        And return cls.stream_end_token when stream is end.
        """
        response = openai.ChatCompletion.create(*args, **kwargs,
                                                model=model,
                                                messages=messages,
                                                stream=stream)
        answer: str = ''
        if stream:
            for chunk in response:
                if len(chunk["choices"]) > 0:
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content is not None:
                        stream_func(content)
                        answer += content
            stream_func(cls.stream_end_token)
        else:
            answer = response["choices"][0]["message"]["content"]
        return answer

    @classmethod
    def generate(cls, prompt: str, model: str,
                 stream: bool = False,
                 stream_func: callable = None,
                 *args, **kwargs) -> str:
        """
        If stream is true, run stream_func for each response.
        And return cls.stream_end_token when stream is end.
        """
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            stream=stream,
            *args, **kwargs
        )
        answer: str = ''
        if stream:
            for event in response:
                if len(event['choices']) > 0:
                    text = event['choices'][0]['text']
                    stream_func(text)
                    answer += text
            stream_func(cls.stream_end_token)
        else:
            answer = response["choices"][0]["text"]
        return answer
