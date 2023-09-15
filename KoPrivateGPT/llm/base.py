from abc import ABC, abstractmethod
from typing import List

import openai

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class BaseLLM(ABC):
    def __init__(self, retrieval: BaseRetrieval):
        self.retrieval = retrieval
        self.retrieved_passages: List[Passage] = []
        self.chat_history: List[dict] = []
        self.chat_offset: int = 6

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

    @staticmethod
    def generate_chat(messages: List[dict], model: str,
                      stream: bool = False,
                      stream_func: callable = None,
                      *args, **kwargs) -> str:
        response = openai.ChatCompletion.create(*args, **kwargs,
                                                model=model,
                                                messages=messages,
                                                stream=stream)
        answer: str = ''
        if stream:
            for chunk in response:
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    stream_func(content)
                    answer += content
        else:
            answer = response["choices"][0]["message"]["content"]
        return answer

    @staticmethod
    def generate(prompt: str, model: str,
                 stream: bool = False,
                 stream_func: callable = None,
                 *args, **kwargs) -> str:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            stream=stream,
            *args, **kwargs
        )
        answer: str = ''
        if stream:
            for event in response:
                text = event['choices'][0]['text']
                stream_func(text)
                answer += text
        else:
            answer = response["choices"][0]["text"]
        return answer

    def add_chat_history(self, query: str, answer: str):
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

    def clear_chat_history(self):
        store_chat_history = self.chat_history
        self.chat_history.clear()
        return store_chat_history
