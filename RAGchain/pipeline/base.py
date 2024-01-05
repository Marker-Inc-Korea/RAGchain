from abc import ABC, abstractmethod
from typing import Optional, List, Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import Runnable

from RAGchain.schema import Passage, RAGchainPromptTemplate, RAGchainChatPromptTemplate


class BaseIngestPipeline(ABC):
    """
    Base class for all pipelines
    """

    def __init__(self):
        self.run: Optional[Runnable] = None
        self._make_runnable()
        if self.run is None:
            raise NotImplementedError("You should implement __make_runnable method")

    @abstractmethod
    def _make_runnable(self):
        """initialize runnable"""
        pass


class BaseRunPipeline(ABC):
    default_prompt = RAGchainPromptTemplate.from_template(
        """
        Given the information, answer the question. If you don't know the answer, don't make up 
        the answer, just say you don't know.

        Information :
        {passages}

        Question: {question}

        Answer:
        """
    )

    default_chat_prompt = RAGchainChatPromptTemplate.from_messages(
        [
            ("system", "Given the information, answer the question. If you don't know the answer, don't make up the "
                       "answer, just say you don't know."
                       "Information : \n{passages}"),
            ("human", "Question: {question}"),
            ("ai", "Answer: ")
        ]
    )

    def __init__(self):
        self.run: Optional[Runnable] = None
        self._make_runnable()
        if self.run is None:
            raise NotImplementedError("You should implement __make_runnable method")

    @abstractmethod
    def _make_runnable(self):
        """initialize runnable"""
        pass

    @abstractmethod
    def get_passages_and_run(self, questions: List[str], top_k: int = 5) -> tuple[
        List[str], List[List[Passage]], List[List[float]]]:
        """
        Run the pipeline for evaluator, and get retrieved passages and rel scores.
        It is same with pipeline.run.batch, but returns passages and rel scores.
        Return List of answer, List of passages, Relevance score of passages.

        :param questions: List of questions.
        :param top_k: The number of passages to retrieve.
        It is the same as retrieval_options top_k.
        Default is 5.
        """
        pass

    def _get_default_prompt(self, llm: BaseLanguageModel,
                            prompt: Optional[Union[RAGchainPromptTemplate, RAGchainChatPromptTemplate]] = None,
                            default_prompt: Optional[RAGchainPromptTemplate] = None,
                            default_chat_prompt: Optional[RAGchainChatPromptTemplate] = None):
        if default_prompt is None:
            default_prompt = self.default_prompt
        if default_chat_prompt is None:
            default_chat_prompt = self.default_chat_prompt

        if prompt is None:
            if isinstance(llm, BaseLLM):
                prompt = default_prompt
            elif isinstance(llm, BaseChatModel):
                prompt = default_chat_prompt
            else:
                raise NotImplementedError("RAGchain only supports types that are either BaseLLM or BaseChatModel.")
        return prompt
