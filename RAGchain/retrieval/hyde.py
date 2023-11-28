import logging
from typing import List, Union
from uuid import UUID

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel

from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage

logger = logging.getLogger(__name__)


class HyDERetrieval(BaseRetrieval):
    """
    HyDE Retrieval, which inspired by "Precise Zero-shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496.pdf)
    At retrieval, LLM model creates hypothetical passage.
    And then, retrieve passages using hypothetical passage as query.
    """
    BASIC_SYSTEM_PROMPT = "Please write a passage to answer the question"

    def __init__(self, retrieval: BaseRetrieval, llm: BaseLanguageModel,
                 system_prompt: str = None, *args, **kwargs):
        """
        :param retrieval: retrieval instance to use
        :param llm: llm to use for hypothetical passage generation. HyDE Retrieval supports both chat and completion LLMs.
        :param system_prompt: system prompt to use when generating hypothetical passage
        """
        super().__init__()
        self.retrieval = retrieval
        self.llm = llm
        self.system_prompt = self.BASIC_SYSTEM_PROMPT if system_prompt is None else system_prompt
        prompt = self.__make_prompt()
        self.runnable = prompt | self.llm | StrOutputParser()

    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> \
            List[Passage]:
        ids = self.retrieve_id(query, top_k, *args, **kwargs)
        result = self.retrieval.fetch_data(ids)
        return result

    def ingest(self, passages: List[Passage]):
        self.retrieval.ingest(passages)

    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[
        Union[str, UUID]]:
        ids, scores = self.retrieve_id_with_scores(query, top_k, *args, **kwargs)
        return ids

    def retrieve_id_with_scores(self, query: str, top_k: int = 5, *args, **kwargs) -> \
            tuple[List[Union[str, UUID]], List[float]]:
        hyde_answer = self.runnable.invoke({"question": query})
        # logging
        logger.info(f"HyDE answer : {hyde_answer}")
        return self.retrieval.retrieve_id_with_scores(query=hyde_answer.strip(), top_k=top_k, *args, **kwargs)

    def delete(self, ids: List[Union[str, UUID]]):
        self.retrieval.delete(ids)

    def __make_prompt(self):
        if isinstance(self.llm, BaseLLM):
            return PromptTemplate.from_template(
                self.system_prompt + "\nQuestion: {question}\nPassage:"
            )
        elif isinstance(self.llm, BaseChatModel):
            return ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "Question: {Question}"),
                ("ai", "Passage: ")
            ])
        else:
            raise NotImplementedError("Only support LLM or ChatModel")
