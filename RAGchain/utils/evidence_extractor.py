from typing import List

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel

from RAGchain.schema import Passage

# This prompt is originated from RETA-LLM
BASIC_SYSTEM_PROMPT = """From the given document, please select and output the relevant document fragments which are related to the query.
Note that the output must be fragments of the original document, not a summary of the document. 
If there is no fragment related to the query in the document, please output 'No Fragment'.
"""


class EvidenceExtractor:
    """
    EvidenceExtractor is a class that extracts relevant evidences based on a given question and a list of passages.

    :example:
    >>> from RAGchain.utils.evidence_extractor import EvidenceExtractor
    >>> from RAGchain.schema import Passage
    >>> from langchain.llms.openai import OpenAI
    >>>
    >>> passages = [
    ...     Passage(content="Lorem ipsum dolor sit amet"),
    ...     Passage(content="Consectetur adipiscing elit"),
    ...     Passage(content="Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua")
    ... ]
    >>>
    >>> question = "What is Lorem ipsum?"
    >>> extractor = EvidenceExtractor(OpenAI())
    >>> result = extractor.extract(question, passages)

    >>> print(result)
    """

    def __init__(self, llm: BaseLanguageModel, system_prompt: str = None):
        """
        Initialize the EvidenceExtractor class.

        :param llm: The language model to be used for evidence extraction. You can use both Chat and Completion models.
        :param system_prompt: The system prompt to be used. If not provided, the default system prompt will be used.
        """
        self.llm = llm
        self.system_prompt = system_prompt if system_prompt is not None else BASIC_SYSTEM_PROMPT

    def extract(self, question: str, passages: List[Passage]) -> str:
        """
        Extract method extracts relevant document evidences based on a question and a list of passages.

        :param question: The question for which relevant document fragments need to be extracted.
        :param passages: A list of Passage objects that contain the content of the documents.

        :return: The extracted relevant document fragments.
        """
        content_str = "\n".join([passage.content for passage in passages])
        runnable = self.__make_prompt() | self.llm | StrOutputParser()
        answer = runnable.invoke({
            "question": question,
            "content_str": content_str,
        })
        return answer

    def __make_prompt(self):
        if isinstance(self.llm, BaseLLM):
            return PromptTemplate.from_template(
                self.system_prompt +
                "Document content: {content_str}\n\nquery: {question}]\n\nrelevant document fragments:"
            )
        elif isinstance(self.llm, BaseChatModel):
            return ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "Document content: {content_str}\n\nquery: {question}"),
                ("ai", "relevant document fragments: ")
            ])
        else:
            raise NotImplementedError("Only support LLM or ChatModel")
