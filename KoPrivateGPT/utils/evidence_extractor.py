from typing import List

import openai

from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import set_api_base

# This prompt is originated from RETA-LLM
BASIC_SYSTEM_PROMPT = """From the given document, please select and output the relevant document fragments which are related to the query.
Note that the output must be fragments of the original document, not a summary of the document. 
If there is no fragment related to the query in the document, please output 'No Fragment'.
"""


class EvidenceExtractor:
    """
    EvidenceExtractor is a class that extracts relevant evidences based on a given question and a list of passages.

    :example:
    >>> from KoPrivateGPT.utils.evidence_extractor import EvidenceExtractor
    >>> from KoPrivateGPT.schema import Passage
    >>>
    >>> passages = [
    ...     Passage(content="Lorem ipsum dolor sit amet"),
    ...     Passage(content="Consectetur adipiscing elit"),
    ...     Passage(content="Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua")
    ... ]
    >>>
    >>> question = "What is Lorem ipsum?"
    >>> extractor = EvidenceExtractor()
    >>> result = extractor.extract(question, passages)

    >>> print(result)
    """
    def __init__(self, system_prompt: str = None, model_name: str = "gpt-3.5-turbo", api_base: str = None):
        """
        Initialize the EvidenceExtractor class.

        :param system_prompt: The system prompt to be used. If not provided, the default system prompt will be used.
        :param model_name: The name of the model to be used. The default model is "gpt-3.5-turbo".
        :param api_base: The base URL for the custom model. Default is None.
        """
        self.system_prompt = system_prompt if system_prompt is not None else BASIC_SYSTEM_PROMPT
        self.model_name = model_name
        set_api_base(api_base)

    def extract(self, question: str, passages: List[Passage], model_kwargs: dict = {}) -> str:
        """
        Extract method extracts relevant document evidences based on a question and a list of passages.

        :param question: The question for which relevant document fragments need to be extracted.
        :param passages: A list of Passage objects that contain the content of the documents.
        :param model_kwargs: Optional keyword arguments to be passed to the openai api for customization. Default is {}.

        :return: The extracted relevant document fragments.
        """
        content_str = "\n".join([passage.content for passage in passages])
        user_prompt = f"Document content: {content_str}\n\nquery: {question}]\n\nrelevant document fragments:"
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **model_kwargs
        )
        return completion["choices"][0]["message"]["content"]
