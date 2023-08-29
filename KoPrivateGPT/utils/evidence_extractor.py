from typing import List, Optional

import openai

from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import set_api_base

# This prompt is originated from RETA-LLM
BASIC_SYSTEM_PROMPT = """From the given document, please select and output the relevant document fragments which are related to the query.
Note that the output must be fragments of the original document, not a summary of the document. 
If there is no fragment related to the query in the document, please output nothing.
"""


class EvidenceExtractor:
    def __init__(self, system_prompt: str = None, model_name: str = "gpt-3.5-turbo", api_base: str = None):
        self.system_prompt = system_prompt if system_prompt is not None else BASIC_SYSTEM_PROMPT
        self.model_name = model_name
        set_api_base(api_base)

    def extract(self, question: str, passages: List[Passage], model_kwargs: Optional[dict]) -> str:
        content_str = "\n".join([passage.content for passage in passages])
        user_prompt = f"Document content: {content_str}\n\nquery: {question}]\n\nrelevant document fragments:"
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            message=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **model_kwargs
        )
        return completion["choices"][0]["message"]["content"]
