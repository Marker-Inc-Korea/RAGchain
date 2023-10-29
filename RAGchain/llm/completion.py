from typing import List, Callable

from RAGchain.llm.base import BaseLLM
from RAGchain.schema import Passage
from RAGchain.utils.util import set_api_base


class CompletionLLM(BaseLLM):
    """
    This LLM module generate question with original completion API. Not using chat models.
    So, it does not support chat history feature.
    """

    def __init__(self, model_name: str = "text-davinci-003", api_base: str = None,
                 prompt_func: Callable[[List[Passage], str], str] = None,
                 stream_func: Callable[[str], None] = None
                 ):
        """
        Initializes an instance of the CompletionLLM class.

        :param model_name: The name or identifier of the llm model to be used. Default is "text-davinci-003".
        :param api_base: The base URL of the llm API endpoint. Default is None.
        :param prompt_func: A callable function used for generating prompts based on passages and user query. The input of prompt_func will be the list of retrieved passages and user query. The output of prompt_func should be a string. Default is CompletionLLM.get_messages.
        :param stream_func: A callable function used for streaming generated responses. You have to implement if you want to use stream.
        This stream_func will be called when the stream is received. Default is None.
        """
        super().__init__()
        self.model_name = model_name
        set_api_base(api_base)
        self.prompt_func = prompt_func if prompt_func is not None else self.get_messages
        self.stream_func = stream_func

    def ask(self, query: str, passages: List[Passage], stream: bool = False,
            *args, **kwargs) -> tuple[str, List[Passage]]:
        answer = self.generate(self.prompt_func(passages, query),
                               self.model_name,
                               stream=stream,
                               stream_func=self.stream_func, *args, **kwargs)
        return answer, passages

    @staticmethod
    def get_messages(passages: List[Passage], question: str) -> str:
        """
        Default prompt for CompletionLLM.
        """
        context_str = "\n\n".join([passage.content for passage in passages])
        return f"""
        Given the information, answer the question. If you don't know the answer, don't make up 
        the answer, just say you don't know.
        
        Information:
        {context_str}
        
        Question:
        {question}
        
        Answer:
        """
