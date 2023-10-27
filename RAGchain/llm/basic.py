from typing import List, Callable

from RAGchain.llm.base import BaseLLM
from RAGchain.schema import Passage
from RAGchain.utils.util import set_api_base


class BasicLLM(BaseLLM):
    """
    Basic LLM module for question answering with retrieved passages.
    It supports stream and chat history features as default.

    :example:
    >>> from RAGchain.llm.basic import BasicLLM
    >>> from RAGchain.retrieval import BM25Retrieval

    >>> passages = [<your passages>]
    >>> llm = BasicLLM()
    >>> answer, passages = llm.ask("What is the purpose of this framework based on the document?", passages)
    >>> print(answer)

    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_base: str = None,
                 prompt_func: Callable[[List[Passage], str], List[dict]] = None,
                 stream_func: Callable[[str], None] = None,
                 *args, **kwargs):
        """
        Initializes an instance of the BasicLLM class.

        :param model_name: The name or identifier of the llm model to be used. Default is "gpt-3.5-turbo".
        :param api_base: The base URL of the llm API endpoint. Default is None.
        :param prompt_func: A callable function used for generating prompts based on passages and user query. The input of prompt_func will be the list of retrieved passages and user query. The output of prompt_func should be a list of dictionaries with "role" and "content" keys, which is openai style chat prompts. Default is BasicLLM.get_messages.
        :param stream_func: A callable function used for streaming generated responses. You have to implement if you want to use stream. This stream_func will be called when the stream is received. Default is None.
        """
        super().__init__()
        self.model_name = model_name
        set_api_base(api_base)
        self.get_message = self.get_messages if prompt_func is None else prompt_func
        self.stream_func = stream_func

    def ask(self, query: str, passages: List[Passage], stream: bool = False,
            *args, **kwargs) -> tuple[str, List[Passage]]:
        """
        Ask a question to the LLM model and get answer and used passages.
        :param query: question
        :param passages: passages to use for answering the question
        :param stream: if stream is true, use stream feature. Default is False.
        :param args: optional parameter for llm api (openai style)
        :param kwargs: optional parameter for llm api (openai style)

        :return answer: The answer to the question that llm generated.
        :return passages: The list of passages used to generate the answer.
        """
        answer = self.generate_chat(messages=self.chat_history[-self.chat_offset:] + self.get_message(passages, query),
                                    model=self.model_name,
                                    stream=stream,
                                    stream_func=self.stream_func, *args, **kwargs)
        self.add_chat_history(query, answer)
        return answer, passages

    @staticmethod
    def get_messages_ko(passages: List[Passage], question: str) -> List[dict]:
        """
        Korean example of prompt_func. It generates openai style chat prompts based on passages and user query.
        """
        context_str = "\n\n".join([passage.content for passage in passages])
        system_prompt = f"""주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
                    질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요."""
        user_prompt = f"""정보 : 
            {context_str}

            질문: {question}
            """
        main_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "다음은 질문에 대한 한국어 답변입니다. "}
        ]
        return main_messages

    @staticmethod
    def get_messages(passages: List[Passage], question: str) -> List[dict]:
        """
        Example of prompt_func. It generates openai style chat prompts based on passages and user query.
        """
        context_str = "\n\n".join([passage.content for passage in passages])
        system_prompt = f"""Given the information, answer the question. If you don't know the answer, don't make up 
        the answer, just say you don't know."""
        user_prompt = f"""Information : 
            {context_str}

            Question: {question}
            """
        main_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "The following is the answer to the question. "}
        ]
        return main_messages
