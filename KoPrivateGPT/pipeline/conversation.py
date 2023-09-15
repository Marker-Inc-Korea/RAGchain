from abc import ABC, abstractmethod

from typing import List


class ConversationPipeline(ABC):
    @abstractmethod
    def run(self, query: str, chat_history: List, *args, **kwargs):
        pass

    @staticmethod
    def transform_chat_history(chat_history: List[tuple[str, str]]) -> List[dict]:
        """
        [('test_query1', 'test_answer1'), ('test_query2', 'test_answer2')]
        -> [{"role": "user", "content": "test_query1"}, {"role": "assistant", "content": "test_answer1"},
        {"role": "user", "content": "test_query2"}, {"role": "assistant", "content": "test_answer2"}]
        """
        output_list = []
        for item in chat_history:
            output_list.append({"role": "user", "content": item[0]})
            output_list.append({"role": "assistant", "content": item[1]})
        return output_list
